#!/usr/bin/env python3
"""Step 04: native TDX upstream trace plus READ-ONLY shared gauge-topology analysis.

Usage (one region-index build, then read-only basin runs):
  python 1_delineation/tdx/04_trace_tdx_upstream.py --control CONTROL --build-topology-index --index-only
  python 1_delineation/tdx/04_trace_tdx_upstream.py --control CONTROL --require-topology-index

Source TDX GeoPackages are never modified. Shared network *relationships* are
assessed, not applied: no geometry cutting, polygon partition or fractional
reach IDs are produced by Step 04. Standalone controls without a shared registry
continue to work. The regional SQLite index can be shared by many Slurm tasks.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import sqlite3
import sys
import tempfile
import tomllib
from collections import Counter, deque
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyproj import CRS
from shapely.geometry import Point

INDEX_VERSION = 'tdx-step04-regional-index-v1'
PROJECT = Path(__file__).resolve().parents[2]


def norm(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ('none', 'nan', 'null'):
        return None
    return s[:-2] if s.endswith('.0') and s[:-2].lstrip('-').isdigit() else s


def ident(value):
    return '"' + str(value).replace('"', '""') + '"'


def open_ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def layer_info(con, layer):
    row = con.execute('SELECT column_name,srs_id FROM gpkg_geometry_columns WHERE table_name=?', (layer,)).fetchone()
    if row is None:
        raise ValueError(f'No geometry layer metadata for {layer}')
    geom_col, srs_id = row
    cols = {r[1] for r in con.execute(f'PRAGMA table_info({ident(layer)})')}
    if 'fid' not in cols:
        raise ValueError(f'Expected fid primary key in {layer}')
    crs = CRS.from_epsg(srs_id) if srs_id and srs_id > 0 else None
    if crs is None:
        raise ValueError(f'Unknown CRS for layer {layer} (srs_id={srs_id})')
    return geom_col, crs, cols


def gpkg_shape(blob):
    from shapely import wkb
    if blob is None:
        return None
    b = bytes(blob)
    if len(b) < 8 or b[:2] != b'GP':
        raise ValueError('Invalid GeoPackage geometry header')
    envelope = (b[3] >> 1) & 7
    sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    if envelope not in sizes:
        raise ValueError(f'Unsupported GeoPackage envelope type: {envelope}')
    return wkb.loads(b[8 + sizes[envelope]:])


def resolve_step_json(explicit, diag_root, site, step):
    if step == 2:
        candidates = [diag_root / 'tdx_region_selection.json',
                      diag_root / 'step02' / f'{site}_tdx_region_selection.json']
    elif step == 3:
        candidates = [diag_root / 'tdx_outlet_selection.json',
                      diag_root / 'step03' / f'{site}_tdx_outlet_selection.json']
    else:
        raise ValueError('Unknown step number')
    if explicit is not None:
        candidates = [explicit]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'Step {step:02d} input JSON missing for {site}: {candidates}')


def db_signature(region, fields):
    """Bound to *actual* upstream source files (including replacements/repairs)."""
    stream = Path(region['streamnet_file']).resolve()
    basin = Path(region['basins_file']).resolve()
    result = {'version': INDEX_VERSION, 'tdx_id': str(region['tdx_id']),
              'stream_layer': region['streamnet_layer'], 'basin_layer': region['basins_layer'],
              'reach_field': fields['reach'], 'ds_field': fields['ds'],
              'us1_field': fields['us1'], 'us2_field': fields['us2'],
              'basin_reach_field': fields['basin']}
    for key, path in (('stream', stream), ('basin', basin)):
        stat = path.stat()
        result[key + '_path'] = str(path)
        result[key + '_size'] = stat.st_size
        result[key + '_mtime_ns'] = stat.st_mtime_ns
    return result


def check_index(path, expected):
    """Refuse to reuse stale/incomplete region indexes."""
    if not path.is_file():
        return False
    con = open_ro(path)
    try:
        found = con.execute('SELECT value FROM metadata WHERE key=?', ('signature',)).fetchone()
        if not found or json.loads(found[0]) != expected:
            raise ValueError(f'Stale topology index: {path}. Rebuild with --build-topology-index --index-only')
        for table in ('links', 'basin_fids'):
            if not con.execute('SELECT 1 FROM sqlite_master WHERE type="table" AND name=?', (table,)).fetchone():
                raise ValueError(f'Incomplete topology index: missing {table}')
        return True
    except sqlite3.Error as exc:
        raise ValueError(f'Unreadable topology index {path}: {exc}') from exc
    finally:
        con.close()


def stage_topology(source, layer, fields, dest):
    """Geometry-free stream indexing; requires input GeoPackage attributes."""
    dest.execute('CREATE TABLE links (rid TEXT PRIMARY KEY, fid INTEGER NOT NULL, ds TEXT, us1 TEXT, us2 TEXT)')
    query = (f'SELECT fid,{ident(fields["reach"])},{ident(fields["ds"])},'
             f'{ident(fields["us1"])},{ident(fields["us2"])} FROM {ident(layer)}')
    count, duplicates, batch = 0, 0, []
    for fid, rid, ds, us1, us2 in source.execute(query):
        rid = norm(rid)
        if rid is None:
            continue
        batch.append((rid, fid, norm(ds), norm(us1), norm(us2)))
        if len(batch) >= 10000:
            before = dest.total_changes
            dest.executemany('INSERT OR IGNORE INTO links VALUES (?,?,?,?,?)', batch)
            duplicates += len(batch) - (dest.total_changes - before)
            count += len(batch)
            batch.clear()
    if batch:
        before = dest.total_changes
        dest.executemany('INSERT OR IGNORE INTO links VALUES (?,?,?,?,?)', batch)
        duplicates += len(batch) - (dest.total_changes - before)
        count += len(batch)
    dest.execute('CREATE INDEX links_ds ON links(ds)')
    if duplicates:
        raise ValueError(f'{duplicates} duplicate TDX reach IDs in region: check source before indexing')
    return count


def stage_basin_fids(source, layer, basin_field, dest):
    """Index each native polygon's fid; multiple polygons per reach are valid."""
    dest.execute('CREATE TABLE basin_fids (rid TEXT NOT NULL, fid INTEGER NOT NULL, PRIMARY KEY (rid,fid))')
    query = f'SELECT fid,{ident(basin_field)} FROM {ident(layer)}'
    count, batch = 0, []
    for fid, rid in source.execute(query):
        rid = norm(rid)
        if rid is None:
            continue
        batch.append((rid, fid))
        if len(batch) >= 10000:
            dest.executemany('INSERT OR IGNORE INTO basin_fids VALUES (?,?)', batch)
            count += len(batch)
            batch.clear()
    if batch:
        dest.executemany('INSERT OR IGNORE INTO basin_fids VALUES (?,?)', batch)
        count += len(batch)
    return count


def build_index(path, signature, stream_con, basin_con, region, fields):
    """Explicit single-writer build; atomic publish for later read-only Slurm jobs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix('.lock')
    with lock_path.open('a+') as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        # Another builder may have completed while this process waited.
        try:
            is_valid = check_index(path, signature)
        except ValueError:
            is_valid = False  # Explicit --build request permits replacement of a stale index.
        if is_valid:
            return {'index': str(path), 'reused': True}
        fd, name = tempfile.mkstemp(prefix=path.stem + '.', suffix='.sqlite.tmp', dir=path.parent)
        os.close(fd)
        tmp = Path(name)
        try:
            db = sqlite3.connect(tmp)
            try:
                db.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)')
                print('Indexing regional TDX stream connectivity (attributes only)...', flush=True)
                stream_count = stage_topology(stream_con, region['streamnet_layer'], fields, db)
                print('Indexing native polygon FIDs (no geometry decoding)...', flush=True)
                basin_count = stage_basin_fids(basin_con, region['basins_layer'], fields['basin'], db)
                db.execute('INSERT INTO metadata VALUES (?,?)', ('signature', json.dumps(signature, sort_keys=True)))
                db.execute('INSERT INTO metadata VALUES (?,?)', ('stream_count', str(stream_count)))
                db.execute('INSERT INTO metadata VALUES (?,?)', ('basin_count', str(basin_count)))
                db.commit()
            finally:
                db.close()
            os.replace(tmp, path)
            return {'index': str(path), 'reused': False,
                    'stream_count': stream_count, 'basin_polygon_count': basin_count}
        finally:
            if tmp.exists():
                tmp.unlink()


def trace_upstream(db, outlet, max_reaches):
    if not db.execute('SELECT 1 FROM links WHERE rid=?', (outlet,)).fetchone():
        raise ValueError(f'Outlet LINKNO {outlet} not found in stream network')
    seen, queue, warnings = set(), deque([outlet]), []
    while queue:
        rid = queue.popleft()
        if rid in seen:
            continue
        seen.add(rid)
        if len(seen) > max_reaches:
            raise RuntimeError(f'More than {max_reaches} upstream reaches; check topology')
        row = db.execute('SELECT ds,us1,us2 FROM links WHERE rid=?', (rid,)).fetchone()
        if row is None:
            continue
        ds, us1, us2 = row
        if ds == rid:
            warnings.append(f'Self-referencing downstream link: {rid}')
        upstream = {r[0] for r in db.execute('SELECT rid FROM links WHERE ds=?', (rid,))}
        declared = {v for v in (us1, us2) if v not in (None, '0', '-1')}
        if declared and upstream != declared and len(warnings) < 25:
            warnings.append(f'Upstream pointers differ for {rid}: declared={sorted(declared)} actual={sorted(upstream)}')
        queue.extend(upstream - seen)
    return seen, warnings


def downstream_chain(db, outlet, max_reaches):
    """Follow DSLINKNO only; use to detect gauges below the basin outlet."""
    result, warnings, rid = set(), [], outlet
    for _ in range(max_reaches):
        row = db.execute('SELECT ds FROM links WHERE rid=?', (rid,)).fetchone()
        if row is None or norm(row[0]) in (None, '0', '-1'):
            break
        rid = norm(row[0])
        if rid in result or rid == outlet:
            warnings.append('Cycle detected in downstream TDX reach chain')
            break
        if not db.execute('SELECT 1 FROM links WHERE rid=?', (rid,)).fetchone():
            warnings.append(f'Downstream link {rid} outside current TDX region')
            break
        result.add(rid)
    else:
        warnings.append(f'Downstream chain exceeded {max_reaches} reaches')
    return result, warnings


def load_shared_gauges(path, region_id):
    """Only the region's catalogued snapped gauges; leave unresolved gauges visible."""
    if not path.is_file():
        raise FileNotFoundError(f'Shared gauge registry not found: {path}')
    gauges = []
    with path.open(newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        required = {'site_id', 'tdx_id', 'parent_tdx_reach_id',
                    'status', 'fraction_from_line_start', 'group_status'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f'Registry missing fields: {sorted(required - set(reader.fieldnames or []))}')
        seen = set()
        for record in reader:
            if norm(record.get('tdx_id')) != str(region_id):
                continue
            site = record.get('site_id', '').strip()
            if not site or site in seen:
                raise ValueError(f'Duplicate/empty site_id in registry for region {region_id}: {site}')
            seen.add(site)
            if norm(record.get('parent_tdx_reach_id')) in (None, '0'):
                continue
            if record.get('status', '').upper() == 'FAILED':
                continue
            gauges.append(record)
    return gauges


def classify_same_reach(orientation, outlet_fraction, other_fraction, eps=1e-8):
    """Flow-aware position check, using registry fractions on the *original* line."""
    if outlet_fraction is None or other_fraction is None:
        return 'UNRESOLVED_SAME_REACH_ORDER'
    if orientation not in ('downstream_to_upstream', 'upstream_to_downstream'):
        return 'UNRESOLVED_SAME_REACH_ORDER'
    delta = other_fraction - outlet_fraction
    if abs(delta) <= eps:
        return 'COINCIDENT_SAME_REACH_GAUGE'
    upstream = delta > 0 if orientation == 'downstream_to_upstream' else delta < 0
    return 'UPSTREAM_SAME_REACH' if upstream else 'DOWNSTREAM_SAME_REACH'


def fraction(value):
    try:
        f = float(value)
        return f if 0. <= f <= 1. else None
    except (TypeError, ValueError):
        return None


def analyze_shared_gauges(registry, site, tdx_id, outlet_reach, outlet, upstream, downstream):
    """Read-only relation classification, NOT the geometry-splitting operation."""
    gauges = load_shared_gauges(registry, tdx_id)
    own = next((g for g in gauges if g['site_id'] == site), None)
    orientation = outlet.get('line_orientation', '')
    own_fraction = fraction(own.get('fraction_from_line_start')) if own else None
    if own and norm(own['parent_tdx_reach_id']) != outlet_reach:
        raise ValueError(f'Shared registry/Step 03 mismatch: {site} parent {own["parent_tdx_reach_id"]} vs {outlet_reach}')
    relations = []
    for g in gauges:
        if g['site_id'] == site:
            continue
        parent = norm(g['parent_tdx_reach_id'])
        if parent == outlet_reach:
            relation = classify_same_reach(orientation, own_fraction, fraction(g.get('fraction_from_line_start')))
        elif parent in upstream:
            relation = 'UPSTREAM_OTHER_REACH'
        elif parent in downstream:
            relation = 'DOWNSTREAM_OTHER_REACH'
        else:
            continue  # Different tributary/unrelated network; NOT a nested gauge.
        relations.append({
            'site_id': g['site_id'], 'tdx_id': tdx_id, 'parent_tdx_reach_id': parent,
            'relation_to_outlet': relation, 'snap_status': g['status'],
            'group_status': g.get('group_status', ''), 'qa_flags': g.get('qa_flags', ''),
            'fraction_from_line_start': g.get('fraction_from_line_start', ''),
        })
    relations.sort(key=lambda row: (row['relation_to_outlet'], row['parent_tdx_reach_id'], row['site_id']))
    counts = dict(sorted(Counter(r['relation_to_outlet'] for r in relations).items()))
    flagged = [r for r in relations if (r['snap_status'].upper() != 'PROVISIONAL' or
               r['group_status'] != 'CANDIDATE_READY_FOR_TOPOLOGY' or
               r['relation_to_outlet'].startswith(('UNRESOLVED', 'COINCIDENT')))]
    return {
        'site_id': site, 'tdx_id': tdx_id,
        'outlet_parent_tdx_reach_id': outlet_reach,
        'outlet_orientation': orientation,
        'outlet_fraction_from_line_start': own_fraction,
        'outlet_registry_record_found': own is not None,
        'registry_path': str(registry), 'same_region_assigned_gauge_count': len(gauges),
        'relationship_counts': counts, 'related_gauge_count': len(relations),
        'related_gauges_needing_review': [r['site_id'] for r in flagged],
        'relationships': relations,
        'shared_relationships_analyzed': True,
        'split_reaches_generated': False,
        'fractional_ids_finalized': False,
        'final_export_authorized': False,
        'note': 'Upstream/downstream relationships are candidate metadata only. Geometry/HRU cuts are not performed in Step 04.'
    }


def write_relationship_csv(path, relationships):
    fields = ['site_id', 'tdx_id', 'parent_tdx_reach_id', 'relation_to_outlet',
              'snap_status', 'group_status', 'qa_flags', 'fraction_from_line_start']
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(relationships)


def native_polygons_from_index(basin_con, basin_layer, geom_col, basin_field, topology, upstream):
    result, matched = [], set()
    stmt = f'SELECT {ident(geom_col)} FROM {ident(basin_layer)} WHERE fid=?'
    for rid in sorted(upstream):
        for (fid,) in topology.execute('SELECT fid FROM basin_fids WHERE rid=?', (rid,)):
            geomrow = basin_con.execute(stmt, (fid,)).fetchone()
            if not geomrow or geomrow[0] is None:
                continue
            geom = gpkg_shape(geomrow[0])
            if geom is not None and not geom.is_empty:
                result.append({'reach_id': rid, 'geometry': geom})
                matched.add(rid)
    return result, matched


def native_polygons_full_scan(basin_con, basin_layer, geom_col, basin_field, upstream):
    result, matched = [], set()
    stmt = f'SELECT {ident(basin_field)},{ident(geom_col)} FROM {ident(basin_layer)}'
    for rid, blob in basin_con.execute(stmt):
        rid = norm(rid)
        if rid not in upstream or blob is None:
            continue
        geom = gpkg_shape(blob)
        if geom is not None and not geom.is_empty:
            result.append({'reach_id': rid, 'geometry': geom})
            matched.add(rid)
    return result, matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True, type=Path)
    parser.add_argument('--region-json', type=Path)
    parser.add_argument('--outlet-json', type=Path)
    parser.add_argument('--max-reaches', type=int, default=100000)
    parser.add_argument('--topology-index', type=Path, help='Override automatic shared SQLite path')
    parser.add_argument('--build-topology-index', action='store_true', help='Explicitly build/rebuild shared read-only source index')
    parser.add_argument('--index-only', action='store_true', help='Build/index-check only; no Step 03 or geometries needed')
    parser.add_argument('--require-topology-index', action='store_true', help='Disallow repeated per-basin full regional scans')
    parser.add_argument('--skip-shared-gauge-analysis', action='store_true')
    parser.add_argument('--require-shared-registry', action='store_true')
    args = parser.parse_args()
    if args.index_only and not args.build_topology_index:
        parser.error('--index-only requires --build-topology-index')
    with args.control.open('rb') as fh:
        cfg = tomllib.load(fh)
    dom, output, mapping = cfg['domain'], cfg['output'], cfg.get('tdx_fields', {})
    site = dom['site_id']
    root = Path(output['output_directory']) / site
    diag_root = root / output.get('diagnostics_subdirectory', 'diagnostics')
    diag = diag_root / 'step04'
    region_path = resolve_step_json(args.region_json, diag_root, site, 2)
    region = json.loads(region_path.read_text())
    if cfg.get('nested_gauges', {}).get('enabled', False):
        raise RuntimeError('nested_gauges.enabled=true requests physical splitting, not yet implemented in Steps 04-06. Keep false; relationships are analyzed automatically.')
    if cfg.get('headwater', {}).get('enabled', False):
        raise RuntimeError('headwater.enabled=true is not yet implemented. Keep false until Step 05 integrates it.')
    if str(region['tdx_id']) != str(cfg.get('input', {}).get('tdx_id', region['tdx_id'])):
        raise ValueError('Region JSON disagrees with control input.tdx_id')
    fields = {
        'reach': mapping.get('reach_id', 'LINKNO'),
        'ds': mapping.get('downstream_id', 'DSLINKNO'),
        'us1': mapping.get('upstream_id_1', 'USLINKNO1'),
        'us2': mapping.get('upstream_id_2', 'USLINKNO2'),
        'basin': mapping.get('basin_reach_id', 'streamID')}
    stream_con, basin_con = open_ro(region['streamnet_file']), open_ro(region['basins_file'])
    try:
        stream_layer, basin_layer = region['streamnet_layer'], region['basins_layer']
        sgeom, scrs, scols = layer_info(stream_con, stream_layer)
        bgeom, bcrs, bcols = layer_info(basin_con, basin_layer)
        for key in ('reach', 'ds', 'us1', 'us2'):
            if fields[key] not in scols:
                raise ValueError(f'Missing TDX stream field {fields[key]}')
        if fields['basin'] not in bcols:
            raise ValueError(f'Missing TDX basin field {fields["basin"]}')
        signature = db_signature(region, fields)
        index_path = args.topology_index or (PROJECT / 'indexes/tdx_regional_topology' / f'{region["tdx_id"]}_topology.sqlite')
        index_path = index_path.resolve()
        if args.build_topology_index:
            report = build_index(index_path, signature, stream_con, basin_con, region, fields)
            print('Regional topology index:', json.dumps(report, indent=2), flush=True)
        has_index = check_index(index_path, signature)
        if args.index_only:
            print(json.dumps({'tdx_id': region['tdx_id'], 'topology_index': str(index_path),
                              'index_valid': has_index, 'input_sources_unchanged': True}, indent=2))
            return
        if args.require_topology_index and not has_index:
            raise FileNotFoundError(f'Build shared regional topology first: {index_path}')
        outlet_path = resolve_step_json(args.outlet_json, diag_root, site, 3)
        outlet = json.loads(outlet_path.read_text())
        if str(region['tdx_id']) != str(outlet['tdx_id']):
            raise ValueError('Step 02 and Step 03 region IDs differ')
        if abs(float(dom['lat'])-float(outlet['gauge_lat']))>1e-9 or abs(float(dom['lon'])-float(outlet['gauge_lon']))>1e-9:
            raise ValueError('Gauge coordinates changed; rerun Step 03')
        rid = norm(outlet['reach_id'])
        matched_parent = norm(cfg.get('gauge_match', {}).get('parent_tdx_reach_id'))
        if matched_parent and matched_parent != rid:
            raise ValueError(f'TOML gauge_match parent {matched_parent} differs from Step 03 reach {rid}')
        metric = CRS.from_user_input(outlet['metric_crs'])
        slope_col = mapping.get('slope', 'Slope')
        has_slope = slope_col in scols
        diag.mkdir(parents=True, exist_ok=True)
        warnings = []
        # Read-only regional index, or old temporary scan for standalone legacy runs.
        with tempfile.TemporaryDirectory(prefix='tdx_step04_') as td:
            if has_index:
                topo = open_ro(index_path)
                indexed_polygons = True
                print(f'Reusing read-only regional topology index {index_path}', flush=True)
                n_scanned = int(topo.execute('SELECT value FROM metadata WHERE key="stream_count"').fetchone()[0])
            else:
                topo = sqlite3.connect(Path(td) / 'topology.sqlite')
                print('No shared index; scanning stream topology for this basin (legacy mode)...', flush=True)
                n_scanned = stage_topology(stream_con, stream_layer, fields, topo)
                topo.commit()
                indexed_polygons = False
            try:
                upstream, trace_warnings = trace_upstream(topo, rid, args.max_reaches)
                warnings.extend(trace_warnings)
                downstream, ds_warnings = downstream_chain(topo, rid, args.max_reaches)
                warnings.extend(ds_warnings)
                print(f'Traced {len(upstream)} upstream reaches from {n_scanned} regional records', flush=True)
                registry_opt = cfg.get('nested_gauges', {}).get('registry_file', '')
                registry = Path(registry_opt) if registry_opt else None
                shared = None
                if not args.skip_shared_gauge_analysis and registry and registry.is_file():
                    shared = analyze_shared_gauges(registry, site, str(region['tdx_id']), rid, outlet, upstream, downstream)
                elif args.require_shared_registry:
                    raise FileNotFoundError(f'Shared registry not available: {registry}')
                stream_rows = []
                geom_stmt = (f'SELECT {ident(sgeom)}, {ident(slope_col)} FROM {ident(stream_layer)} WHERE fid=?'
                             if has_slope else f'SELECT {ident(sgeom)} FROM {ident(stream_layer)} WHERE fid=?')
                for link in sorted(upstream):
                    fid_ds = topo.execute('SELECT fid,ds FROM links WHERE rid=?', (link,)).fetchone()
                    if fid_ds is None:
                        continue
                    fid, ds = fid_ds
                    geomrow = stream_con.execute(geom_stmt, (fid,)).fetchone()
                    geom = gpkg_shape(geomrow[0]) if geomrow and geomrow[0] is not None else None
                    if geom is not None and not geom.is_empty:
                        slope = geomrow[1] if has_slope else None
                        stream_rows.append({'reach_id': link, 'downstream_id': ds or '',
                                            'tdx_slope': float(slope) if slope is not None else None,
                                            'geometry': geom})
                if indexed_polygons:
                    print('Fetching indexed native polygons by FID...', flush=True)
                    basin_rows, matched = native_polygons_from_index(basin_con, basin_layer, bgeom, fields['basin'], topo, upstream)
                else:
                    print('Scanning native basin polygons (legacy mode)...', flush=True)
                    basin_rows, matched = native_polygons_full_scan(basin_con, basin_layer, bgeom, fields['basin'], upstream)
            finally:
                topo.close()
        missing = upstream - matched
        if missing:
            warnings.append(f'{len(missing)} reaches have no matching subbasin; examples: {sorted(missing)[:10]}')
        if not basin_rows:
            raise RuntimeError('No matching native subbasins found')
        if not stream_rows:
            raise RuntimeError('No upstream stream geometries found')
        bgdf = gpd.GeoDataFrame(basin_rows, geometry='geometry', crs=bcrs).to_crs('EPSG:4326')
        sgdf = gpd.GeoDataFrame(stream_rows, geometry='geometry', crs=scrs).to_crs('EPSG:4326')
        sgdf['reach_length_m'] = sgdf.to_crs(metric).geometry.length.to_numpy()
        # TDX stores some confluences as 0 m reaches with no subbasin polygon.
        # Drop them and link their inflows to the reach below, so every reach
        # keeps exactly one HRU; removed junctions are logged for traceability.
        rid_str = sgdf['reach_id'].astype(str)
        has_hru = {str(x) for x in matched}
        is_junction = (~rid_str.isin(has_hru)) & (sgdf['reach_length_m'] == 0)
        junction_down = dict(zip(rid_str[is_junction], sgdf.loc[is_junction, 'downstream_id'].astype(str)))
        def below_junction(ds):
            seen = set()
            while ds in junction_down and ds not in seen:
                seen.add(ds); ds = junction_down[ds]
            return ds
        removed_junctions = []
        for jid in junction_down:
            removed_junctions.append({'junction_id': jid, 'flows_into': below_junction(jid),
                                      'upstream_reaches': ';'.join(rid_str[sgdf['downstream_id'].astype(str) == jid])})
        if removed_junctions:
            ds_str = sgdf['downstream_id'].astype(str)
            relink = ds_str.isin(junction_down)
            sgdf.loc[relink, 'downstream_id'] = [type(v)(below_junction(str(v))) if v != '' else v
                                                 for v in sgdf.loc[relink, 'downstream_id']]
            sgdf = sgdf[~is_junction].reset_index(drop=True)
            warnings.append(f'Removed {len(removed_junctions)} zero-length TDX junction reaches without subbasins: '
                            f'{[j["junction_id"] for j in removed_junctions][:10]}')
        bp = bgdf.to_crs(metric)
        areas = bp.geometry.area / 1e6
        area_sum = float(areas.sum())
        outlet_area = float(areas[bgdf['reach_id'] == rid].sum())
        union_area = float(bp.geometry.union_all().area / 1e6)
        overlap = max(0., area_sum - union_area)
        if overlap > max(.01, .001 * area_sum):
            warnings.append(f'Overlapping polygon area approximately {overlap:.3f} km2')
        if outlet_area == 0:
            warnings.append('Outlet reach has no associated native polygon')
        observed = dom.get('observed_area_km2')
        observed = float(observed) if observed is not None else None
        difference = 100*(union_area-observed)/observed if observed and observed > 0 else None
        threshold = float(cfg.get('outlet', {}).get('area_difference_warning_pct', 20))
        if difference is not None and abs(difference) > threshold:
            warnings.append(f'Full native polygon area differs from observed by {difference:+.1f}%')
        if shared and not shared['outlet_registry_record_found']:
            warnings.append('Outlet site absent from regional shared registry')
        if shared and shared['related_gauges_needing_review']:
            warnings.append(f'{len(shared["related_gauges_needing_review"])} connected gauges require registry/snap review')
        root.mkdir(parents=True, exist_ok=True)
        gpkg = root / 'tdx_upstream_native.gpkg'
        if gpkg.exists():
            gpkg.unlink()  # generated per-site intermediate only, never source files
        bgdf.to_file(gpkg, layer='native_subbasins', driver='GPKG')
        sgdf.to_file(gpkg, layer='upstream_reaches', driver='GPKG', mode='a')
        points = gpd.GeoDataFrame([
            {'site_id': site, 'point_type': 'original', 'geometry': Point(float(dom['lon']), float(dom['lat']))},
            {'site_id': site, 'point_type': 'snapped', 'geometry': Point(float(outlet['snapped_lon']), float(outlet['snapped_lat']))}
        ], crs='EPSG:4326')
        points.to_file(gpkg, layer='outlet_points', driver='GPKG', mode='a')
        csv_path = diag / f'{site}_tdx_upstream_subbasins.csv'
        with csv_path.open('w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['reach_id', 'area_km2', 'outlet_subbasin'])
            for link, area in zip(bgdf['reach_id'], areas):
                writer.writerow([link, round(float(area), 8), link == rid])
        junction_csv = diag / f'{site}_tdx_removed_junctions.csv'
        with junction_csv.open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=['junction_id', 'flows_into', 'upstream_reaches'])
            writer.writeheader()
            writer.writerows(removed_junctions)
        fig, ax = plt.subplots(figsize=(11, 9))
        bp.plot(ax=ax, facecolor='#cfe4c6', edgecolor='#77936d', linewidth=.5, alpha=.75)
        sgdf.to_crs(metric).plot(ax=ax, color='#2775a5', linewidth=1.2)
        sgdf[sgdf['reach_id'] == rid].to_crs(metric).plot(ax=ax, color='#d84c3b', linewidth=2.6)
        gp = points.to_crs(metric)
        ax.scatter(gp.geometry.iloc[0].x, gp.geometry.iloc[0].y, c='black', marker='x', s=100, linewidth=2, zorder=8, label='Original gauge')
        ax.scatter(gp.geometry.iloc[1].x, gp.geometry.iloc[1].y, c='#ffbf45', edgecolor='black', s=90, zorder=9, label='Snapped outlet')
        ax.set_aspect('equal')
        ax.set_title(f'{site} | Native TDX network and subbasins\nFull outlet subbasin included; not clipped')
        ax.set_xlabel(f'Easting (m; {metric.to_string()})')
        ax.set_ylabel('Northing (m)')
        ax.legend(loc='best'); ax.grid(alpha=.15)
        fig.text(.5, .01, f'Native polygon union: {union_area:.2f} km2 | Observed: {observed if observed is not None else "NA"} km2', ha='center')
        jpg = diag / f'{site}_tdx_upstream_catchment_preview.jpg'
        fig.savefig(jpg, dpi=220, bbox_inches='tight')
        plt.close(fig)
        shared_path = None
        if shared is not None:
            shared_path = diag / f'{site}_tdx_shared_gauge_topology.json'
            shared_path.write_text(json.dumps(shared, indent=2)+'\n')
            write_relationship_csv(diag / f'{site}_tdx_related_gauges.csv', shared['relationships'])
        summary = {
            'site_id': site, 'tdx_id': region['tdx_id'], 'outlet_reach_id': rid,
            'upstream_reach_count': len(upstream), 'native_subbasin_polygon_count': len(bgdf),
            'unique_subbasin_reach_count': len(matched), 'total_region_reaches_scanned': n_scanned,
            'missing_subbasin_reach_count': len(missing),
            'removed_zero_length_junctions': len(removed_junctions),
            'removed_junctions_csv': str(junction_csv),
            'native_polygon_area_sum_km2': round(area_sum, 6),
            'native_polygon_union_area_km2': round(union_area, 6),
            'outlet_native_subbasin_area_km2': round(outlet_area, 6),
            'observed_area_km2': observed,
            'full_polygon_area_difference_pct': round(difference,3) if difference is not None else None,
            'gauge_interpolated_area_km2': outlet.get('tdx_area_estimate_km2'),
            'outlet_subbasin_treatment': 'full_native_polygon_not_split_at_gauge',
            'regional_topology_index_reused': has_index, 'regional_topology_index': str(index_path) if has_index else None,
            'shared_gauge_relationships_analyzed': shared is not None,
            'shared_related_gauge_count': shared['related_gauge_count'] if shared else None,
            'shared_gauge_relationship_counts': shared['relationship_counts'] if shared else None,
            'shared_gauge_topology_json': str(shared_path) if shared_path else None,
            'shared_topology_applied': False, 'fractional_ids_finalized': False,
            'headwater_extension_applied': False, 'final_export_authorized': False,
            'region_input_json': str(region_path), 'outlet_input_json': str(outlet_path),
            'intermediate_gpkg': str(gpkg), 'preview_jpg': str(jpg), 'warnings': warnings,
        }
        content = json.dumps(summary, indent=2)+'\n'
        (diag / f'{site}_tdx_upstream_summary.json').write_text(content)
        (diag_root / 'tdx_upstream_summary.json').write_text(content)
        print(json.dumps(summary, indent=2))
    finally:
        stream_con.close()
        basin_con.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
