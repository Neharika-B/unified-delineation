#!/usr/bin/env python3
"""Step 05: provisional TDX watershed/HRU refinement using Copernicus D8.

Keeps the established Step 04 native GeoPackage and Step 05 output contract.
An independent DEM watershed is NEVER truncated to the native TDX union.
The DEM watershed is partitioned provisionally: upstream native HRUs intersected
with D8; leftover land belongs to the original outlet-parent reach. No final
nested fractional ID or upstream headwater river geometry is created here.
Source GeoPackages and original reach IDs remain unchanged.

Examples:
  python 1_delineation/tdx/05_refine_tdx_outlet.py --control 0_control_files/tdx_camels_spat/control_CAN_01AK006.toml --dem-min-accumulation-cells 100
  python 1_delineation/tdx/05_refine_tdx_outlet.py --control 0_control_files/control_USGS_10343500.toml --method full
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import tomllib
from contextlib import ExitStack
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import CRS, Geod, Transformer
from rasterio.features import shapes
from rasterio.merge import merge
from shapely.geometry import Point, shape
from shapely.ops import unary_union

D8_MAP = (64, 128, 1, 2, 4, 8, 16, 32)


def polygon(g):
    """Clean polygonal geometry; do not reinterpret nonpolygonal artifacts."""
    from shapely import make_valid
    if g is None or g.is_empty:
        return g
    if not g.is_valid:
        g = make_valid(g)
    if g.geom_type == 'GeometryCollection':
        g = unary_union([x for x in g.geoms if x.geom_type in ('Polygon', 'MultiPolygon')])
    if g.geom_type not in ('Polygon', 'MultiPolygon'):
        raise ValueError(f'Expected polygon geometry, found {g.geom_type}')
    return g


def full_path(value, project):
    if not value:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else project / p


def tile_paths(source, bounds4326):
    """Resolve Copernicus GLO-30 tiles by 1-degree tile name, no recursive scan."""
    p = Path(source)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(p)
    west, south, east, north = map(float, bounds4326)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError('DEM bounds outside ordinary longitude/latitude tiles')
    found, missing = [], []
    for lat in range(math.floor(south), math.ceil(north)):
        for lon in range(math.floor(west), math.ceil(east)):
            ns = f'N{lat:02d}' if lat >= 0 else f'S{-lat:02d}'
            ew = f'E{lon:03d}' if lon >= 0 else f'W{-lon:03d}'
            stem = f'Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM'
            f = p / stem / (stem + '.tif')
            (found if f.is_file() else missing).append(f)
    if missing:
        raise FileNotFoundError('Missing Copernicus tiles: ' + '; '.join(str(p) for p in missing[:8]))
    return found


def approximate_cell_area_km2(affine, crs, lon, lat):
    """Approximate representative DEM cell area near gauge for accumulation-based ranking."""
    crs = CRS.from_user_input(crs)
    if crs.is_geographic:
        # A geodesic 1-cell footprint at the gauge latitude.
        dx, dy = abs(float(affine.a)), abs(float(affine.e))
        geod = Geod(ellps='WGS84')
        x = [lon-dx/2, lon+dx/2, lon+dx/2, lon-dx/2]
        y = [lat-dy/2, lat-dy/2, lat+dy/2, lat+dy/2]
        signed_area, _ = geod.polygon_area_perimeter(x, y)
        return abs(signed_area) / 1e6
    if crs.axis_info and any((a.unit_name or '').lower() not in ('metre','meter','metres','meters') for a in crs.axis_info):
        raise ValueError('Projected DEM CRS must use metres for area estimate')
    return abs(float(affine.a) * float(affine.e) - float(affine.b) * float(affine.d)) / 1e6


def fill_small_voids(dem, invalid, max_void_fraction=0.02, max_connected_void_cells=1500):
    """Interpolate only small DEM voids; reject extensive gaps before hydrological conditioning."""
    from scipy.ndimage import distance_transform_edt, label
    n = int(invalid.sum())
    if n == 0:
        return dem, {'void_cells_filled':0, 'void_fraction':0., 'max_void_component_cells':0}
    fraction = n / invalid.size
    groups, n_groups = label(invalid)
    counts = np.bincount(groups[invalid].ravel())
    largest = int(counts.max()) if counts.size else 0
    if fraction > max_void_fraction or largest > max_connected_void_cells:
        raise RuntimeError(f'Unsafe Copernicus NoData: {n} cells ({fraction:.2%}), largest patch={largest}; extend/repair source mosaic')
    idx = distance_transform_edt(invalid, return_distances=False, return_indices=True)
    result = dem.copy()
    result[invalid] = result[tuple(idx[:, invalid])]
    if not np.isfinite(result).all():
        raise RuntimeError('DEM still contains nonfinite cells after small-void fill')
    return result, {'void_cells_filled':n, 'void_fraction':round(fraction,7), 'max_void_component_cells':largest}


def select_candidates(acc, valid, affine, crs, point, observed_km2,
                      radius_m, min_acc_requested, max_candidates=30):
    """Rank actual D8 cells near the gauge; area is a QA constraint, not a forced result.

    Smaller is better: abs(log(accumulation_area / observed_area)) and distance.
    A configured global min-accumulation floor is adaptively relaxed for tiny basins.
    """
    from rasterio.transform import rowcol, xy
    t = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    x, y = t.transform(point.x, point.y)
    rr, cc = rowcol(affine, x, y)
    if not (0 <= rr < acc.shape[0] and 0 <= cc < acc.shape[1]):
        raise RuntimeError('Gauge is outside Copernicus DEM window')
    ccrs = CRS.from_user_input(crs)
    if ccrs.is_geographic:
        sx, sy = 111320. * math.cos(math.radians(point.y)), 110574.
    else:
        sx, sy = 1., 1.
    dr = max(1, int(math.ceil(radius_m / (abs(affine.e) * sy))))
    dc = max(1, int(math.ceil(radius_m / (abs(affine.a) * sx))))
    cell_km2 = approximate_cell_area_km2(affine, crs, point.x, point.y)
    expected_cells = observed_km2 / cell_km2 if observed_km2 and observed_km2 > 0 else None
    if expected_cells:
        adaptive_floor = max(25, int(math.floor(0.1 * expected_cells)))
        minacc = min(int(min_acc_requested), adaptive_floor)
    else:
        minacc = int(min_acc_requested)
    candidates = []
    for r in range(max(0,rr-dr), min(acc.shape[0],rr+dr+1)):
        for c in range(max(0,cc-dc), min(acc.shape[1],cc+dc+1)):
            if not valid[r,c] or not math.isfinite(float(acc[r,c])):
                continue
            xx, yy = xy(affine,r,c)
            dist = math.hypot((xx-x)*sx, (yy-y)*sy)
            accumulation = float(acc[r,c])
            if dist > radius_m or accumulation < minacc:
                continue
            area_km2 = accumulation * cell_km2
            relative_error = 100*(area_km2/observed_km2 - 1) if observed_km2 and observed_km2 > 0 else None
            if expected_cells:
                area_score = abs(math.log(max(area_km2,1.e-12)/observed_km2))
                score = 2.0 * area_score + 0.5 * (dist/radius_m)
            else:
                score = -math.log(max(accumulation,1)) + 0.5*dist/radius_m
            candidates.append({'row':r,'col':c,'x':float(xx),'y':float(yy),
                'distance_m':float(dist),'accumulation_cells':accumulation,
                'estimated_area_km2':float(area_km2), 'estimated_area_difference_pct':relative_error,
                'score':float(score)})
    candidates.sort(key=lambda row:(row['score'],row['distance_m'],row['row'],row['col']))
    if not candidates:
        raise RuntimeError(f'No D8 cell meeting effective min accumulation {minacc} within {radius_m:g} m of gauge')
    return candidates[:max_candidates], {'dem_effective_min_accumulation_cells':minacc,
             'dem_configured_min_accumulation_cells':int(min_acc_requested),
             'dem_representative_cell_area_km2':cell_km2,
             'dem_expected_cells_from_observed_area':expected_cells,
             'dem_total_candidate_cells':len(candidates)}


def dem_boundary(directory, native, snapped, site, observed, settings, diag, options):
    """Produce independent D8 watershed without clipping to native TDX polygon."""
    try:
        from pysheds.grid import Grid
    except ImportError as exc:
        raise RuntimeError('Copernicus D8 requires pysheds in the unified_delineation environment') from exc
    # Older versions of pysheds use removed NumPy np.in1d.
    if not hasattr(np, 'in1d'):
        np.in1d = np.isin
    metric = native.estimate_utm_crs()
    if metric is None:
        raise RuntimeError('Cannot identify UTM zone for DEM window')
    buffer_m = options.dem_buffer_m if options.dem_buffer_m is not None else float(settings.get('dem_buffer_m',3000))
    snap_radius = options.dem_snap_radius_m if options.dem_snap_radius_m is not None else float(settings.get('dem_snap_radius_m',300))
    minacc = options.dem_min_accumulation_cells if options.dem_min_accumulation_cells is not None else int(settings.get('dem_min_accumulation_cells',10000))
    max_cells = int(settings.get('dem_max_cells',30000000))
    max_void_patch = (options.max_connected_void_cells if options.max_connected_void_cells is not None else 1500)
    if max_void_patch < 1:
        raise ValueError('max-connected-void-cells must be positive')
    if buffer_m <= 0 or snap_radius <= 0 or minacc < 1:
        raise ValueError('Invalid DEM buffer / snap radius / min accumulation settings')
    from rasterio.warp import transform_bounds
    win = native.to_crs(metric).geometry.union_all().buffer(buffer_m)
    bounds4326 = gpd.GeoSeries([win], crs=metric).to_crs('EPSG:4326').total_bounds
    tiles = tile_paths(directory, bounds4326)
    with ExitStack() as stack:
        ds = [stack.enter_context(rasterio.open(f)) for f in tiles]
        crs = ds[0].crs
        if crs is None or any(d.crs != crs for d in ds):
            raise ValueError('DEM tiles have missing or incompatible CRS')
        bb = transform_bounds('EPSG:4326', crs, *bounds4326, densify_pts=21)
        # Snap the window outward to the Copernicus pixel grid so merge()
        # does not leave an unfilled last column/row at tile borders.
        rx, ry = ds[0].res
        x0, y0 = ds[0].bounds.left, ds[0].bounds.top
        bb = (x0 + math.floor((bb[0] - x0) / rx) * rx,
              y0 + math.floor((bb[1] - y0) / ry) * ry,
              x0 + math.ceil((bb[2] - x0) / rx) * rx,
              y0 + math.ceil((bb[3] - y0) / ry) * ry)
        # Copernicus tiles carry no nodata value, so merge() would otherwise
        # treat 0 m (sea level) as nodata. NaN marks only cells never filled.
        mosaic, affine = merge(ds, bounds=bb, indexes=1, masked=True, nodata=np.nan)
        raw = np.ma.asarray(mosaic[0])
        if raw.size > max_cells:
            raise RuntimeError(f'DEM window {raw.size:,} cells > dem_max_cells={max_cells:,}; use a smaller local domain / different tiling strategy')
        invalid = np.ma.getmaskarray(raw) | ~np.isfinite(np.asarray(raw.data))
        filled = np.asarray(raw.data, dtype=np.float32)
        filled, void_qc = fill_small_voids(filled, invalid, max_connected_void_cells=max_void_patch)
        void_qc['dem_max_connected_void_cells_allowed'] = max_void_patch
        void_qc['dem_large_void_preview_override_used'] = bool(void_qc['max_void_component_cells'] > 1500)
        print(f'Loaded Copernicus DEM window {filled.shape}, interpolated {void_qc["void_cells_filled"]} small voids',flush=True)
        with tempfile.TemporaryDirectory(prefix='tdx_step05_') as tmp:
            tif = Path(tmp)/'local_dem.tif'
            with rasterio.open(tif,'w',driver='GTiff',height=filled.shape[0],width=filled.shape[1],
                 count=1,crs=crs,transform=affine,dtype='float32',compress='deflate') as target:
                target.write(filled,1)
            grid = Grid.from_raster(str(tif))
            surface = grid.read_raster(str(tif))
            print('Conditioning DEM and deriving D8 network...',flush=True)
            conditioned = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(surface)))
            fdir = grid.flowdir(conditioned,dirmap=D8_MAP)
            acc = grid.accumulation(fdir,dirmap=D8_MAP)
            candidates, cand_qc = select_candidates(acc,np.isfinite(filled),affine,crs,snapped,observed,snap_radius,minacc)
            selected = candidates[0]
            # Explicit xytype='index' matches row/column convention in pysheds Grid.catchment.
            catch = grid.catchment(x=selected['col'],y=selected['row'],fdir=fdir,dirmap=D8_MAP,xytype='index')
            mask = np.asarray(catch,dtype=bool)
            mask[selected['row'],selected['col']] = True
            void_qc['dem_filled_void_cells_in_watershed'] = int((mask & invalid).sum())
            if not mask.any():
                raise RuntimeError('DEM watershed is empty')
            edge = bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
            if edge:
                raise RuntimeError('D8 watershed touches the DEM window edge; increase dem_buffer_m (no truncated basin exported)')
            geom = polygon(unary_union([shape(g) for g,v in shapes(mask.astype('uint8'),mask=mask,transform=affine) if v == 1]))
            boundary = gpd.GeoDataFrame({'source':['Copernicus GLO-30 D8']},geometry=[geom],crs=crs).to_crs('EPSG:4326')
            diag5 = diag/'step05'
            diag5.mkdir(exist_ok=True,parents=True)
            csv_file = diag5/f'{site}_dem_candidate_cells.csv'
            with csv_file.open('w',newline='') as f:
                writer = csv.DictWriter(f,fieldnames=list(selected))
                writer.writeheader(); writer.writerows(candidates)
            rasters = {}
            if options.save_d8_rasters:
                for name,array,dtype in [('conditioned_dem',conditioned,'float32'),('flow_direction_D8',fdir,'uint8'),
                                          ('flow_accumulation',acc,'float64'),('watershed_mask',mask,'uint8')]:
                    path = diag5/f'{site}_{name}.tif'
                    with rasterio.open(path,'w',driver='GTiff',height=filled.shape[0],width=filled.shape[1],count=1,
                         dtype=dtype,crs=crs,transform=affine,compress='deflate') as output:
                        output.write(np.asarray(array,dtype=dtype),1)
                    rasters[name] = str(path)
            details = {'dem_tiles':[str(f) for f in tiles], 'dem_window_cells':int(filled.size),
                'dem_candidate_cells_csv':str(csv_file), 'dem_candidate_selection':'area_distance_ranked_not_forced',
                'dem_selected_cell_row':selected['row'],'dem_selected_cell_col':selected['col'],
                'dem_snap_distance_m':round(selected['distance_m'],3),
                'dem_accumulation_cells':round(selected['accumulation_cells'],2),
                'dem_accumulation_area_estimate_km2':round(selected['estimated_area_km2'],5),
                'dem_accumulation_area_difference_pct':(round(selected['estimated_area_difference_pct'],2)
                    if selected['estimated_area_difference_pct'] is not None else None),
                'dem_watershed_cell_count':int(mask.sum()), 'dem_watershed_touches_window_edge':edge,
                'dem_method':'pysheds conditioned Copernicus D8',
                'dem_raster_files':rasters, **void_qc, **cand_qc}
            return boundary,details


def provisional_partition(native, watershed, outlet_id):
    """Preserve every native upstream TDX HRU; cut ONLY the outlet HRU.

    The independent DEM watershed is used only as a spatial cutter for the
    original outlet polygon. It does not redefine upstream native HRUs, and
    no upstream reach/HRU ID is removed. Any TDX area outside the native
    outlet polygon is NOT added or assigned to a different HRU.
    """
    selected = native.copy()
    selected['reach_id'] = selected['reach_id'].astype(str)
    if selected.reach_id.duplicated().any():
        raise ValueError('Duplicate native reach/HRU ID in Step 04 subbasins')
    original = polygon(selected.geometry.union_all())
    upstream = selected[selected.reach_id != outlet_id].copy()
    outlet = selected[selected.reach_id == outlet_id]
    if len(outlet) != 1:
        raise ValueError(f'Expected exactly one native outlet HRU for {outlet_id}')
    upstream_union = polygon(upstream.geometry.union_all()) if len(upstream) else None
    outlet_native = polygon(outlet.geometry.iloc[0])
    outlet_cut = polygon(outlet_native.intersection(watershed))
    # Avoid introducing overlaps if source TDX polygons have minor edge errors.
    if upstream_union is not None:
        outlet_cut = polygon(outlet_cut.difference(upstream_union))
    if outlet_cut is None or outlet_cut.is_empty or outlet_cut.area <= 0:
        raise RuntimeError('DEM watershed does not intersect the native outlet HRU; check pour point')
    records = [
        {'reach_id': row.reach_id, 'partition_source': 'native_TDX_unchanged',
         'geometry': row.geometry}
        for row in upstream.itertuples()
    ]
    records.append({'reach_id': outlet_id,
                    'partition_source': 'native_TDX_outlet_cut_by_D8',
                    'geometry': outlet_cut})
    out = gpd.GeoDataFrame(records, geometry='geometry', crs=native.crs)
    if len(out) != len(selected) or set(out.reach_id) != set(selected.reach_id):
        raise RuntimeError('Step 05 must retain exactly the original TDX HRU IDs')
    return out, original, upstream_union


def km2(geometry, crs, metric):
    return float(gpd.GeoSeries([geometry],crs=crs).to_crs(metric).area.iloc[0]/1e6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--method',choices=['full','dem_refined','reference_clip','hydrosheds_v2'])
    parser.add_argument('--dem-min-accumulation-cells',type=int,
                        help='Minimum D8 accumulation to evaluate; actual effective threshold is lowered for small observed basins')
    parser.add_argument('--dem-buffer-m',type=float)
    parser.add_argument('--dem-snap-radius-m',type=float)
    parser.add_argument('--save-d8-rasters',action='store_true')
    parser.add_argument('--max-connected-void-cells',type=int,
                        help='Explicit exploratory override of default 1500-cell contiguous NoData guardrail. Any override used is flagged REVIEW.')
    args = parser.parse_args()
    control = args.control.resolve()
    # Control can be inside 0_control_files/tdx_camels_spat or top 0_control_files.
    project = Path(__file__).resolve().parents[2]
    with control.open('rb') as handle:
        cfg = tomllib.load(handle)
    dom, output, settings = cfg['domain'],cfg['output'],cfg.get('delineation',{})
    site = dom['site_id']; method = args.method or settings.get('outlet_subbasin_method','full')
    root = Path(output['output_directory'])/site
    diag = root/output.get('diagnostics_subdirectory','diagnostics')
    diag.mkdir(parents=True,exist_ok=True)
    source = root/'tdx_upstream_native.gpkg'
    if not source.is_file():
        raise FileNotFoundError(f'Run Step 04 first: {source}')
    native = gpd.read_file(source,layer='native_subbasins').to_crs('EPSG:4326')
    streams = gpd.read_file(source,layer='upstream_reaches').to_crs('EPSG:4326')
    points = gpd.read_file(source,layer='outlet_points').to_crs('EPSG:4326')
    outlet_path = diag/'tdx_outlet_selection.json'
    if not outlet_path.is_file():
        outlet_path = diag/'step03'/f'{site}_tdx_outlet_selection.json'
    outlet = json.loads(outlet_path.read_text())
    outlet_id = str(outlet['reach_id'])
    native['reach_id'] = native['reach_id'].astype(str)
    if native.empty or outlet_id not in set(native.reach_id):
        raise ValueError('No native polygon for selected outlet reach')
    if native.crs is None or streams.crs is None:
        raise ValueError('Native GPKG must contain usable CRS metadata')
    native['geometry'] = native.geometry.apply(polygon)
    if native.geometry.is_empty.any():
        raise ValueError('Empty native polygon geometry')
    snapped = Point(float(outlet['snapped_lon']),float(outlet['snapped_lat']))
    metric = settings.get('projected_crs') or str(native.estimate_utm_crs())
    if not metric:
        raise RuntimeError('Cannot identify metric CRS')
    from pyproj import CRS
    if CRS.from_user_input(metric).is_geographic:
        raise ValueError('delineation.projected_crs must be projected')
    observed = float(dom['observed_area_km2']) if dom.get('observed_area_km2') else None
    native_union = polygon(native.geometry.union_all())
    extra = {}
    if method == 'full':
        boundary = gpd.GeoDataFrame({'source':['full native subbasins']},geometry=[native_union],crs=native.crs)
        refined = native.copy()
        refined['partition_source'] = 'native_original'
        upstream_native = native[native.reach_id != outlet_id]
        upstream_geom = upstream_native.geometry.union_all() if len(upstream_native) else None
    elif method == 'reference_clip':
        reference = full_path(settings.get('reference_catchment_file'),project)
        if reference is None or not reference.is_file():
            raise FileNotFoundError('reference_clip requires an existing delineation.reference_catchment_file')
        gg = gpd.read_file(reference)
        if gg.empty or gg.crs is None:
            raise ValueError('Reference boundary is empty or lacks CRS')
        geometry = polygon(gg.to_crs('EPSG:4326').geometry.union_all())
        boundary = gpd.GeoDataFrame({'source':['reference_clip']},geometry=[geometry],crs='EPSG:4326')
        refined,native_union,upstream_geom = provisional_partition(native,geometry,outlet_id)
        extra['reference_catchment_file'] = str(reference)
        extra['independent_reference_boundary_area_km2'] = km2(geometry, 'EPSG:4326', metric)
        boundary = gpd.GeoDataFrame({'source':['TDX upstream + clipped outlet HRU']},
                                   geometry=[polygon(refined.geometry.union_all())], crs=native.crs)
    elif method == 'dem_refined':
        dem = full_path(settings.get('dem_directory') or cfg.get('input',{}).get('dem_directory'),project)
        if dem is None:
            raise ValueError('Missing delineation.dem_directory for dem_refined')
        dem_watershed,extra = dem_boundary(dem,native,snapped,site,observed,settings,diag,args)
        dem_shape = polygon(dem_watershed.geometry.union_all())
        extra['independent_dem_watershed_area_km2'] = km2(dem_shape, dem_watershed.crs, metric)
        refined,native_union,upstream_geom = provisional_partition(native,dem_shape,outlet_id)
        boundary = gpd.GeoDataFrame({'source':['TDX upstream + D8-cut outlet HRU']},
                                   geometry=[polygon(refined.geometry.union_all())], crs=native.crs)
    elif method == 'hydrosheds_v2':
        raise RuntimeError('HydroSHEDS ACC/RIV/BAS alone cannot reconstruct a unique D8 watershed; use dem_refined or reference_clip')
    else:
        raise ValueError(method)
    boundary_geom = polygon(boundary.geometry.union_all())
    union_refined = polygon(refined.geometry.union_all())
    boundary_area = km2(boundary_geom,boundary.crs,metric)
    refined_area = km2(union_refined,refined.crs,metric)
    native_area = km2(native_union,native.crs,metric)
    outside_area = km2(polygon(boundary_geom.difference(native_union)),native.crs,metric)
    native_outside = km2(polygon(native_union.difference(boundary_geom)),native.crs,metric)
    coverage_error = km2(polygon(boundary_geom.symmetric_difference(union_refined)),native.crs,metric)
    if coverage_error > max(0.001, boundary_area*1.e-5):
        raise RuntimeError(f'Refined polygon coverage differs from D8 watershed by {coverage_error:.5f} km2')
    discrepancy = 100 * (refined_area/observed-1.) if observed and observed>0 else None
    refined = refined.copy()
    refined['area_km2'] = refined.to_crs(metric).geometry.area/1e6
    refined['outlet_unit'] = refined.reach_id == outlet_id
    warnings = []
    qa = []
    if method == 'dem_refined' and extra.get('dem_large_void_preview_override_used'):
        qa.append('DEM_LARGE_VOID_INTERPOLATED_FOR_PREVIEW')
        warnings.append('DEM NoData patch exceeds default 1500 cells; exploratory interpolation requires manual review')
    if method == 'dem_refined' and extra.get('dem_filled_void_cells_in_watershed', 0):
        qa.append('DEM_FILLED_VOID_WITHIN_DELINEATED_WATERSHED')
        warnings.append(f"{extra['dem_filled_void_cells_in_watershed']} interpolated DEM void cells lie inside delineated watershed")
    threshold = float(cfg.get('headwater',{}).get('max_area_difference_pct',
                    cfg.get('outlet',{}).get('area_difference_warning_pct',20)))
    if discrepancy is not None and abs(discrepancy)>threshold:
        qa.append('DEM_AREA_OUTSIDE_TOLERANCE')
        warnings.append(f'Refined drainage area differs from observed by {discrepancy:+.1f}% (> {threshold:g}%)')
    if method == 'dem_refined' and extra.get('dem_snap_distance_m',0)>150:
        qa.append('DEM_POUR_POINT_OVER_150M_FROM_TDX_SNAP')
    if method == 'dem_refined' and extra.get('dem_effective_min_accumulation_cells') < extra.get('dem_configured_min_accumulation_cells'):
        warnings.append('Accumulation threshold was adaptively reduced for small observed basin; inspect candidates')
    if method == 'dem_refined' and outside_area > max(.01, boundary_area*.02):
        qa.append('DEM_CONTRIBUTING_AREA_OUTSIDE_NATIVE_TDX')
        warnings.append('Some contributing area lies outside the native TDX boundary; no additional land is assigned by Step 05')
    if method == 'dem_refined' and native_outside > max(.01, native_area*.02):
        warnings.append('Part of original native TDX catchment lies outside independent D8 watershed')
    topology_path = diag/'step04'/f'{site}_tdx_shared_gauge_topology.json'
    related_gauges = None
    if topology_path.exists():
        topology = json.loads(topology_path.read_text())
        related_gauges = topology.get('related_gauge_count')
        if related_gauges:
            qa.append('NESTED_GAUGES_REQUIRE_SHARED_HRU_PARTITION')
    if method != 'full':
        warnings.append('Upstream TDX HRUs preserved; only the outlet HRU is cut; shared nested splits are handled in Step 05b')
    target = root/'tdx_upstream_refined.gpkg'
    if target.exists():
        target.unlink()
    refined.to_file(target,layer='refined_subbasins',driver='GPKG')
    streams.to_file(target,layer='upstream_reaches',driver='GPKG',mode='a')
    points.to_file(target,layer='outlet_points',driver='GPKG',mode='a')
    boundary.to_file(target,layer='refinement_boundary',driver='GPKG',mode='a')
    csv_path = diag/'tdx_refined_subbasins.csv'
    with csv_path.open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['reach_id','area_km2','outlet_unit','partition_source'])
        for r in refined.itertuples():
            writer.writerow([r.reach_id,round(float(r.area_km2),8),r.outlet_unit,r.partition_source])
    # Save independent intermediate for human review; no change to source topology.
    fig,ax=plt.subplots(figsize=(10,8))
    native.to_crs(metric).plot(ax=ax,edgecolor='#666666',facecolor='none',linestyle='--',linewidth=1)
    refined.to_crs(metric).plot(ax=ax,facecolor='#cfe4c6',edgecolor='#537d54',alpha=.65,linewidth=.6)
    boundary.to_crs(metric).boundary.plot(ax=ax,color='#2d7650',linewidth=1.7)
    streams.to_crs(metric).plot(ax=ax,color='#2775a5',linewidth=1.)
    streams[streams.reach_id.astype(str)==outlet_id].to_crs(metric).plot(ax=ax,color='#d84c3b',linewidth=2)
    sp=gpd.GeoSeries([snapped],crs='EPSG:4326').to_crs(metric).iloc[0]
    ax.scatter(sp.x,sp.y,color='black',marker='x',s=95,zorder=9,label='TDX snapped gauge')
    ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(loc='best')
    ax.set_title(f'{site}: {method} candidate | native dashed, independent boundary green')
    fig.text(.5,.01,f'Native {native_area:.2f} | D8/refined {refined_area:.2f} | Observed {observed if observed else "NA"} km2',ha='center')
    preview=diag/'tdx_refined_catchment_preview.jpg'
    fig.savefig(preview,dpi=220,bbox_inches='tight');plt.close(fig)
    summary = {'site_id':site,'method':method,'outlet_parent_tdx_reach_id':outlet_id,
        'native_gpkg':str(source),'refined_gpkg':str(target),
        'native_area_km2':round(native_area,6),
        'independent_watershed_area_km2':round(boundary_area,6),
        'refined_area_km2':round(refined_area,6),
        'observed_area_km2':observed,
        'refined_area_difference_pct':round(discrepancy,3) if discrepancy is not None else None,
        'dem_watershed_outside_native_tdx_km2':round(outside_area,6),
        'native_tdx_outside_dem_watershed_km2':round(native_outside,6),
        'partition_coverage_error_km2':round(coverage_error,8),
        'native_subbasin_count':len(native),'refined_subbasin_count':len(refined),
        'shared_related_gauge_count':related_gauges,'qa_flags':sorted(set(qa)),
        'qa_status':'REVIEW' if qa else 'PROVISIONAL',
        'headwater_extension_geometry_generated':False,
        'tdx_original_reach_ids_preserved':True,
        'upstream_tdx_hrus_preserved_unchanged': True,
        'dem_applied_to_outlet_hru_only':method == 'dem_refined',
        'nested_fractional_reach_ids_applied':False,
        'shared_nested_hru_partition_performed':False,
        'final_export_authorized':False,
        'preview_jpg':str(preview), 'warnings':warnings, **extra}
    (diag/'tdx_refinement_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    diag5=diag/'step05';diag5.mkdir(exist_ok=True,parents=True)
    (diag5/f'{site}_tdx_refinement_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
