#!/usr/bin/env bash
#SBATCH --job-name=tdx_snap02b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-71%8
#SBATCH --output=slurm_logs/tdx_snapping/tdx_snap_%A_%a.out
#SBATCH --error=slurm_logs/tdx_snapping/tdx_snap_%A_%a.err

# 02b: TDX reach snapping for the validated CAMELS-SPAT catalogue.
# One Slurm array task processes 20 distinct gauges.
# For 1,426 gauges, array 0-71 covers every gauge exactly once.
# Submit from the project root after creating slurm_logs/tdx_snapping.

set -euo pipefail
umask 002

PROJECT="/work/comphyd_lab/users/arman.haddadchi/unified_delineation_code"
PYTHON="${HOME}/.conda/envs/unified_delineation/bin/python"

CATALOG="${PROJECT}/indexes/gauge_catalogs/camels-spat_gauge_catalog_validated.csv"
REGION_INDEX="${PROJECT}/indexes/tdx_regions.csv"
WORKER="${PROJECT}/1_delineation/tdx/02b_snap_gauge_catalog.py"
OUTPUT_DIR="${PROJECT}/indexes/gauge_catalogs/tdx_snapping/batch_500m"

BATCH_SIZE=20
SEARCH_RADIUS_M=500

for path in "$CATALOG" "$REGION_INDEX" "$WORKER"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: Missing required file: $path" >&2
        exit 1
    fi
done
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found/executable at $PYTHON" >&2
    exit 1
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: Run this worker using sbatch --array, not directly." >&2
    exit 1
fi

# Count records rather than assuming the CSV will always contain 1,426 gauges.
N_GAUGES="$($PYTHON -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8-sig"))))' "$CATALOG")"
EXPECTED_TASKS=$(( (N_GAUGES + BATCH_SIZE - 1) / BATCH_SIZE ))
if (( EXPECTED_TASKS != 72 )); then
    echo "ERROR: Catalogue needs ${EXPECTED_TASKS} tasks, but #SBATCH --array is 0-71 (72 tasks). Update array range before submitting." >&2
    exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= EXPECTED_TASKS )); then
    echo "ERROR: Array index ${SLURM_ARRAY_TASK_ID} is outside the ${EXPECTED_TASKS} required tasks." >&2
    exit 1
fi

START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
STOP=$(( START + BATCH_SIZE ))
if (( STOP > N_GAUGES )); then
    STOP="$N_GAUGES"
fi

# One core per task; many tasks can run concurrently across ARC nodes.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT"

echo "Job: ${SLURM_JOB_ID:-unknown}, array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Catalogue gauge count: ${N_GAUGES}"
echo "Gauge indices: [${START}, ${STOP})"
echo "Area units: km2"
echo "Output directory: ${OUTPUT_DIR}"

"$PYTHON" -u "$WORKER" \
    --catalog "$CATALOG" \
    --region-index "$REGION_INDEX" \
    --output-dir "$OUTPUT_DIR" \
    --start "$START" \
    --stop "$STOP" \
    --radius-m "$SEARCH_RADIUS_M" \
    --area-units m2

echo "DONE: task ${SLURM_ARRAY_TASK_ID}; gauge indices [${START}, ${STOP})"
