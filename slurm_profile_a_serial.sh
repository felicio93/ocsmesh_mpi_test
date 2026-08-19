#!/bin/bash
# =============================================================================
# Profile A — _apply_features cost (serial, rank-0-only stage)
# =============================================================================
# GOAL: Quantify the constraint application cost (~3h/tile) in serial_mp.
# This stage is NOT parallelized by MPI — it runs entirely on rank 0.
# Measuring it shows WHY serial_mp is slow and why _apply_features is the
# next major parallelization target (see TODO(mpi) in collector.py).
#
# Workload: 3 CUDEM tiles (full recipe — all constraints, contour, channel).
# Mode: serial_mp ONLY (parallel/mpi don't help this stage).
# Partition: windfall (24h) — 3 tiles × ~3h/tile constraint = ~9h minimum.
#
# Key outputs:
#   - Per-stage cProfile breakdown showing _apply_features dominates
#   - Serial wall-time per tile (topo_bound ~3h, courant ~3h, etc.)
#   - Baseline for Amdahl's-law ceiling on MPI speedup
#
# NOTE: do NOT set LIGHT_FEATURES or SKIP_CONSTRAINTS here.
# The full recipe is the point — we WANT the slow constraint stage to run.
#
# Submit: sbatch slurm_profile_a_serial.sh
# =============================================================================

#SBATCH --job-name=ocsmesh_profile_a
#SBATCH --account=nos-surge
#SBATCH --partition=hercules            # or windfall if job exceeds 8h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/profile_a_%j.out
#SBATCH --error=logs/profile_a_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

# Profile A: full recipe, 3 tiles, serial_mp only.
LIGHT_FEATURES=0
SKIP_TOPOFUNC=0
SKIP_CONSTRAINTS=0
ALL_FLAGS="--full-pipeline"   # full recipe + end-to-end (geom+hfun+MeshDriver)

RESULTS_DIR="${PROJ}/results/profile_a_serial_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}" logs

load_ocsmesh_env
print_final_config
echo "   Profile       : A — _apply_features cost (full recipe)"
echo "   Constraint tiles: 3 (topo_bound + topo_func + courant, ~3h each)"

# Build the Profile A manifest (3 CUDEM tiles) from the full smoke manifest.
if [ ! -f "${PROFILE_A_MANIFEST}" ]; then
    if [ ! -f "${FULL_SMOKE_MANIFEST}" ]; then
        echo "ERROR: ${FULL_SMOKE_MANIFEST} not found. Run download_dems.py first."
        exit 1
    fi
    echo "--- Building Profile A manifest (${PROFILE_A_N_CUDEM} CUDEM tiles) ---"
    srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/trim_manifest.py" \
        --in  "${FULL_SMOKE_MANIFEST}" \
        --out "${PROFILE_A_MANIFEST}" \
        --n-cudem "${PROFILE_A_N_CUDEM}"
fi
MANIFEST="${PROFILE_A_MANIFEST}"

export TMPDIR="${RESULTS_DIR}/tmp"
mkdir -p "${TMPDIR}"

echo ""
echo "--- Profile A: serial_mp, full recipe, ${PROFILE_A_N_CUDEM} CUDEM tiles ---"
srun --mpi=pmi2 -n 1 bash -c "\
    export TMPDIR='${RESULTS_DIR}/tmp'; \
    exec python '${SCRIPT_DIR}/run_benchmark.py' \
        --manifest '${MANIFEST}' --shapefile '${STOFS_SHAPEFILE}' \
        --out-dir '${RESULTS_DIR}' --nprocs '${NPROCS}' \
        --hmin '${HMIN}' --hmax '${HMAX}' \
        --modes serial_mp"

echo ""
echo "--- Generating Profile A report ---"
srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
    --results-dir "${RESULTS_DIR}" \
    --out         "${RESULTS_DIR}/report_profile_a.txt"

echo ""
echo "================================================================="
echo " Profile A complete. Results: ${RESULTS_DIR}"
echo " Key metric: serial_mp wall_time / ${PROFILE_A_N_CUDEM} tiles ="
echo "   per-tile constraint cost (check report_profile_a.txt)"
echo "================================================================="
