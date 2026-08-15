#!/bin/bash
# =============================================================================
# Shared configuration for the FINAL OCSMesh MPI benchmark
# =============================================================================
# Sourced by every slurm_final_*.sh script so that ALL modes
# (serial_mp / parallel / mpi single-node / mpi multi-node) run against the
# IDENTICAL workload: same DEM manifest, same shapefile, same refinement
# recipe, same hmin/hmax. This is what makes the cross-mode comparison and
# the numerical-equivalence check valid.
#
# DO NOT hardcode manifest/recipe values in the individual job scripts —
# change them HERE only.
#
# Usage inside a job script:
#     source "$(dirname "${BASH_SOURCE[0]}")/final_config.sh"
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────
PROJ="/work2/noaa/nos-surge/felicioc/OCSMesh_MPI"
CONDA_BASE="/work2/noaa/nos-surge/felicioc/envs/miniconda3"
CONDA_ENV="ocsmesh_mpi_test"
SCRIPT_DIR="${PROJ}/ocsmesh_mpi_test"
STOFS_SHAPEFILE="${PROJ}/inputs/stofs3.shp"
DEM_OUT_DIR="${PROJ}/stofs_dems"

# ── Workload manifest (SINGLE SOURCE OF TRUTH) ──────────────────────────────
# All final jobs use THIS manifest. Size it so serial_mp fits in ~7h.
#
# Sizing procedure (fill in after the smoke run measures serial_mp per-tile
# cost via the analyze_profile.py stage breakdown):
#   1. From the 15-tile smoke run, read serial_mp wall_time and divide by 15
#      to get seconds/tile (dominated by the topo_func constraint step).
#   2. Pick N_CUDEM so that  N_CUDEM * sec_per_tile  <=  ~6.5h (leave margin).
#   3. Regenerate FINAL_MANIFEST with that N_CUDEM (see trim_manifest.py).
#
# Default points at a to-be-generated 'final' manifest. Until it exists,
# jobs fall back to the 15-tile smoke manifest so they are runnable.
FINAL_MANIFEST="${SCRIPT_DIR}/dem_manifest_final.json"
FULL_SMOKE_MANIFEST="${SCRIPT_DIR}/dem_manifest_smoke.json"
FALLBACK_MANIFEST="${SCRIPT_DIR}/dem_manifest_smoke15.json"

if [ -f "${FINAL_MANIFEST}" ]; then
    MANIFEST="${FINAL_MANIFEST}"
else
    MANIFEST="${FALLBACK_MANIFEST}"
fi

# Number of CUDEM tiles to keep when generating dem_manifest_final.json.
# TODO: set this after measuring serial_mp per-tile cost on the smoke run.
FINAL_N_CUDEM="${FINAL_N_CUDEM:-14}"

# ── Recipe knobs ────────────────────────────────────────────────────────────
# LIGHT_FEATURES=1 skips global add_contour/add_channel (the O(tiles x segments)
# bottleneck). For the FINAL realistic benchmark this should be 0 (full recipe);
# set 1 only if serial_mp cannot otherwise fit in 8h at a useful tile count.
LIGHT_FEATURES="${LIGHT_FEATURES:-0}"
LIGHT_FLAG=""
if [ "${LIGHT_FEATURES}" = "1" ]; then
    LIGHT_FLAG="--light-features"
fi

# Global mesh size bounds (metres). Keep identical across all modes.
HMIN="${HMIN:-1000.0}"
HMAX="${HMAX:-7000.0}"

# Worker count for parallel and per-node MPI. 79 = 80 cores - 1 manager rank.
NPROCS="${NPROCS:-79}"

# ── Environment loader (call once at the top of each job) ───────────────────
load_ocsmesh_env() {
    module purge
    module load intel-oneapi-compilers/2022.2.1
    module load intel-oneapi-mpi/2021.7.1
    module load hdf5/1.12.2
    module load netcdf-c/4.9.0
    module load netcdf-fortran/4.6.0
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
}

# Print the resolved config so every job log records exactly what it ran.
print_final_config() {
    echo "================================================================="
    echo " OCSMesh FINAL benchmark config"
    echo "   Job ID        : ${SLURM_JOB_ID:-<none>}"
    echo "   Nodes         : ${SLURM_NODELIST:-<none>}"
    echo "   Manifest      : ${MANIFEST}"
    echo "   Shapefile     : ${STOFS_SHAPEFILE}"
    echo "   LIGHT_FEATURES: ${LIGHT_FEATURES}  (flag='${LIGHT_FLAG}')"
    echo "   hmin / hmax   : ${HMIN} / ${HMAX}"
    echo "   NPROCS        : ${NPROCS}"
    echo "================================================================="
}
