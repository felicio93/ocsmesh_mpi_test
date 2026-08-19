#!/bin/bash
# =============================================================================
# Shared configuration for the FINAL OCSMesh MPI benchmark
# =============================================================================
# Sourced by every slurm_final_*.sh and slurm_profile_*.sh script so ALL
# modes run against the IDENTICAL workload (same manifest, recipe, hmin/hmax).
#
# Two benchmark profiles:
#
#   Profile A — _apply_features cost (the serial, un-MPI-parallelized stage)
#     Full recipe, 2-3 tiles, serial_mp only, windfall partition.
#     Goal: quantify ~3h/tile constraint cost; show it dominates and is the
#     next parallelization target.
#     Script: slurm_profile_a_serial.sh
#
#   Profile B — _calculate_and_write_hfun_to_disk speedup (what MPI accelerates)
#     LIGHT_FEATURES=1 + SKIP_CONSTRAINTS=1 (skip rank-0-only stages).
#     ~18 CUDEM tiles, all modes (serial_mp / parallel / mpi 1-node / multinode).
#     Goal: measure MPI speedup on the part it actually parallelizes (Gmsh).
#     Scripts: slurm_final_serial_mp.sh / parallel.sh / mpi_1node.sh / multinode.sh
#
# DO NOT hardcode manifest/recipe values in the individual job scripts.
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────
PROJ="/work2/noaa/nos-surge/felicioc/OCSMesh_MPI"
CONDA_BASE="/work2/noaa/nos-surge/felicioc/envs/miniconda3"
CONDA_ENV="ocsmesh_mpi_test"
SCRIPT_DIR="${PROJ}/ocsmesh_mpi_test"
STOFS_SHAPEFILE="${PROJ}/inputs/stofs3.shp"
DEM_OUT_DIR="${PROJ}/stofs_dems"

# ── Profile A manifest: 3 CUDEM tiles (full recipe, serial_mp only) ──────────
# ~3h/tile × 3 constraint tiles = ~9h; use windfall (24h) partition.
PROFILE_A_N_CUDEM="${PROFILE_A_N_CUDEM:-3}"
PROFILE_A_MANIFEST="${SCRIPT_DIR}/dem_manifest_profile_a.json"

# ── Profile B manifest: 18 CUDEM tiles (skip-constraints, all modes) ─────────
# At ~25 min/tile Gmsh, serial_mp takes ~7.5h (just fits 8h).
# parallel / mpi finish in minutes — strong speedup demonstration.
PROFILE_B_N_CUDEM="${PROFILE_B_N_CUDEM:-18}"
PROFILE_B_MANIFEST="${SCRIPT_DIR}/dem_manifest_profile_b.json"

# Default MANIFEST points at Profile B (used by slurm_final_*.sh scripts).
# Override in individual scripts as needed.
FULL_SMOKE_MANIFEST="${SCRIPT_DIR}/dem_manifest_smoke.json"
if [ -f "${PROFILE_B_MANIFEST}" ]; then
    MANIFEST="${PROFILE_B_MANIFEST}"
else
    MANIFEST="${SCRIPT_DIR}/dem_manifest_smoke7.json"   # smoke fallback
fi

# ── Recipe knobs ────────────────────────────────────────────────────────────
# Profile A: all flags OFF (full recipe).
# Profile B: LIGHT_FEATURES=1 + SKIP_CONSTRAINTS=1 (isolate meshdata stage).
# Individual scripts override these as needed.
LIGHT_FEATURES="${LIGHT_FEATURES:-0}"
LIGHT_FLAG=""
if [ "${LIGHT_FEATURES}" = "1" ]; then
    LIGHT_FLAG="--light-features"
fi

SKIP_TOPOFUNC="${SKIP_TOPOFUNC:-0}"
SKIP_TOPOFUNC_FLAG=""
if [ "${SKIP_TOPOFUNC}" = "1" ]; then
    SKIP_TOPOFUNC_FLAG="--skip-topofunc"
fi

SKIP_CONSTRAINTS="${SKIP_CONSTRAINTS:-0}"
SKIP_CONSTRAINTS_FLAG=""
if [ "${SKIP_CONSTRAINTS}" = "1" ]; then
    SKIP_CONSTRAINTS_FLAG="--skip-constraints"
fi

# FULL_PIPELINE=1 runs the complete end-to-end workflow (geom + hfun +
# MeshDriver final mesh) and records per-stage wall times. For the FINAL
# benchmark this should be 1 so we profile every stage and generate the
# actual mesh (mesh_<mode>.2dm) in addition to the hfun size field.
FULL_PIPELINE="${FULL_PIPELINE:-1}"
FULL_PIPELINE_FLAG=""
if [ "${FULL_PIPELINE}" = "1" ]; then
    FULL_PIPELINE_FLAG="--full-pipeline"
fi

ALL_FLAGS="${LIGHT_FLAG} ${SKIP_TOPOFUNC_FLAG} ${SKIP_CONSTRAINTS_FLAG} ${FULL_PIPELINE_FLAG}"

# Global mesh size bounds (metres). Identical across all modes.
HMIN="${HMIN:-1000.0}"
HMAX="${HMAX:-7000.0}"

# Worker count. 79 = 80 cores - 1 MPI manager rank.
NPROCS="${NPROCS:-79}"

# ── Environment loader ───────────────────────────────────────────────────────
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

print_final_config() {
    echo "================================================================="
    echo " OCSMesh benchmark config"
    echo "   Job ID          : ${SLURM_JOB_ID:-<none>}"
    echo "   Nodes           : ${SLURM_NODELIST:-<none>}"
    echo "   Manifest        : ${MANIFEST}"
    echo "   LIGHT_FEATURES  : ${LIGHT_FEATURES}"
    echo "   SKIP_TOPOFUNC   : ${SKIP_TOPOFUNC}"
    echo "   SKIP_CONSTRAINTS: ${SKIP_CONSTRAINTS}"
    echo "   FULL_PIPELINE   : ${FULL_PIPELINE}"
    echo "   hmin / hmax     : ${HMIN} / ${HMAX}"
    echo "   NPROCS          : ${NPROCS}"
    echo "================================================================="
}
