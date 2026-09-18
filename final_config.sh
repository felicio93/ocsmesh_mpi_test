#!/bin/bash
# =============================================================================
# Shared configuration for the FINAL OCSMesh MPI benchmark
# =============================================================================
# Sourced by every slurm_final_*.sh, slurm_smoke_*.sh, and slurm_anas_*.sh
# script so ALL modes run against the IDENTICAL workload (same manifest,
# recipe, hmin/hmax).
#
# Execution modes supported:
#   serial_true   — true single-core baseline
#   serial_mp     — serial mode, Pool steps use NPROCS workers
#   parallel      — full multiprocessing Pool
#   mpi           — MPI via MPIExecutor, NPROCS workers per rank
#   mpi_no_pool   — MPI, 1 core per rank (pure MPI, no internal Pool)
#   mpi_hybrid    — MPI, auto cores per rank (MPI + internal Pool)
#
# Two benchmark profiles:
#
#   Profile A — _apply_features cost (serial, rank-0-only stages)
#     Full recipe, 3 tiles, serial_mp only, windfall partition.
#     Goal: quantify constraint cost; show it dominates and is next target.
#     Script: slurm_profile_a_serial.sh
#
#   Profile B — MPI speedup on meshdata dispatch
#     LIGHT_FEATURES=1 + SKIP_CONSTRAINTS=1 (skip rank-0-only stages).
#     ~18 CUDEM tiles, serial_mp / parallel / mpi / mpi_no_pool / mpi_hybrid.
#     Goal: measure MPI speedup on the parallelized stage (Gmsh meshdata).
#     Scripts: slurm_final_serial_mp.sh / parallel.sh / mpi_1node.sh /
#              mpi_multinode.sh
#
#   Profile C — Anas's new operations (all MPI-dispatched)
#     Config F (no boxes) and Config G (full pipeline with boxes).
#     Tests serial_mp / mpi_no_pool / mpi_hybrid on real STOFS DEMs.
#     Thin-rank variant (many 1-core ranks) and fat-rank variant
#     (few multi-core ranks to show mpi_hybrid advantage).
#     Scripts: slurm_anas_matrix.sh / slurm_smoke_config_F.sh /
#              slurm_smoke_config_G.sh
#
# DO NOT hardcode manifest/recipe values in the individual job scripts.
# =============================================================================

# ── Paths ────────────────────────────────────────────────────────────────────
PROJ="/work2/noaa/nos-surge/felicioc/OCSMesh_MPI"
CONDA_BASE="/work2/noaa/nos-surge/felicioc/envs/miniconda3"
CONDA_ENV="ocsmesh_mpi_test"
SCRIPT_DIR="${PROJ}/ocsmesh_mpi_test"
STOFS_SHAPEFILE="${PROJ}/inputs/stofs3.shp"
DEM_OUT_DIR="${PROJ}/stofs_dems"

# ── Profile A manifest: 3 CUDEM tiles (full recipe, serial_mp only) ──────────
PROFILE_A_N_CUDEM="${PROFILE_A_N_CUDEM:-3}"
PROFILE_A_MANIFEST="${SCRIPT_DIR}/dem_manifest_profile_a.json"

# ── Profile B manifest: 18 CUDEM tiles (skip-constraints, all modes) ─────────
PROFILE_B_N_CUDEM="${PROFILE_B_N_CUDEM:-18}"
PROFILE_B_MANIFEST="${SCRIPT_DIR}/dem_manifest_profile_b.json"

# ── Smoke manifests ───────────────────────────────────────────────────────────
FULL_SMOKE_MANIFEST="${SCRIPT_DIR}/dem_manifest_smoke.json"    # 38 CUDEM tiles
SMOKE15_MANIFEST="${SCRIPT_DIR}/dem_manifest_smoke15.json"     # 14 CUDEM tiles
SMOKE7_MANIFEST="${SCRIPT_DIR}/dem_manifest_smoke7.json"       #  6 CUDEM tiles

# ── Profile C manifests: Anas's new operation configs ────────────────────────
# Thin-rank variant: 1 rank per tile.
#   38 CUDEM tiles + 1 GEBCO -> 39 ranks (1 manager + 38 workers).
#   Use FULL_SMOKE_MANIFEST (38 tiles already on disk).
PROFILE_C_THIN_MANIFEST="${FULL_SMOKE_MANIFEST}"
PROFILE_C_THIN_RANKS=39        # 1 manager + 38 workers
PROFILE_C_THIN_NTASKS=39
PROFILE_C_THIN_CPUS_PER_TASK=1

# Fat-rank variant: 8 worker ranks x 8 cores each = 64 cores.
#   Needs 8 CUDEM tiles (+ 1 GEBCO = 9 total).
#   Build from smoke manifest if not present.
PROFILE_C_N_CUDEM_FAT="${PROFILE_C_N_CUDEM_FAT:-8}"
PROFILE_C_FAT_MANIFEST="${SCRIPT_DIR}/dem_manifest_profile_c_fat.json"
PROFILE_C_FAT_RANKS=9          # 1 manager + 8 workers
PROFILE_C_FAT_NTASKS=9
PROFILE_C_FAT_CPUS_PER_TASK=8

# Default MANIFEST for general scripts
if [ -f "${PROFILE_B_MANIFEST}" ]; then
    MANIFEST="${PROFILE_B_MANIFEST}"
else
    MANIFEST="${SMOKE15_MANIFEST}"
fi

# ── Recipe knobs ─────────────────────────────────────────────────────────────
# Individual scripts override these as needed. Defaults are Profile B values
# (skip slow stages to isolate meshdata dispatch).
LIGHT_FEATURES="${LIGHT_FEATURES:-0}"
LIGHT_FLAG=""
[ "${LIGHT_FEATURES}" = "1" ] && LIGHT_FLAG="--light-features"

SKIP_TOPOFUNC="${SKIP_TOPOFUNC:-0}"
SKIP_TOPOFUNC_FLAG=""
[ "${SKIP_TOPOFUNC}" = "1" ] && SKIP_TOPOFUNC_FLAG="--skip-topofunc"

SKIP_CONSTRAINTS="${SKIP_CONSTRAINTS:-0}"
SKIP_CONSTRAINTS_FLAG=""
[ "${SKIP_CONSTRAINTS}" = "1" ] && SKIP_CONSTRAINTS_FLAG="--skip-constraints"

SKIP_BOX_REFINEMENTS="${SKIP_BOX_REFINEMENTS:-1}"
SKIP_BOX_REFINEMENTS_FLAG=""
[ "${SKIP_BOX_REFINEMENTS}" = "1" ] && \
    SKIP_BOX_REFINEMENTS_FLAG="--skip-box-refinements"

FULL_PIPELINE="${FULL_PIPELINE:-1}"
FULL_PIPELINE_FLAG=""
[ "${FULL_PIPELINE}" = "1" ] && FULL_PIPELINE_FLAG="--full-pipeline"

# Config F / G flags (new — Anas's fully MPI-dispatched configs)
CONFIG_F="${CONFIG_F:-0}"
CONFIG_F_FLAG=""
[ "${CONFIG_F}" = "1" ] && CONFIG_F_FLAG="--config-f"

CONFIG_G="${CONFIG_G:-0}"
CONFIG_G_FLAG=""
[ "${CONFIG_G}" = "1" ] && CONFIG_G_FLAG="--config-g"

ALL_FLAGS="${LIGHT_FLAG} ${SKIP_TOPOFUNC_FLAG} ${SKIP_CONSTRAINTS_FLAG} \
${SKIP_BOX_REFINEMENTS_FLAG} ${FULL_PIPELINE_FLAG} \
${CONFIG_F_FLAG} ${CONFIG_G_FLAG}"

# ── Global mesh size bounds ───────────────────────────────────────────────────
HMIN="${HMIN:-1000.0}"
HMAX="${HMAX:-7000.0}"

# ── Worker counts ─────────────────────────────────────────────────────────────
# Standard single-node: 80 cores, 1 manager + 79 workers
NPROCS="${NPROCS:-79}"

# Thin-rank MPI: 1 core per rank, 1 rank per tile
# (use PROFILE_C_THIN_NTASKS for --ntasks)
NPROCS_THIN=1

# Fat-rank MPI hybrid: 8 workers x 8 cores each
# (use PROFILE_C_FAT_NTASKS / PROFILE_C_FAT_CPUS_PER_TASK for SLURM)
NPROCS_FAT="${PROFILE_C_FAT_CPUS_PER_TASK}"

# ── Environment loader ────────────────────────────────────────────────────────
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

# ── Fat-manifest builder ──────────────────────────────────────────────────────
# Builds the 8-tile fat-rank manifest if not already present.
ensure_fat_manifest() {
    if [ ! -f "${PROFILE_C_FAT_MANIFEST}" ]; then
        if [ ! -f "${FULL_SMOKE_MANIFEST}" ]; then
            echo "ERROR: ${FULL_SMOKE_MANIFEST} not found." \
                 "Run download_dems.py first."
            exit 1
        fi
        echo "--- Building fat-rank manifest " \
             "(${PROFILE_C_N_CUDEM_FAT} CUDEM tiles) ---"
        srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/trim_manifest.py" \
            --in  "${FULL_SMOKE_MANIFEST}" \
            --out "${PROFILE_C_FAT_MANIFEST}" \
            --n-cudem "${PROFILE_C_N_CUDEM_FAT}"
    fi
}

# ── Config printer ────────────────────────────────────────────────────────────
print_final_config() {
    echo "================================================================="
    echo " OCSMesh benchmark config"
    echo "   Job ID           : ${SLURM_JOB_ID:-<none>}"
    echo "   Nodes            : ${SLURM_NODELIST:-<none>}"
    echo "   Manifest         : ${MANIFEST}"
    echo "   LIGHT_FEATURES   : ${LIGHT_FEATURES}"
    echo "   SKIP_TOPOFUNC    : ${SKIP_TOPOFUNC}"
    echo "   SKIP_CONSTRAINTS : ${SKIP_CONSTRAINTS}"
    echo "   SKIP_BOX_REFS    : ${SKIP_BOX_REFINEMENTS}"
    echo "   CONFIG_F         : ${CONFIG_F}"
    echo "   CONFIG_G         : ${CONFIG_G}"
    echo "   FULL_PIPELINE    : ${FULL_PIPELINE}"
    echo "   hmin / hmax      : ${HMIN} / ${HMAX}"
    echo "   NPROCS           : ${NPROCS}"
    echo "================================================================="
}
