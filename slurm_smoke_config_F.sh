#!/bin/bash
# =============================================================================
# SLURM job script — Smoke-test Config F (dedicated, 8h)
# =============================================================================
# Config F: "all MPI-dispatched ops, no shape bottleneck"
#   flag: --config-f
#
# Maps to Anas's Config E. Every operation in this config is fully
# MPI-dispatched across worker ranks:
#
#   Operation                    MPI dispatch?
#   ─────────────────────────────────────────
#   add_subtidal_flow_limiter    YES (all tiles)
#   add_constant_value           YES (all tiles)
#   add_topo_bound_constraint    YES (all tiles)
#   add_topo_func_constraint     YES (all tiles, named fn _half_depth)
#   add_contour                  YES (all tiles)
#   add_channel                  YES (all tiles)
#   add_patch                    NOT INCLUDED
#   add_feature                  NOT INCLUDED
#
# add_patch / add_feature are intentionally excluded. Including them
# introduces a coordinator-only serialization step that stalls all worker
# ranks and masks the mpi_hybrid vs mpi_no_pool difference. Config G adds
# them back once we have the Config F baseline.
#
# Runs THREE execution modes for direct comparison:
#   serial_mp    — baseline (serial execution_mode, NPROCS Pool workers)
#   mpi_no_pool  — pure MPI, 1 core per rank
#   mpi_hybrid   — MPI + internal Pool per rank (auto core budget)
#
# Two SLURM variants:
#   Variant 1 — THIN ranks: 1 core/rank, 1 rank/tile (38 tiles + 1 mgr)
#     Shows MPI inter-tile scaling. mpi_hybrid degenerates to mpi_no_pool
#     at 1 core/rank — confirms parity and validates the dispatch path.
#     --ntasks=39 --cpus-per-task=1
#
#   Variant 2 — FAT ranks: 8 workers x 8 cores/rank (8 tiles + 1 mgr)
#     Shows true mpi_hybrid advantage: each rank gets an 8-core internal
#     Pool for intra-tile parallelism on the 8112x8112 STOFS DEMs.
#     --ntasks=9 --cpus-per-task=8
#
# Real STOFS DEMs (8112x8112 px per tile) are ~29x heavier than Anas's
# synthetic 1500x1500 tiles. Per-tile work is substantially larger, which
# favors MPI more strongly and makes mpi_hybrid more impactful.
#
# Expected walltime (38 CUDEM tiles, thin variant):
#   serial_mp    ~2-3h  (38 tiles serial, flow+const+constraints+contour)
#   mpi_no_pool  ~15-30 min
#   mpi_hybrid   ~15-30 min (same as no_pool at 1 core/rank — expected)
#   total        ~3-4h — within 8h
#
# Expected walltime (8 CUDEM tiles, fat variant):
#   serial_mp    ~30-60 min
#   mpi_no_pool  ~5-10 min
#   mpi_hybrid   ~3-6 min  (8-core Pool per rank should show speedup)
#   total        ~1-2h — within 8h
#
# Submit:
#   sbatch slurm_smoke_config_F.sh
#   VARIANT=fat sbatch slurm_smoke_config_F.sh
#   MODES="serial_mp mpi_no_pool" sbatch slurm_smoke_config_F.sh
# =============================================================================

#SBATCH --job-name=ocsmesh_smoke_F
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/smoke_F_%j.out
#SBATCH --error=logs/smoke_F_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

# ── Config ────────────────────────────────────────────────────────────────────
CFG="F"
CFG_FLAGS="--config-f"
CFG_DESC="all MPI-dispatched ops, no shape bottleneck (maps to Anas Config E)"
MODES="${MODES:-serial_mp mpi_no_pool mpi_hybrid}"

# Variant: thin (default) or fat
VARIANT="${VARIANT:-thin}"

RESULTS_ROOT="${PROJ}/results/smoke_F_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_ROOT}" logs

load_ocsmesh_env

echo "================================================================="
echo " OCSMesh Smoke-Test — Config ${CFG}"
echo " ${CFG_DESC}"
echo " Job ID   : ${SLURM_JOB_ID}"
echo " Node     : ${SLURM_NODELIST}"
echo " Date     : $(date)"
echo " Variant  : ${VARIANT}"
echo " Modes    : ${MODES}"
echo " Flags    : ${CFG_FLAGS}"
echo " Results  : ${RESULTS_ROOT}"
echo "================================================================="

srun --mpi=pmi2 -n 1 python -c \
    "from mpi4py import MPI; print('mpi4py:', MPI.Get_version())"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── Variant-specific settings ─────────────────────────────────────────────────
if [ "${VARIANT}" = "fat" ]; then
    echo ""
    echo "--- FAT rank variant: ${PROFILE_C_FAT_RANKS} ranks x " \
         "${PROFILE_C_FAT_CPUS_PER_TASK} cores/rank ---"
    ensure_fat_manifest
    MANIFEST="${PROFILE_C_FAT_MANIFEST}"
    MPI_NTASKS="${PROFILE_C_FAT_RANKS}"
    MPI_CPUS="${PROFILE_C_FAT_CPUS_PER_TASK}"
    # serial_mp baseline uses same total core budget as fat MPI
    # (8 workers x 8 cores = 64 cores)
    SERIAL_NPROCS=$(( (MPI_NTASKS - 1) * MPI_CPUS ))
else
    echo ""
    echo "--- THIN rank variant: ${PROFILE_C_THIN_RANKS} ranks x 1 core/rank ---"
    MANIFEST="${PROFILE_C_THIN_MANIFEST}"
    MPI_NTASKS="${PROFILE_C_THIN_RANKS}"
    MPI_CPUS="${PROFILE_C_THIN_CPUS_PER_TASK}"
    SERIAL_NPROCS="${NPROCS}"
fi

echo "    Manifest  : ${MANIFEST}"
echo "    MPI ranks : ${MPI_NTASKS}"
echo "    Cores/rank: ${MPI_CPUS}"
echo "    serial_mp nprocs: ${SERIAL_NPROCS}"

# ── Run all requested modes ───────────────────────────────────────────────────
for MODE in ${MODES}; do
    OUT_DIR="${RESULTS_ROOT}/${VARIANT}/${MODE}"
    MODE_TMPDIR="${OUT_DIR}/tmp"
    mkdir -p "${OUT_DIR}" "${MODE_TMPDIR}"

    echo ""
    echo "--- Config ${CFG} / ${VARIANT} / mode ${MODE} ---"

    if [ "${MODE}" = "serial_mp" ]; then
        # serial_mp: single rank, Pool workers
        srun --mpi=pmi2 -n 1 bash -c "\
            export TMPDIR='${MODE_TMPDIR}'; \
            exec python '${SCRIPT_DIR}/run_benchmark.py' \
                --manifest    '${MANIFEST}' \
                --shapefile   '${STOFS_SHAPEFILE}' \
                --out-dir     '${OUT_DIR}' \
                --nprocs      '${SERIAL_NPROCS}' \
                --hmin        '${HMIN}' \
                --hmax        '${HMAX}' \
                --modes       serial_mp \
                ${CFG_FLAGS}"

    elif [ "${MODE}" = "mpi_no_pool" ]; then
        # mpi_no_pool: MPI ranks, 1 core per rank
        srun --mpi=pmi2 \
             --ntasks="${MPI_NTASKS}" \
             --cpus-per-task=1 \
             bash -c "\
            export TMPDIR='${MODE_TMPDIR}'; \
            exec python '${SCRIPT_DIR}/run_benchmark.py' \
                --manifest    '${MANIFEST}' \
                --shapefile   '${STOFS_SHAPEFILE}' \
                --out-dir     '${OUT_DIR}' \
                --nprocs      1 \
                --hmin        '${HMIN}' \
                --hmax        '${HMAX}' \
                --modes       mpi_no_pool \
                ${CFG_FLAGS}"

    elif [ "${MODE}" = "mpi_hybrid" ]; then
        # mpi_hybrid: MPI ranks, auto cores per rank
        srun --mpi=pmi2 \
             --ntasks="${MPI_NTASKS}" \
             --cpus-per-task="${MPI_CPUS}" \
             --overcommit \
             bash -c "\
            export TMPDIR='${MODE_TMPDIR}'; \
            exec python '${SCRIPT_DIR}/run_benchmark.py' \
                --manifest    '${MANIFEST}' \
                --shapefile   '${STOFS_SHAPEFILE}' \
                --out-dir     '${OUT_DIR}' \
                --nprocs      '${MPI_CPUS}' \
                --hmin        '${HMIN}' \
                --hmax        '${HMAX}' \
                --modes       mpi_hybrid \
                ${CFG_FLAGS}"

    else
        # Any other mode (mpi, parallel, serial_true)
        srun --mpi=pmi2 \
             --ntasks="${MPI_NTASKS}" \
             --cpus-per-task="${MPI_CPUS}" \
             --overcommit \
             bash -c "\
            export TMPDIR='${MODE_TMPDIR}'; \
            exec python '${SCRIPT_DIR}/run_benchmark.py' \
                --manifest    '${MANIFEST}' \
                --shapefile   '${STOFS_SHAPEFILE}' \
                --out-dir     '${OUT_DIR}' \
                --nprocs      '${SERIAL_NPROCS}' \
                --hmin        '${HMIN}' \
                --hmax        '${HMAX}' \
                --modes       ${MODE} \
                ${CFG_FLAGS}"
    fi
done

# ── Report ────────────────────────────────────────────────────────────────────
REPORT_DIRS=""
for MODE in ${MODES}; do
    REPORT_DIRS="${REPORT_DIRS} ${RESULTS_ROOT}/${VARIANT}/${MODE}"
done

echo ""
echo "--- Generating report for Config ${CFG} (${VARIANT}) ---"
srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
    --results-dir ${REPORT_DIRS} \
    --out         "${RESULTS_ROOT}/report_config_${CFG}_${VARIANT}.txt"

echo ""
echo "================================================================="
echo " Config ${CFG} (${VARIANT}) complete."
echo " Results : ${RESULTS_ROOT}/${VARIANT}"
echo " Report  : ${RESULTS_ROOT}/report_config_${CFG}_${VARIANT}.txt"
echo "================================================================="
