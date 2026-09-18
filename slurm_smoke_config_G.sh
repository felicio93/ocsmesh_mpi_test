#!/bin/bash
# =============================================================================
# SLURM job script — Smoke-test Config G (dedicated, 8h)
# =============================================================================
# Config G: "full pipeline, all ops MPI-dispatched including shapes"
#   flag: --config-g
#
# Maps to Anas's Config F. Extends Config F by adding shape-based
# refinements (add_patch + add_feature) which are the operations Anas
# parallelized via _user_shapes_task_worker. Every operation is fully
# MPI-dispatched:
#
#   Operation                    MPI dispatch?
#   ─────────────────────────────────────────
#   add_subtidal_flow_limiter    YES (all tiles)
#   add_constant_value           YES (all tiles)
#   add_topo_bound_constraint    YES (all tiles)
#   add_topo_func_constraint     YES (all tiles, named fn _half_depth)
#   add_contour                  YES (all tiles)
#   add_channel                  YES (all tiles)
#   add_patch                    YES (BOX2 SC/GA + BOX3 Gulf Coast)
#   add_feature                  YES (BOX2 + BOX3 line features)
#
# BOX2 = (-80..-77, 31..35) SC/GA coast, target_size=1000 m
# BOX3 = (-90..-86, 28..31) Gulf Coast,  target_size=800 m
# Both confirmed inside STOFS domain. Both go through
# _user_shapes_task_worker — the new worker Anas added in this PR.
#
# What Config G vs Config F tells you:
#   The pure cost of add_patch + add_feature on the real STOFS 8112x8112
#   tiles when dispatched via _user_shapes_task_worker. If mpi_no_pool
#   and mpi_hybrid show similar times to Config F, the shape dispatch
#   is well-parallelized. If they are slower, the KDTree distance
#   expansion inside each worker is the bottleneck.
#
# Runs THREE execution modes:
#   serial_mp    — baseline
#   mpi_no_pool  — pure MPI, 1 core per rank
#   mpi_hybrid   — MPI + internal Pool per rank (auto core budget)
#
# Two SLURM variants:
#   Variant 1 — THIN ranks: 1 core/rank, 1 rank/tile
#     --ntasks=39 --cpus-per-task=1
#     (38 CUDEM tiles from smoke manifest + 1 manager)
#
#   Variant 2 — FAT ranks: 8 workers x 8 cores/rank
#     --ntasks=9 --cpus-per-task=8
#     (8 CUDEM tiles from fat manifest + 1 manager)
#     This is the most informative variant for mpi_hybrid because
#     each rank can use 8 cores for the KDTree distance expansion
#     inside add_patch / add_feature.
#
# Expected walltime (38 CUDEM tiles, thin variant):
#   serial_mp    ~3-5h  (shape ops add ~1-2h on top of Config F)
#   mpi_no_pool  ~20-40 min
#   mpi_hybrid   ~20-40 min (same as no_pool at 1 core/rank)
#   total        ~4-6h — within 8h
#
# Expected walltime (8 CUDEM tiles, fat variant):
#   serial_mp    ~45-90 min
#   mpi_no_pool  ~8-15 min
#   mpi_hybrid   ~4-8 min  (8-core Pool handles KDTree expansion)
#   total        ~1-3h — within 8h
#
# NOTE: If thin variant serial_mp overruns 8h, resubmit on windfall:
#   #SBATCH --partition=windfall
#   #SBATCH --time=24:00:00
#
# Submit:
#   sbatch slurm_smoke_config_G.sh
#   VARIANT=fat sbatch slurm_smoke_config_G.sh
#   MODES="serial_mp mpi_no_pool" sbatch slurm_smoke_config_G.sh
#   VARIANT=fat MODES="mpi_no_pool mpi_hybrid" sbatch slurm_smoke_config_G.sh
# =============================================================================

#SBATCH --job-name=ocsmesh_smoke_G
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/smoke_G_%j.out
#SBATCH --error=logs/smoke_G_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

# ── Config ────────────────────────────────────────────────────────────────────
CFG="G"
CFG_FLAGS="--config-g"
CFG_DESC="full pipeline all ops MPI-dispatched including shapes (maps to Anas Config F)"
MODES="${MODES:-serial_mp mpi_no_pool mpi_hybrid}"

# Variant: thin (default) or fat
VARIANT="${VARIANT:-thin}"

RESULTS_ROOT="${PROJ}/results/smoke_G_${SLURM_JOB_ID}"
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
echo " NOTE: BOX2=(-80,-77,31,35) SC/GA + BOX3=(-90,-86,28,31) Gulf Coast"
echo "       Both use _user_shapes_task_worker (Anas's new worker)."
echo "================================================================="

srun --mpi=pmi2 -n 1 python -c \
    "from mpi4py import MPI; print('mpi4py:', MPI.Get_version())"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── Variant-specific settings ─────────────────────────────────────────────────
if [ "${VARIANT}" = "fat" ]; then
    echo ""
    echo "--- FAT rank variant: ${PROFILE_C_FAT_RANKS} ranks x" \
         "${PROFILE_C_FAT_CPUS_PER_TASK} cores/rank ---"
    echo "    This is the most informative variant for mpi_hybrid:"
    echo "    each rank uses ${PROFILE_C_FAT_CPUS_PER_TASK} cores for"
    echo "    KDTree distance expansion inside add_patch / add_feature."
    ensure_fat_manifest
    MANIFEST="${PROFILE_C_FAT_MANIFEST}"
    MPI_NTASKS="${PROFILE_C_FAT_RANKS}"
    MPI_CPUS="${PROFILE_C_FAT_CPUS_PER_TASK}"
    SERIAL_NPROCS=$(( (MPI_NTASKS - 1) * MPI_CPUS ))
else
    echo ""
    echo "--- THIN rank variant: ${PROFILE_C_THIN_RANKS} ranks x 1 core/rank ---"
    MANIFEST="${PROFILE_C_THIN_MANIFEST}"
    MPI_NTASKS="${PROFILE_C_THIN_RANKS}"
    MPI_CPUS="${PROFILE_C_THIN_CPUS_PER_TASK}"
    SERIAL_NPROCS="${NPROCS}"
fi

echo "    Manifest        : ${MANIFEST}"
echo "    MPI ranks       : ${MPI_NTASKS}"
echo "    Cores/rank      : ${MPI_CPUS}"
echo "    serial_mp nprocs: ${SERIAL_NPROCS}"

# ── Run all requested modes ───────────────────────────────────────────────────
for MODE in ${MODES}; do
    OUT_DIR="${RESULTS_ROOT}/${VARIANT}/${MODE}"
    MODE_TMPDIR="${OUT_DIR}/tmp"
    mkdir -p "${OUT_DIR}" "${MODE_TMPDIR}"

    echo ""
    echo "--- Config ${CFG} / ${VARIANT} / mode ${MODE} ---"

    if [ "${MODE}" = "serial_mp" ]; then
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
