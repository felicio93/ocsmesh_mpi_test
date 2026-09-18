#!/bin/bash
# =============================================================================
# SLURM job script — Anas MPI operations matrix benchmark
# =============================================================================
# Runs Configs F and G across three execution modes (serial_mp,
# mpi_no_pool, mpi_hybrid) in both thin-rank and fat-rank variants.
# This is the primary script for benchmarking all operations Anas
# parallelized in his PR on real STOFS DEMs.
#
# Matrix structure:
#
#   Config F (no shapes)  x  {serial_mp, mpi_no_pool, mpi_hybrid}
#                         x  {thin, fat}
#
#   Config G (with shapes) x {serial_mp, mpi_no_pool, mpi_hybrid}
#                          x {thin, fat}
#
# Total: 2 configs x 3 modes x 2 variants = 12 runs
#
# What each comparison tells you:
#
#   Config F serial_mp vs mpi_no_pool (thin):
#     Pure MPI inter-tile speedup for flow+const+constraints+contour/channel
#     on real 8112x8112 STOFS tiles. No shape work. Cleanest speedup number.
#
#   Config F mpi_no_pool vs mpi_hybrid (fat):
#     Does an 8-core internal Pool per rank speed up the per-tile work?
#     On 8112x8112 tiles the intra-tile parallelism in contour/channel
#     should be significant.
#
#   Config G vs Config F (same mode):
#     Pure cost of add_patch + add_feature via _user_shapes_task_worker
#     on real STOFS tiles. Isolates Anas's shape parallelization.
#
#   Config G mpi_no_pool vs mpi_hybrid (fat):
#     Does a larger per-rank Pool help the KDTree distance expansion
#     inside add_patch / add_feature?
#
# Run order (fastest first so critical results arrive early):
#   1. Config F thin  — mpi_no_pool (fastest, ~15 min)
#   2. Config F thin  — mpi_hybrid  (~15 min, should match no_pool)
#   3. Config G thin  — mpi_no_pool (~20-40 min)
#   4. Config G thin  — mpi_hybrid  (~20-40 min)
#   5. Config F fat   — mpi_no_pool (~5-10 min)
#   6. Config F fat   — mpi_hybrid  (~3-6 min — KEY result)
#   7. Config G fat   — mpi_no_pool (~8-15 min)
#   8. Config G fat   — mpi_hybrid  (~4-8 min — KEY result)
#   9. Config F thin  — serial_mp   (~2-3h — baseline, runs last)
#  10. Config G thin  — serial_mp   (~3-5h — baseline with shapes)
#  11. Config F fat   — serial_mp   (~30-60 min)
#  12. Config G fat   — serial_mp   (~45-90 min)
#
# Serial baselines run last so MPI results arrive first. If the job
# approaches 8h before serial_mp finishes, cancel and resubmit serial_mp
# only using slurm_smoke_config_F.sh / slurm_smoke_config_G.sh.
#
# Submit options:
#   sbatch slurm_anas_matrix.sh
#
#   # Only Config F (skip Config G):
#   CONFIGS="F" sbatch slurm_anas_matrix.sh
#
#   # Only MPI modes (skip serial_mp baselines — fast validation):
#   MODES="mpi_no_pool mpi_hybrid" sbatch slurm_anas_matrix.sh
#
#   # Only fat variant:
#   VARIANTS="fat" sbatch slurm_anas_matrix.sh
#
#   # MPI modes only, fat variant, both configs (full key result):
#   CONFIGS="F G" MODES="mpi_no_pool mpi_hybrid" VARIANTS="fat" \
#       sbatch slurm_anas_matrix.sh
#
#   # Full matrix (all 12 runs):
#   CONFIGS="F G" MODES="serial_mp mpi_no_pool mpi_hybrid" \
#       VARIANTS="thin fat" sbatch slurm_anas_matrix.sh
# =============================================================================

#SBATCH --job-name=ocsmesh_anas_matrix
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/anas_matrix_%j.out
#SBATCH --error=logs/anas_matrix_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

# ── Matrix selection (override at submit time) ────────────────────────────────
CONFIGS="${CONFIGS:-F G}"
MODES="${MODES:-mpi_no_pool mpi_hybrid serial_mp}"
VARIANTS="${VARIANTS:-thin fat}"

RESULTS_ROOT="${PROJ}/results/anas_matrix_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_ROOT}" logs

load_ocsmesh_env

echo "================================================================="
echo " OCSMesh Anas MPI Operations Matrix Benchmark"
echo " Job ID   : ${SLURM_JOB_ID}"
echo " Node     : ${SLURM_NODELIST}"
echo " Date     : $(date)"
echo " Configs  : ${CONFIGS}"
echo " Modes    : ${MODES}"
echo " Variants : ${VARIANTS}"
echo " Results  : ${RESULTS_ROOT}"
echo "================================================================="
echo ""
echo " What this measures:"
echo "   Config F — flow+const+constraints+contour/channel (no shapes)"
echo "              All ops MPI-dispatched. No coordinator bottleneck."
echo "              Maps to Anas Config E."
echo "   Config G — Config F + add_patch + add_feature (BOX2+BOX3)"
echo "              _user_shapes_task_worker. Maps to Anas Config F."
echo "   thin     — 1 core/rank, 1 rank/tile. Inter-tile MPI scaling."
echo "   fat      — 8 cores/rank, 8 worker ranks. True hybrid advantage."
echo "================================================================="

srun --mpi=pmi2 -n 1 python -c \
    "from mpi4py import MPI; print('mpi4py:', MPI.Get_version())"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── Map config letter to run_benchmark.py flag ────────────────────────────────
config_flag() {
    case "$1" in
        F) echo "--config-f" ;;
        G) echo "--config-g" ;;
        *) echo "__INVALID__" ;;
    esac
}

config_desc() {
    case "$1" in
        F) echo "all MPI-dispatched ops, no shapes (Anas Config E)" ;;
        G) echo "full pipeline with shapes (Anas Config F)" ;;
        *) echo "INVALID" ;;
    esac
}

# ── Build fat manifest if needed ──────────────────────────────────────────────
if echo "${VARIANTS}" | grep -q "fat"; then
    ensure_fat_manifest
fi

# ── Run order: MPI modes first (fast), serial_mp last (slow) ─────────────────
# Reorder modes so serial_mp always runs last regardless of MODES order.
MPI_ONLY_MODES=""
SERIAL_MODES=""
for MODE in ${MODES}; do
    if [ "${MODE}" = "serial_mp" ] || [ "${MODE}" = "serial_true" ] || \
       [ "${MODE}" = "parallel" ]; then
        SERIAL_MODES="${SERIAL_MODES} ${MODE}"
    else
        MPI_ONLY_MODES="${MPI_ONLY_MODES} ${MODE}"
    fi
done
ORDERED_MODES="${MPI_ONLY_MODES} ${SERIAL_MODES}"

echo ""
echo "Run order: ${ORDERED_MODES}"
echo ""

# ── Main matrix loop ──────────────────────────────────────────────────────────
for VARIANT in ${VARIANTS}; do

    # Set variant-specific parameters
    if [ "${VARIANT}" = "fat" ]; then
        MANIFEST="${PROFILE_C_FAT_MANIFEST}"
        MPI_NTASKS="${PROFILE_C_FAT_RANKS}"
        MPI_CPUS="${PROFILE_C_FAT_CPUS_PER_TASK}"
        SERIAL_NPROCS=$(( (MPI_NTASKS - 1) * MPI_CPUS ))
    else
        MANIFEST="${PROFILE_C_THIN_MANIFEST}"
        MPI_NTASKS="${PROFILE_C_THIN_RANKS}"
        MPI_CPUS="${PROFILE_C_THIN_CPUS_PER_TASK}"
        SERIAL_NPROCS="${NPROCS}"
    fi

    echo "#################################################################"
    echo "# VARIANT: ${VARIANT}"
    echo "#   Manifest  : ${MANIFEST}"
    echo "#   MPI ranks : ${MPI_NTASKS}  (1 mgr + $((MPI_NTASKS-1)) workers)"
    echo "#   Cores/rank: ${MPI_CPUS}"
    echo "#   serial_mp nprocs: ${SERIAL_NPROCS}"
    echo "#################################################################"

    for CFG in ${CONFIGS}; do
        FLAG="$(config_flag "${CFG}")"
        if [ "${FLAG}" = "__INVALID__" ]; then
            echo "WARNING: unknown config '${CFG}', skipping."
            continue
        fi

        echo ""
        echo "========================================="
        echo "  Config ${CFG}: $(config_desc "${CFG}")"
        echo "  flag: ${FLAG}"
        echo "========================================="

        for MODE in ${ORDERED_MODES}; do
            OUT_DIR="${RESULTS_ROOT}/config_${CFG}/${VARIANT}/${MODE}"
            MODE_TMPDIR="${OUT_DIR}/tmp"
            mkdir -p "${OUT_DIR}" "${MODE_TMPDIR}"

            echo ""
            echo "--- Config ${CFG} / ${VARIANT} / ${MODE} ---"

            if [ "${MODE}" = "serial_mp" ] || \
               [ "${MODE}" = "serial_true" ] || \
               [ "${MODE}" = "parallel" ]; then
                # Rank-0-only modes
                srun --mpi=pmi2 -n 1 bash -c "\
                    export TMPDIR='${MODE_TMPDIR}'; \
                    exec python '${SCRIPT_DIR}/run_benchmark.py' \
                        --manifest  '${MANIFEST}' \
                        --shapefile '${STOFS_SHAPEFILE}' \
                        --out-dir   '${OUT_DIR}' \
                        --nprocs    '${SERIAL_NPROCS}' \
                        --hmin      '${HMIN}' \
                        --hmax      '${HMAX}' \
                        --modes     ${MODE} \
                        ${FLAG}"

            elif [ "${MODE}" = "mpi_no_pool" ]; then
                # Pure MPI: 1 core per rank regardless of variant
                srun --mpi=pmi2 \
                     --ntasks="${MPI_NTASKS}" \
                     --cpus-per-task=1 \
                     bash -c "\
                    export TMPDIR='${MODE_TMPDIR}'; \
                    exec python '${SCRIPT_DIR}/run_benchmark.py' \
                        --manifest  '${MANIFEST}' \
                        --shapefile '${STOFS_SHAPEFILE}' \
                        --out-dir   '${OUT_DIR}' \
                        --nprocs    1 \
                        --hmin      '${HMIN}' \
                        --hmax      '${HMAX}' \
                        --modes     mpi_no_pool \
                        ${FLAG}"

            elif [ "${MODE}" = "mpi_hybrid" ]; then
                # Hybrid MPI: full core budget per rank
                srun --mpi=pmi2 \
                     --ntasks="${MPI_NTASKS}" \
                     --cpus-per-task="${MPI_CPUS}" \
             --overcommit \
                     bash -c "\
                    export TMPDIR='${MODE_TMPDIR}'; \
                    exec python '${SCRIPT_DIR}/run_benchmark.py' \
                        --manifest  '${MANIFEST}' \
                        --shapefile '${STOFS_SHAPEFILE}' \
                        --out-dir   '${OUT_DIR}' \
                        --nprocs    '${MPI_CPUS}' \
                        --hmin      '${HMIN}' \
                        --hmax      '${HMAX}' \
                        --modes     mpi_hybrid \
                        ${FLAG}"

            else
                # mpi or any other mode
                srun --mpi=pmi2 \
                     --ntasks="${MPI_NTASKS}" \
                     --cpus-per-task="${MPI_CPUS}" \
             --overcommit \
                     bash -c "\
                    export TMPDIR='${MODE_TMPDIR}'; \
                    exec python '${SCRIPT_DIR}/run_benchmark.py' \
                        --manifest  '${MANIFEST}' \
                        --shapefile '${STOFS_SHAPEFILE}' \
                        --out-dir   '${OUT_DIR}' \
                        --nprocs    '${SERIAL_NPROCS}' \
                        --hmin      '${HMIN}' \
                        --hmax      '${HMAX}' \
                        --modes     ${MODE} \
                        ${FLAG}"
            fi

        done  # MODE

        # ── Per-config per-variant report ─────────────────────────────────
        REPORT_DIRS=""
        for MODE in ${ORDERED_MODES}; do
            REPORT_DIRS="${REPORT_DIRS} \
                ${RESULTS_ROOT}/config_${CFG}/${VARIANT}/${MODE}"
        done

        echo ""
        echo "--- Report: Config ${CFG} / ${VARIANT} ---"
        srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
            --results-dir ${REPORT_DIRS} \
            --out \
            "${RESULTS_ROOT}/config_${CFG}/report_${CFG}_${VARIANT}.txt"

    done  # CFG

done  # VARIANT

# ── Combined cross-variant summary ────────────────────────────────────────────
# Merge all result dirs into one report for easy cross-comparison.
echo ""
echo "--- Combined report (all configs, all variants) ---"
ALL_DIRS=""
for CFG in ${CONFIGS}; do
    for VARIANT in ${VARIANTS}; do
        for MODE in ${ORDERED_MODES}; do
            D="${RESULTS_ROOT}/config_${CFG}/${VARIANT}/${MODE}"
            [ -d "${D}" ] && ALL_DIRS="${ALL_DIRS} ${D}"
        done
    done
done

srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
    --results-dir ${ALL_DIRS} \
    --out         "${RESULTS_ROOT}/report_anas_matrix_combined.txt"

echo ""
echo "================================================================="
echo " Anas matrix benchmark complete."
echo " Results root : ${RESULTS_ROOT}"
echo ""
echo " Per-config reports:"
for CFG in ${CONFIGS}; do
    for VARIANT in ${VARIANTS}; do
        echo "   Config ${CFG} / ${VARIANT}: " \
             "${RESULTS_ROOT}/config_${CFG}/report_${CFG}_${VARIANT}.txt"
    done
done
echo ""
echo " Combined report:"
echo "   ${RESULTS_ROOT}/report_anas_matrix_combined.txt"
echo ""
echo " Key comparisons to read first:"
echo "   1. Config F fat: serial_mp vs mpi_no_pool vs mpi_hybrid"
echo "      -> Pure MPI + hybrid speedup, no shape bottleneck"
echo "   2. Config G fat: serial_mp vs mpi_no_pool vs mpi_hybrid"
echo "      -> Shape parallelization (_user_shapes_task_worker) cost"
echo "   3. Config G vs Config F (same mode, fat variant)"
echo "      -> Isolated cost of add_patch + add_feature on real DEMs"
echo "================================================================="
