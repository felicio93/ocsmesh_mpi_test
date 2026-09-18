#!/bin/bash
# =============================================================================
# SLURM job script — smoke-test MATRIX benchmark
# =============================================================================
# Runs a 7-config cost ladder where each step adds exactly one cost class,
# so the contribution of each pipeline stage can be isolated. Configs A-E
# are the original ladder. Configs F and G are new, testing Anas's fully
# MPI-dispatched operations on real STOFS DEMs.
#
# The ladder:
#
#   Config   Operations                                  New vs previous
#   ──────   ──────────────────────────────────────────  ────────────────
#   A        flow_limiter (modulo, 1/tile)               baseline
#   B        flow_limiter + const_value (every tile)     +per-tile density
#   C        B + constraints (no topofunc)               +constraint stage
#   D        C + contour/channel + boxes                 +features stage
#   E        contour only (no constraints, no boxes)     isolates PR#251
#   F        flow+const+constraints+contour/channel      all MPI-dispatched
#            (all tiles, no boxes)                       no shape bottleneck
#   G        F + patch + feature (BOX2+BOX3)             +shape dispatch
#
# Execution modes per config:
#   Configs A-E: serial_mp / parallel / mpi       (original ladder)
#   Configs F-G: serial_mp / mpi_no_pool / mpi_hybrid  (Anas's modes)
#
# Rank configurations:
#   Configs A-E: standard single-node (80 ranks, 1 core/rank)
#   Config F-G thin: 39 ranks x 1 core  (38 CUDEM + 1 mgr)
#   Config F-G fat:  9 ranks x 8 cores  (8 CUDEM + 1 mgr)
#
# topo_func_constraint is INCLUDED in Configs F and G (Anas fixed the
# lambda guard — named function _half_depth is picklable). It is still
# EXCLUDED from Configs A-E because the old modulo scheme assigns it
# only to some tiles and the PR#251 branch handles it differently.
#
# Walltime guide (single node, exclusive):
#   A, B       : ~1-2h total (fast per-tile refs only)
#   C          : ~3-5h total (constraint stage via parallel Pool)
#   D          : ~6-8h total (global contour/channel — most expensive)
#   E          : ~2-3h total (contour only — isolates PR#251)
#   F thin     : ~3-4h total (all MPI-dispatched, 38 tiles)
#   F fat      : ~1-2h total (8 tiles, fat ranks)
#   G thin     : ~4-6h total (adds shape work)
#   G fat      : ~1-3h total (8 tiles, fat ranks)
#
#   Budget 8h for the full matrix. If D or G thin overrun, submit them
#   separately on windfall (slurm_smoke_config_D.sh / _G.sh).
#
# Submit options:
#   # Full matrix (all 7 configs, may approach 8h):
#   sbatch slurm_smoke_matrix.sh
#
#   # Original ladder only:
#   CONFIGS="A B C D E" sbatch slurm_smoke_matrix.sh
#
#   # Anas configs only:
#   CONFIGS="F G" sbatch slurm_smoke_matrix.sh
#
#   # Anas configs, MPI modes only (fast validation, ~1h):
#   CONFIGS="F G" ANAS_MODES="mpi_no_pool mpi_hybrid" \
#       sbatch slurm_smoke_matrix.sh
#
#   # Single config:
#   CONFIGS="F" sbatch slurm_smoke_matrix.sh
#   CONFIGS="G" VARIANTS="fat" sbatch slurm_smoke_matrix.sh
# =============================================================================

#SBATCH --job-name=ocsmesh_smoke_matrix
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/smoke_matrix_%j.out
#SBATCH --error=logs/smoke_matrix_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

# ── Matrix selection (override at submit time) ────────────────────────────────
CONFIGS="${CONFIGS:-A B C D E F G}"

# Modes for original configs A-E
ORIG_MODES="${ORIG_MODES:-mpi parallel serial_mp}"

# Modes for Anas configs F-G
ANAS_MODES="${ANAS_MODES:-mpi_no_pool mpi_hybrid serial_mp}"

# Variants for F-G (thin and/or fat)
VARIANTS="${VARIANTS:-thin fat}"

# Tile count for original configs A-E
N_CUDEM="${N_CUDEM:-14}"

RESULTS_ROOT="${PROJ}/results/smoke_matrix_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_ROOT}" logs

load_ocsmesh_env

echo "================================================================="
echo " OCSMesh Smoke-Matrix Benchmark"
echo " Job ID    : ${SLURM_JOB_ID}"
echo " Node      : ${SLURM_NODELIST}"
echo " Date      : $(date)"
echo " Configs   : ${CONFIGS}"
echo " A-E modes : ${ORIG_MODES}"
echo " F-G modes : ${ANAS_MODES}"
echo " F-G vars  : ${VARIANTS}"
echo " Results   : ${RESULTS_ROOT}"
echo "================================================================="

srun --mpi=pmi2 -n 1 python -c \
    "from mpi4py import MPI; print('mpi4py:', MPI.Get_version())"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export TMPDIR="${RESULTS_ROOT}/tmp"
mkdir -p "${TMPDIR}"

# ── Step 1: Download DEMs if needed ──────────────────────────────────────────
if [ ! -f "${FULL_SMOKE_MANIFEST}" ]; then
    echo ""
    echo "--- Downloading DEMs (MA_NH_ME) ---"
    python "${SCRIPT_DIR}/download_dems.py" \
        --out-dir  "${DEM_OUT_DIR}" \
        --manifest "${FULL_SMOKE_MANIFEST}" \
        --only MA_NH_ME
else
    echo ""
    echo "--- DEMs already downloaded ---"
fi

# ── Step 2: Build manifests ───────────────────────────────────────────────────
# Original A-E manifest (N_CUDEM tiles)
ORIG_MANIFEST="${SCRIPT_DIR}/dem_manifest_smoke$((N_CUDEM + 1)).json"
if [ ! -f "${ORIG_MANIFEST}" ]; then
    echo ""
    echo "--- Trimming manifest to $((N_CUDEM + 1)) tiles (Configs A-E) ---"
    srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/trim_manifest.py" \
        --in      "${FULL_SMOKE_MANIFEST}" \
        --out     "${ORIG_MANIFEST}" \
        --n-cudem "${N_CUDEM}"
else
    echo "--- Orig manifest found: ${ORIG_MANIFEST} ---"
fi

# Fat-rank manifest for F-G fat variant
if echo "${CONFIGS}" | grep -qE "[FG]" && \
   echo "${VARIANTS}" | grep -q "fat"; then
    ensure_fat_manifest
fi

# ── Map config to flags and description ──────────────────────────────────────
config_flags() {
    case "$1" in
        A) echo "--skip-constraints --skip-box-refinements --light-features" ;;
        B) echo "--all-fast-refinements" ;;
        C) echo "--skip-topofunc --light-features" ;;
        D) echo "--skip-topofunc" ;;
        E) echo "--skip-constraints --skip-box-refinements" ;;
        F) echo "--config-f" ;;
        G) echo "--config-g" ;;
        *) echo "__INVALID__" ;;
    esac
}

config_desc() {
    case "$1" in
        A) echo "no constraint, 1 ref/tile (modulo)" ;;
        B) echo "no constraint, 2 refs/tile (flow+const every tile)" ;;
        C) echo "constraints (no topofunc), no global features" ;;
        D) echo "full recipe minus topofunc (adds contour/channel+boxes)" ;;
        E) echo "no constraints, no boxes (isolates contour PR#251)" ;;
        F) echo "all MPI-dispatched, no shapes (Anas Config E)" ;;
        G) echo "full pipeline with shapes (Anas Config F)" ;;
        *) echo "INVALID" ;;
    esac
}

is_anas_config() {
    case "$1" in
        F|G) return 0 ;;
        *)   return 1 ;;
    esac
}

# ── Main matrix loop ──────────────────────────────────────────────────────────
for CFG in ${CONFIGS}; do
    FLAGS="$(config_flags "${CFG}")"
    if [ "${FLAGS}" = "__INVALID__" ]; then
        echo "WARNING: unknown config '${CFG}', skipping."
        continue
    fi

    echo ""
    echo "#################################################################"
    echo "# CONFIG ${CFG}: $(config_desc "${CFG}")"
    echo "#   flags: ${FLAGS}"
    echo "#################################################################"

    if is_anas_config "${CFG}"; then
        # ── Anas configs F and G ──────────────────────────────────────────
        # Run in both thin and fat variants with Anas modes.
        # MPI modes first, serial_mp last.

        MPI_FIRST=""
        SERIAL_LAST=""
        for MODE in ${ANAS_MODES}; do
            if [ "${MODE}" = "serial_mp" ] || \
               [ "${MODE}" = "serial_true" ] || \
               [ "${MODE}" = "parallel" ]; then
                SERIAL_LAST="${SERIAL_LAST} ${MODE}"
            else
                MPI_FIRST="${MPI_FIRST} ${MODE}"
            fi
        done
        ORDERED_ANAS="${MPI_FIRST} ${SERIAL_LAST}"

        for VARIANT in ${VARIANTS}; do
            if [ "${VARIANT}" = "fat" ]; then
                RUN_MANIFEST="${PROFILE_C_FAT_MANIFEST}"
                MPI_NTASKS="${PROFILE_C_FAT_RANKS}"
                MPI_CPUS="${PROFILE_C_FAT_CPUS_PER_TASK}"
                SERIAL_NPROCS=$(( (MPI_NTASKS - 1) * MPI_CPUS ))
            else
                RUN_MANIFEST="${PROFILE_C_THIN_MANIFEST}"
                MPI_NTASKS="${PROFILE_C_THIN_RANKS}"
                MPI_CPUS="${PROFILE_C_THIN_CPUS_PER_TASK}"
                SERIAL_NPROCS="${NPROCS}"
            fi

            echo ""
            echo "  --- Config ${CFG} / ${VARIANT} ---"
            echo "      Manifest  : ${RUN_MANIFEST}"
            echo "      MPI ranks : ${MPI_NTASKS}"
            echo "      Cores/rank: ${MPI_CPUS}"

            for MODE in ${ORDERED_ANAS}; do
                OUT_DIR="${RESULTS_ROOT}/config_${CFG}/${VARIANT}/${MODE}"
                MODE_TMPDIR="${OUT_DIR}/tmp"
                mkdir -p "${OUT_DIR}" "${MODE_TMPDIR}"

                echo ""
                echo "  --- Config ${CFG} / ${VARIANT} / ${MODE} ---"

                if [ "${MODE}" = "serial_mp" ] || \
                   [ "${MODE}" = "serial_true" ] || \
                   [ "${MODE}" = "parallel" ]; then
                    srun --mpi=pmi2 -n 1 bash -c "\
                        export TMPDIR='${MODE_TMPDIR}'; \
                        exec python '${SCRIPT_DIR}/run_benchmark.py' \
                            --manifest  '${RUN_MANIFEST}' \
                            --shapefile '${STOFS_SHAPEFILE}' \
                            --out-dir   '${OUT_DIR}' \
                            --nprocs    '${SERIAL_NPROCS}' \
                            --hmin      '${HMIN}' \
                            --hmax      '${HMAX}' \
                            --modes     ${MODE} \
                            ${FLAGS}"

                elif [ "${MODE}" = "mpi_no_pool" ]; then
                    srun --mpi=pmi2 \
                         --ntasks="${MPI_NTASKS}" \
                         --cpus-per-task=1 \
                         bash -c "\
                        export TMPDIR='${MODE_TMPDIR}'; \
                        exec python '${SCRIPT_DIR}/run_benchmark.py' \
                            --manifest  '${RUN_MANIFEST}' \
                            --shapefile '${STOFS_SHAPEFILE}' \
                            --out-dir   '${OUT_DIR}' \
                            --nprocs    1 \
                            --hmin      '${HMIN}' \
                            --hmax      '${HMAX}' \
                            --modes     mpi_no_pool \
                            ${FLAGS}"

                elif [ "${MODE}" = "mpi_hybrid" ]; then
                    srun --mpi=pmi2 \
                         --ntasks="${MPI_NTASKS}" \
                         --cpus-per-task="${MPI_CPUS}" \
             --overcommit \
                         bash -c "\
                        export TMPDIR='${MODE_TMPDIR}'; \
                        exec python '${SCRIPT_DIR}/run_benchmark.py' \
                            --manifest  '${RUN_MANIFEST}' \
                            --shapefile '${STOFS_SHAPEFILE}' \
                            --out-dir   '${OUT_DIR}' \
                            --nprocs    '${MPI_CPUS}' \
                            --hmin      '${HMIN}' \
                            --hmax      '${HMAX}' \
                            --modes     mpi_hybrid \
                            ${FLAGS}"

                else
                    srun --mpi=pmi2 \
                         --ntasks="${MPI_NTASKS}" \
                         --cpus-per-task="${MPI_CPUS}" \
             --overcommit \
                         bash -c "\
                        export TMPDIR='${MODE_TMPDIR}'; \
                        exec python '${SCRIPT_DIR}/run_benchmark.py' \
                            --manifest  '${RUN_MANIFEST}' \
                            --shapefile '${STOFS_SHAPEFILE}' \
                            --out-dir   '${OUT_DIR}' \
                            --nprocs    '${SERIAL_NPROCS}' \
                            --hmin      '${HMIN}' \
                            --hmax      '${HMAX}' \
                            --modes     ${MODE} \
                            ${FLAGS}"
                fi

            done  # MODE

            # Per-config per-variant report
            REPORT_DIRS=""
            for MODE in ${ORDERED_ANAS}; do
                REPORT_DIRS="${REPORT_DIRS} \
                    ${RESULTS_ROOT}/config_${CFG}/${VARIANT}/${MODE}"
            done
            echo ""
            echo "  --- Report: Config ${CFG} / ${VARIANT} ---"
            srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
                --results-dir ${REPORT_DIRS} \
                --out \
                "${RESULTS_ROOT}/config_${CFG}/report_${CFG}_${VARIANT}.txt"

        done  # VARIANT

    else
        # ── Original configs A-E ──────────────────────────────────────────
        # Standard single-node run with original ORIG_MODES.
        # MPI uses 80 ranks (1 mgr + 79 workers).

        for MODE in ${ORIG_MODES}; do
            OUT_DIR="${RESULTS_ROOT}/config_${CFG}/${MODE}"
            MODE_TMPDIR="${OUT_DIR}/tmp"
            mkdir -p "${OUT_DIR}" "${MODE_TMPDIR}"

            echo ""
            echo "--- Config ${CFG} / ${MODE} ---"

            if [ "${MODE}" = "mpi" ]; then
                srun --mpi=pmi2 \
                     --ntasks=80 --nodes=1 \
                     bash -c "\
                    export TMPDIR='${MODE_TMPDIR}'; \
                    exec python '${SCRIPT_DIR}/run_benchmark.py' \
                        --manifest  '${ORIG_MANIFEST}' \
                        --shapefile '${STOFS_SHAPEFILE}' \
                        --out-dir   '${OUT_DIR}' \
                        --nprocs    '${NPROCS}' \
                        --hmin      '${HMIN}' \
                        --hmax      '${HMAX}' \
                        --modes     mpi \
                        ${FLAGS}"
            else
                srun --mpi=pmi2 -n 1 bash -c "\
                    export TMPDIR='${MODE_TMPDIR}'; \
                    exec python '${SCRIPT_DIR}/run_benchmark.py' \
                        --manifest  '${ORIG_MANIFEST}' \
                        --shapefile '${STOFS_SHAPEFILE}' \
                        --out-dir   '${OUT_DIR}' \
                        --nprocs    '${NPROCS}' \
                        --hmin      '${HMIN}' \
                        --hmax      '${HMAX}' \
                        --modes     ${MODE} \
                        ${FLAGS}"
            fi

        done  # MODE

        # Per-config report
        REPORT_DIRS=""
        for MODE in ${ORIG_MODES}; do
            REPORT_DIRS="${REPORT_DIRS} ${RESULTS_ROOT}/config_${CFG}/${MODE}"
        done
        echo ""
        echo "--- Report: Config ${CFG} ---"
        srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
            --results-dir ${REPORT_DIRS} \
            --out \
            "${RESULTS_ROOT}/config_${CFG}/report_config_${CFG}.txt"

    fi  # is_anas_config

done  # CFG

# ── Final combined report ─────────────────────────────────────────────────────
echo ""
echo "--- Combined report (all configs) ---"
ALL_DIRS=""
for CFG in ${CONFIGS}; do
    if is_anas_config "${CFG}"; then
        for VARIANT in ${VARIANTS}; do
            for MODE in ${ANAS_MODES}; do
                D="${RESULTS_ROOT}/config_${CFG}/${VARIANT}/${MODE}"
                [ -d "${D}" ] && ALL_DIRS="${ALL_DIRS} ${D}"
            done
        done
    else
        for MODE in ${ORIG_MODES}; do
            D="${RESULTS_ROOT}/config_${CFG}/${MODE}"
            [ -d "${D}" ] && ALL_DIRS="${ALL_DIRS} ${D}"
        done
    fi
done

srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
    --results-dir ${ALL_DIRS} \
    --out         "${RESULTS_ROOT}/report_smoke_matrix_combined.txt"

echo ""
echo "================================================================="
echo " Smoke-matrix benchmark complete."
echo " Results root : ${RESULTS_ROOT}"
echo ""
echo " Per-config reports:"
for CFG in ${CONFIGS}; do
    if is_anas_config "${CFG}"; then
        for VARIANT in ${VARIANTS}; do
            echo "   Config ${CFG} / ${VARIANT}: " \
                "${RESULTS_ROOT}/config_${CFG}/report_${CFG}_${VARIANT}.txt"
        done
    else
        echo "   Config ${CFG}: " \
            "${RESULTS_ROOT}/config_${CFG}/report_config_${CFG}.txt"
    fi
done
echo ""
echo " Combined report:"
echo "   ${RESULTS_ROOT}/report_smoke_matrix_combined.txt"
echo ""
echo " Suggested first runs (fast validation):"
echo "   CONFIGS='F' ANAS_MODES='mpi_no_pool mpi_hybrid' \\"
echo "       VARIANTS='fat' sbatch slurm_smoke_matrix.sh"
echo "   CONFIGS='G' ANAS_MODES='mpi_no_pool mpi_hybrid' \\"
echo "       VARIANTS='fat' sbatch slurm_smoke_matrix.sh"
echo "================================================================="
