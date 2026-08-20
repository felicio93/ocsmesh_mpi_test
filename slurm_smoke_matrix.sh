#!/bin/bash
# =============================================================================
# SLURM job script — smoke-test MATRIX benchmark
# =============================================================================
# Runs a 4-config "cost ladder" where each step adds exactly one cost class,
# so the contribution of each pipeline stage can be isolated. Every config is
# run across THREE execution modes (mpi / parallel / serial_mp), giving 12
# runs total. Results for each config land in a separate sub-directory so the
# per-stage cProfile breakdowns stay comparable.
#
# The ladder (each row adds one cost class to the row above):
#
#   Config   flow+const   constraints(no topofunc)   contour/channel   boxes
#   -------  -----------  ------------------------   ---------------   -----
#   A         1 per tile        -                          -            -
#   B         2 per tile        -                          -            -
#   C         2 per tile        yes                        -            -
#   D         2 per tile        yes                        yes          yes
#
#   A  "no constraint, 1 refinement per tile"  (modulo scheme, one class/tile)
#        flags: --skip-constraints --skip-box-refinements --light-features
#   B  "no constraint, 2 refinements per tile" (flow+const on EVERY tile)
#        flags: --all-fast-refinements
#   C  "with constraints (no topofunc)"        (topo_bound + courant + boxes)
#        flags: --skip-topofunc
#   D  "full recipe minus topofunc"            (adds global contour/channel)
#        flags: --skip-topofunc  (LIGHT_FEATURES off → contour/channel run)
#
# topo_func_constraint is EXCLUDED from every config because its unpicklable
# lambda forces OCSMesh's _apply_constraints to fall back to SERIAL even in
# parallel/mpi mode — which would defeat the point of comparing modes.
#
# WHY this matters after the OCSMesh branch changes:
#   The collector now routes 'mpi' through the same Pool-based parallel path
#   as 'parallel' for _apply_constraints / _apply_flow_limiters /
#   _apply_const_val. Config C vs B isolates whether that change actually
#   sped up the constraint stage; Config D vs C isolates the still-serial
#   _apply_features (contour/channel/box) stage, the next MPI target.
#
# Walltime guide (single node, exclusive, 15 tiles):
#   A ~ short (1 fast ref/tile + meshdata)
#   B ~ short (2 fast refs/tile + meshdata)
#   C ~ moderate (constraints via parallel path + meshdata)
#   D ~ longest (global contour/channel is O(tiles × segments), serial rank-0)
#   Budget 8h; if D overruns, submit D separately or drop it from CONFIGS.
#
# Submit (all four configs):
#   sbatch slurm_smoke_matrix.sh
# Submit a subset:
#   CONFIGS="A B" sbatch slurm_smoke_matrix.sh
#   MODES="mpi parallel" sbatch slurm_smoke_matrix.sh
# =============================================================================

#SBATCH --job-name=ocsmesh_smoke_matrix
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80          # 1 manager + 79 MPI workers
#SBATCH --cpus-per-task=1
#SBATCH --exclusive                   # whole node (512 GB)
#SBATCH --time=08:00:00
#SBATCH --output=logs/smoke_matrix_%j.out
#SBATCH --error=logs/smoke_matrix_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

# ── Paths (edit only if your layout differs) ─────────────────────────────────
PROJ="/work2/noaa/nos-surge/felicioc/OCSMesh_MPI"
CONDA_ENV="ocsmesh_mpi_test"
STOFS_SHAPEFILE="${PROJ}/inputs/stofs3.shp"
DEM_OUT_DIR="${PROJ}/stofs_dems"
SCRIPT_DIR="${PROJ}/ocsmesh_mpi_test"
FULL_SMOKE_MANIFEST="${SCRIPT_DIR}/dem_manifest_smoke.json"
RESULTS_ROOT="${PROJ}/results/smoke_matrix_${SLURM_JOB_ID}"
NPROCS=79   # = ntasks_per_node - 1 (1 MPI manager + 79 workers)

# ── Matrix selection (override at submit time) ───────────────────────────────
# CONFIGS: which cost-ladder rows to run.  MODES: which execution modes.
CONFIGS="${CONFIGS:-A B C D}"
MODES="${MODES:-mpi parallel serial_mp}"

# Number of CUDEM tiles for the smoke matrix (+1 GEBCO = N+1 total).
N_CUDEM="${N_CUDEM:-14}"

set -euo pipefail
mkdir -p "${RESULTS_ROOT}" logs

echo "================================================================="
echo " OCSMesh Smoke-Matrix Benchmark"
echo " Job ID   : ${SLURM_JOB_ID}"
echo " Node     : ${SLURM_NODELIST}"
echo " Date     : $(date)"
echo " Configs  : ${CONFIGS}"
echo " Modes    : ${MODES}"
echo " Results  : ${RESULTS_ROOT}"
echo "================================================================="

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load intel-oneapi-compilers/2022.2.1
module load intel-oneapi-mpi/2021.7.1
module load hdf5/1.12.2
module load netcdf-c/4.9.0
module load netcdf-fortran/4.6.0
source "/work2/noaa/nos-surge/felicioc/envs/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

srun --mpi=pmi2 -n 1 python -c "from mpi4py import MPI; print('mpi4py MPI version:', MPI.Get_version())"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# TMPDIR on /work2 (Lustre). Node-local /tmp overflows when 79 workers each
# write full clipped-raster .tif files. See HERCULES_NOTES.md.
export TMPDIR="${RESULTS_ROOT}/tmp"
mkdir -p "${TMPDIR}"

# ── Step 1: Download DEMs (if needed) ─────────────────────────────────────────
if [ ! -f "${FULL_SMOKE_MANIFEST}" ]; then
    echo ""
    echo "--- Step 1: Downloading DEMs ---"
    python "${SCRIPT_DIR}/download_dems.py" \
        --out-dir "${DEM_OUT_DIR}" \
        --manifest "${FULL_SMOKE_MANIFEST}" \
        --only MA_NH_ME
else
    echo ""
    echo "--- Step 1: DEMs already downloaded (full smoke manifest found) ---"
fi

# ── Step 1b: Trim manifest to N_CUDEM tiles (+1 GEBCO) ───────────────────────
MANIFEST="${SCRIPT_DIR}/dem_manifest_smoke$((N_CUDEM + 1)).json"
if [ ! -f "${MANIFEST}" ]; then
    echo ""
    echo "--- Step 1b: Trimming smoke manifest to $((N_CUDEM + 1)) tiles ---"
    srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/trim_manifest.py" \
        --in  "${FULL_SMOKE_MANIFEST}" \
        --out "${MANIFEST}" \
        --n-cudem "${N_CUDEM}"
else
    echo ""
    echo "--- Step 1b: Trimmed manifest found: ${MANIFEST} ---"
fi

# ── Map a config letter to its run_benchmark.py flags ────────────────────────
# A: no constraint, 1 refinement per tile (modulo, one class/tile)
# B: no constraint, 2 refinements per tile (flow+const on every tile)
# C: with constraints (topo_bound + courant + boxes), NO topofunc
# D: C + global contour/channel (full recipe minus topofunc)
config_flags() {
    case "$1" in
        A) echo "--skip-constraints --skip-box-refinements --light-features" ;;
        B) echo "--all-fast-refinements" ;;
        C) echo "--skip-topofunc --light-features" ;;
        D) echo "--skip-topofunc" ;;
        *) echo "__INVALID__" ;;
    esac
}

config_desc() {
    case "$1" in
        A) echo "no constraint, 1 refinement/tile" ;;
        B) echo "no constraint, 2 refinements/tile" ;;
        C) echo "with constraints (no topofunc), no global features" ;;
        D) echo "full recipe minus topofunc (adds contour/channel + boxes)" ;;
        *) echo "INVALID" ;;
    esac
}

# ── Step 2: Run the matrix ───────────────────────────────────────────────────
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

    for MODE in ${MODES}; do
        OUT_DIR="${RESULTS_ROOT}/config_${CFG}/${MODE}"
        MODE_TMPDIR="${OUT_DIR}/tmp"
        mkdir -p "${OUT_DIR}" "${MODE_TMPDIR}"

        echo ""
        echo "--- Config ${CFG} / mode ${MODE} ---"

        if [ "${MODE}" = "mpi" ]; then
            # MPI: launch 80 ranks (1 manager + 79 workers).
            srun --mpi=pmi2 --ntasks=80 --nodes=1 bash -c "\
                export TMPDIR='${MODE_TMPDIR}'; \
                exec python '${SCRIPT_DIR}/run_benchmark.py' \
                    --manifest '${MANIFEST}' \
                    --shapefile '${STOFS_SHAPEFILE}' \
                    --out-dir '${OUT_DIR}' \
                    --nprocs '${NPROCS}' \
                    --modes mpi ${FLAGS}"
        else
            # parallel / serial_mp: single rank, Pool workers.
            srun --mpi=pmi2 -n 1 bash -c "\
                export TMPDIR='${MODE_TMPDIR}'; \
                exec python '${SCRIPT_DIR}/run_benchmark.py' \
                    --manifest '${MANIFEST}' \
                    --shapefile '${STOFS_SHAPEFILE}' \
                    --out-dir '${OUT_DIR}' \
                    --nprocs '${NPROCS}' \
                    --modes ${MODE} ${FLAGS}"
        fi
    done

    # ── Per-config profiling report across the modes that ran ────────────
    REPORT_DIRS=""
    for MODE in ${MODES}; do
        REPORT_DIRS="${REPORT_DIRS} ${RESULTS_ROOT}/config_${CFG}/${MODE}"
    done
    echo ""
    echo "--- Report for config ${CFG} ---"
    srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
        --results-dir ${REPORT_DIRS} \
        --out         "${RESULTS_ROOT}/config_${CFG}/report_config_${CFG}.txt"
done

echo ""
echo "================================================================="
echo " Smoke-matrix benchmark complete."
echo " Results : ${RESULTS_ROOT}"
echo " Per-config reports: ${RESULTS_ROOT}/config_<A|B|C|D>/report_config_<...>.txt"
echo "================================================================="
