#!/bin/bash
# =============================================================================
# SLURM job script — Smoke-test Config C (dedicated, 8h)
# =============================================================================
# Config C: "with constraints (no topofunc), no global features"
#   flags: --skip-topofunc --light-features
#
# Cost ladder position: B + constraint stage
#   Adds topo_bound_constraint (2 tiles), courant_num_constraint (2 tiles),
#   add_region_constraint (BOX1), add_patch (BOX2), and add_feature (BOX2
#   line) on top of Config A's fast refinements. Global contour/channel
#   are still skipped (--light-features).
#   topo_func_constraint is excluded because its unpicklable lambda forces
#   _apply_constraints to fall back to serial even in parallel/mpi mode.
#
# What this isolates:
#   Whether the OCSMesh fix that routes 'mpi' through _apply_constraints_parallel
#   actually speeds up the constraint stage. Config C vs Config B shows the
#   pure cost of constraints when they run through the Pool-parallel path.
#
# Expected walltime (15 tiles, 80 cores):
#   mpi       ~75 min  (constraint stage ~50 min on rank 0 + meshdata ~6 min)
#   parallel  ~4-5h    (constraint Pool + GEBCO bottleneck)
#   serial_mp ~4-5h    (constraints serial + GEBCO bottleneck)
#   total     ~6-7h — should fit within 8h with this dedicated job
#
# Submit:
#   sbatch slurm_smoke_config_C.sh
#   N_CUDEM=14 sbatch slurm_smoke_config_C.sh
#   MODES="mpi parallel" sbatch slurm_smoke_config_C.sh
# =============================================================================

#SBATCH --job-name=ocsmesh_smoke_C
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/smoke_C_%j.out
#SBATCH --error=logs/smoke_C_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ="/work2/noaa/nos-surge/felicioc/OCSMesh_MPI"
CONDA_ENV="ocsmesh_mpi_test"
STOFS_SHAPEFILE="${PROJ}/inputs/stofs3.shp"
DEM_OUT_DIR="${PROJ}/stofs_dems"
SCRIPT_DIR="${PROJ}/ocsmesh_mpi_test"
FULL_SMOKE_MANIFEST="${SCRIPT_DIR}/dem_manifest_smoke.json"
RESULTS_ROOT="${PROJ}/results/smoke_C_${SLURM_JOB_ID}"
NPROCS=79

# ── Config ────────────────────────────────────────────────────────────────────
CFG="C"
CFG_FLAGS="--skip-topofunc --light-features"
CFG_DESC="with constraints (no topofunc), no global contour/channel"
N_CUDEM="${N_CUDEM:-14}"
MODES="${MODES:-mpi parallel serial_mp}"

set -euo pipefail
mkdir -p "${RESULTS_ROOT}" logs

echo "================================================================="
echo " OCSMesh Smoke-Test — Config ${CFG}"
echo " ${CFG_DESC}"
echo " Job ID   : ${SLURM_JOB_ID}"
echo " Node     : ${SLURM_NODELIST}"
echo " Date     : $(date)"
echo " Modes    : ${MODES}"
echo " Flags    : ${CFG_FLAGS}"
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
    echo "--- Step 1: DEMs already downloaded ---"
fi

# ── Step 1b: Trim manifest ────────────────────────────────────────────────────
MANIFEST="${SCRIPT_DIR}/dem_manifest_smoke$((N_CUDEM + 1)).json"
if [ ! -f "${MANIFEST}" ]; then
    echo ""
    echo "--- Step 1b: Trimming manifest to $((N_CUDEM + 1)) tiles ---"
    srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/trim_manifest.py" \
        --in  "${FULL_SMOKE_MANIFEST}" \
        --out "${MANIFEST}" \
        --n-cudem "${N_CUDEM}"
else
    echo ""
    echo "--- Step 1b: Trimmed manifest found: ${MANIFEST} ---"
fi

# ── Step 2: Run all modes for Config C ────────────────────────────────────────
for MODE in ${MODES}; do
    OUT_DIR="${RESULTS_ROOT}/${MODE}"
    MODE_TMPDIR="${OUT_DIR}/tmp"
    mkdir -p "${OUT_DIR}" "${MODE_TMPDIR}"

    echo ""
    echo "--- Config ${CFG} / mode ${MODE} ---"

    if [ "${MODE}" = "mpi" ]; then
        srun --mpi=pmi2 --ntasks=80 --nodes=1 bash -c "\
            export TMPDIR='${MODE_TMPDIR}'; \
            exec python '${SCRIPT_DIR}/run_benchmark.py' \
                --manifest '${MANIFEST}' \
                --shapefile '${STOFS_SHAPEFILE}' \
                --out-dir '${OUT_DIR}' \
                --nprocs '${NPROCS}' \
                --modes mpi ${CFG_FLAGS}"
    else
        srun --mpi=pmi2 -n 1 bash -c "\
            export TMPDIR='${MODE_TMPDIR}'; \
            exec python '${SCRIPT_DIR}/run_benchmark.py' \
                --manifest '${MANIFEST}' \
                --shapefile '${STOFS_SHAPEFILE}' \
                --out-dir '${OUT_DIR}' \
                --nprocs '${NPROCS}' \
                --modes ${MODE} ${CFG_FLAGS}"
    fi
done

# ── Step 3: Report ────────────────────────────────────────────────────────────
REPORT_DIRS=""
for MODE in ${MODES}; do
    REPORT_DIRS="${REPORT_DIRS} ${RESULTS_ROOT}/${MODE}"
done
echo ""
echo "--- Generating report for Config ${CFG} ---"
srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
    --results-dir ${REPORT_DIRS} \
    --out         "${RESULTS_ROOT}/report_config_${CFG}.txt"

echo ""
echo "================================================================="
echo " Config ${CFG} complete."
echo " Results : ${RESULTS_ROOT}"
echo " Report  : ${RESULTS_ROOT}/report_config_${CFG}.txt"
echo "================================================================="
