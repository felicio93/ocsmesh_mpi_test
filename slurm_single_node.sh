#!/bin/bash
# =============================================================================
# SLURM job script — single-node benchmark
# =============================================================================
# Tests serial, parallel (multiprocessing), and MPI on ONE Hercules node.
# This is the first thing to run: validates the full pipeline before scaling.
#
# Hercules node specs: 80 cores/node, 512 GB/node.
# We request the whole node (--exclusive), use 1 rank for Rank 0 (manager)
# + 79 worker ranks for MPI, and nprocs=79 for the multiprocessing baseline.
#
# Submit: sbatch slurm_single_node.sh
# =============================================================================

#SBATCH --job-name=ocsmesh_bench_1node
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80          # 1 manager + 79 MPI workers
#SBATCH --cpus-per-task=1
#SBATCH --exclusive                   # whole node (512 GB)
#SBATCH --time=08:00:00
#SBATCH --output=logs/bench_1node_%j.out
#SBATCH --error=logs/bench_1node_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

# ── Paths (edit only if your layout differs) ─────────────────────────────────
PROJ="/work2/noaa/nos-surge/felicioc/OCSMesh_MPI"
CONDA_ENV="ocsmesh_mpi_test"                      # conda env with OCSMesh + mpi4py
STOFS_SHAPEFILE="${PROJ}/inputs/stofs3.shp"
DEM_OUT_DIR="${PROJ}/stofs_dems"
MANIFEST="${DEM_OUT_DIR}/dem_manifest.json"
RESULTS_DIR="${PROJ}/results/single_node_${SLURM_JOB_ID}"
SCRIPT_DIR="${PROJ}/ocsmesh_mpi_test"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

mkdir -p "${RESULTS_DIR}" logs

echo "================================================================="
echo " OCSMesh Single-Node Benchmark"
echo " Job ID   : ${SLURM_JOB_ID}"
echo " Node     : ${SLURM_NODELIST}"
echo " Date     : $(date)"
echo " Results  : ${RESULTS_DIR}"
echo "================================================================="

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load intel-oneapi-compilers/2022.2.1
module load intel-oneapi-mpi/2021.7.1
module load hdf5/1.12.2
module load netcdf-c/4.9.0
module load netcdf-fortran/4.6.0

# Activate conda
source "/work2/noaa/nos-surge/felicioc/envs/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Verify mpi4py sees the right MPI
# Must use srun --mpi=pmi2 even for single-rank calls: inside a SLURM allocation
# 'import ocsmesh' triggers MPI_Init, which aborts without a PMI server. See HERCULES_NOTES #7b.
srun --mpi=pmi2 -n 1 python -c "from mpi4py import MPI; print('mpi4py MPI version:', MPI.Get_version())"

# Thread pinning (belt-and-suspenders; ocsmesh also sets these in MPI mode)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Use the SLURM allocation's scratch for temp files (shared FS on single node)
export TMPDIR="/tmp/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

# ── Step 1: Download DEMs (only if manifest is missing) ───────────────────────
if [ ! -f "${MANIFEST}" ]; then
    echo ""
    echo "--- Step 1: Downloading DEMs ---"
    python "${SCRIPT_DIR}/download_dems.py" \
        --out-dir "${DEM_OUT_DIR}" \
        --manifest "${MANIFEST}"
else
    echo ""
    echo "--- Step 1: DEMs already downloaded (manifest found) ---"
fi

# ── Step 2: Serial + Parallel benchmarks (single process) ────────────────────
echo ""
echo "--- Step 2: Serial and Parallel benchmarks ---"
NPROCS=79   # = ntasks_per_node - 1 (leave one for the MPI manager)

srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/run_benchmark.py" \
    --manifest "${MANIFEST}" \
    --shapefile "${STOFS_SHAPEFILE}" \
    --out-dir   "${RESULTS_DIR}/serial_parallel" \
    --nprocs    "${NPROCS}" \
    --modes     serial parallel

# ── Step 3: MPI benchmark (all 80 ranks) ─────────────────────────────────────
echo ""
echo "--- Step 3: MPI benchmark (80 ranks) ---"

# Shared filesystem for MPI intermediate files
MPI_TMPDIR="${RESULTS_DIR}/mpi_tmp"
mkdir -p "${MPI_TMPDIR}"
export TMPDIR="${MPI_TMPDIR}"

# --mpi=pmi2 is required on Hercules (confirmed working; a bare launch or
# the wrong PMI aborts with 'PMI2_Job_GetId returned 14'). See HERCULES_NOTES #3/#4.
srun --mpi=pmi2 --ntasks=80 --nodes=1 python "${SCRIPT_DIR}/run_benchmark.py" \
    --manifest "${MANIFEST}" \
    --shapefile "${STOFS_SHAPEFILE}" \
    --out-dir   "${RESULTS_DIR}/mpi" \
    --nprocs    "${NPROCS}" \
    --modes     mpi

# ── Step 4: Profiling report ──────────────────────────────────────────────────
echo ""
echo "--- Step 4: Generating profiling reports ---"
srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
    --results-dir  "${RESULTS_DIR}" \
    --out          "${RESULTS_DIR}/benchmark_report.txt"

echo ""
echo "================================================================="
echo " Benchmark complete."
echo " Results : ${RESULTS_DIR}"
echo " Report  : ${RESULTS_DIR}/benchmark_report.txt"
echo "================================================================="
