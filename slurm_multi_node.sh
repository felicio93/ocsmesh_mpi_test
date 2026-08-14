#!/bin/bash
# =============================================================================
# SLURM job script — multi-node benchmark
# =============================================================================
# Tests MPI parallelization across multiple Hercules nodes.
# Specifically validates:
#   - Shared filesystem access (Lustre/GPFS) across nodes
#   - MPI point-to-point correctness across inter-node connections
#   - Scalability: speedup as a function of total MPI ranks
#
# Configuration: 4 nodes × 80 cores = 320 ranks total
#   Rank 0  = manager (dispatches tasks)
#   Ranks 1-319 = workers (execute meshdata tasks)
#
# The serial and parallel baselines were already measured in the single-node
# run; this script runs only the MPI mode for the multi-node speedup study.
#
# Submit: sbatch slurm_multi_node.sh
# =============================================================================

#SBATCH --job-name=ocsmesh_bench_multinode
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=80
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=logs/bench_multinode_%j.out
#SBATCH --error=logs/bench_multinode_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

# ── Paths (edit only if your layout differs) ─────────────────────────────────
PROJ="/work2/noaa/nos-surge/felicioc/OCSMesh_MPI"
CONDA_ENV="ocsmesh_mpi_test"
STOFS_SHAPEFILE="${PROJ}/inputs/stofs3.shp"
MANIFEST="${PROJ}/stofs_dems/dem_manifest.json"
# IMPORTANT: RESULTS_DIR MUST be on a shared filesystem visible to ALL nodes
# (Lustre/GPFS) — /work2 qualifies; node-local /tmp does NOT.
RESULTS_DIR="${PROJ}/results/multi_node_${SLURM_JOB_ID}"
SCRIPT_DIR="${PROJ}/ocsmesh_mpi_test"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

TOTAL_RANKS=$((SLURM_NNODES * 80))
NWORKERS=$((TOTAL_RANKS - 1))

mkdir -p "${RESULTS_DIR}" logs

echo "================================================================="
echo " OCSMesh Multi-Node MPI Benchmark"
echo " Job ID   : ${SLURM_JOB_ID}"
echo " Nodes    : ${SLURM_NODELIST}"
echo " Total ranks: ${TOTAL_RANKS}  (1 manager + ${NWORKERS} workers)"
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

source "/work2/noaa/nos-surge/felicioc/envs/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Must use srun --mpi=pmi2 even for single-rank calls: inside a SLURM allocation
# 'import ocsmesh' triggers MPI_Init, which aborts without a PMI server. See HERCULES_NOTES #7b.
srun --mpi=pmi2 -n 1 python -c "from mpi4py import MPI; print('mpi4py MPI version:', MPI.Get_version())"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── CRITICAL: Use shared filesystem for all temp/intermediate files ───────────
# On multi-node HPC jobs, each node has a node-local /tmp that is NOT shared.
# OCSMesh writes intermediate .npz files that workers on OTHER nodes must read.
# Set TMPDIR to your Lustre/GPFS scratch space.
SHARED_SCRATCH="${RESULTS_DIR}/mpi_tmp"
mkdir -p "${SHARED_SCRATCH}"
export TMPDIR="${SHARED_SCRATCH}"

echo ""
echo "--- Shared filesystem path : ${SHARED_SCRATCH} ---"
echo "    (MPIExecutor.verify_shared_filesystem() will test this at runtime)"

# ── Step 1: Verify manifest exists ───────────────────────────────────────────
if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Manifest not found: ${MANIFEST}"
    echo "  Run slurm_single_node.sh first (it downloads DEMs and creates the manifest)."
    exit 1
fi

# ── Step 2: MPI benchmark across all nodes ───────────────────────────────────
echo ""
echo "--- MPI benchmark: ${TOTAL_RANKS} ranks across ${SLURM_NNODES} nodes ---"

# --mpi=pmi2 is required on Hercules (confirmed working; the wrong PMI or a
# bare launch aborts with 'PMI2_Job_GetId returned 14'). See HERCULES_NOTES #3/#4.
srun --mpi=pmi2 \
     --ntasks=${TOTAL_RANKS} \
     --nodes=${SLURM_NNODES} \
     --ntasks-per-node=80 \
     --distribution=cyclic \
     python "${SCRIPT_DIR}/run_benchmark.py" \
        --manifest   "${MANIFEST}" \
        --shapefile  "${STOFS_SHAPEFILE}" \
        --out-dir    "${RESULTS_DIR}" \
        --nprocs     "${NWORKERS}" \
        --modes      mpi

# ── Step 3: Report ────────────────────────────────────────────────────────────
echo ""
echo "--- Generating profiling report ---"
srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
    --results-dir  "${RESULTS_DIR}" \
    --out          "${RESULTS_DIR}/benchmark_report_multinode.txt"

echo ""
echo "================================================================="
echo " Multi-node benchmark complete."
echo " Results : ${RESULTS_DIR}"
echo "================================================================="
