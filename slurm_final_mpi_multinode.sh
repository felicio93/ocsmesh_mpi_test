#!/bin/bash
# =============================================================================
# FINAL benchmark — MPI mode, MULTI node
# =============================================================================
# execution_mode='mpi' across NODES nodes. Tests inter-node MPI correctness,
# shared-filesystem visibility, and scaling of the meshdata dispatch as total
# rank count grows.
#
# Uses the SAME manifest/recipe as all other final jobs (see final_config.sh),
# so the resulting mesh must still match serial_mp / parallel / mpi-1node.
#
# Adjust --nodes below for the scaling study (2, 4, 8, ...). Total ranks =
# nodes * 80; workers = total - 1.
#
# Submit: sbatch slurm_final_mpi_multinode.sh
# =============================================================================

#SBATCH --job-name=ocsmesh_final_mpi_multinode
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=80
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/final_mpi_multinode_%j.out
#SBATCH --error=logs/final_mpi_multinode_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

# Profile B: skip-constraints + light-features to isolate meshdata stage.
# 18 CUDEM tiles — serial_mp ~7.5h (Gmsh only), parallel/mpi in minutes.
LIGHT_FEATURES=1
SKIP_CONSTRAINTS=1
SKIP_TOPOFUNC=1
ALL_FLAGS="--light-features --skip-constraints --skip-topofunc"
MANIFEST="${PROFILE_B_MANIFEST}"

RESULTS_DIR="${PROJ}/results/final_mpi_multinode_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}" logs

load_ocsmesh_env

TOTAL_RANKS=$((SLURM_NNODES * 80))
NWORKERS=$((TOTAL_RANKS - 1))

print_final_config
echo "   Total ranks   : ${TOTAL_RANKS}  (1 manager + ${NWORKERS} workers)"
echo "   Nodes         : ${SLURM_NNODES}"

# CRITICAL: multi-node intermediate .npz files must be on a SHARED filesystem
# (Lustre/GPFS). /work2 qualifies; node-local /tmp does NOT (workers on other
# nodes cannot see rank 0's files). MPIExecutor.verify_shared_filesystem()
# checks this at runtime.
export TMPDIR="${RESULTS_DIR}/mpi_tmp"
mkdir -p "${TMPDIR}"

echo ""
echo "--- FINAL MPI benchmark: ${TOTAL_RANKS} ranks across ${SLURM_NNODES} nodes ---"
srun --mpi=pmi2 \
     --ntasks=${TOTAL_RANKS} \
     --nodes=${SLURM_NNODES} \
     --ntasks-per-node=80 \
     --distribution=cyclic \
     python "${SCRIPT_DIR}/run_benchmark.py" \
        --manifest  "${MANIFEST}" \
        --shapefile "${STOFS_SHAPEFILE}" \
        --out-dir   "${RESULTS_DIR}" \
        --nprocs    "${NWORKERS}" \
        --hmin      "${HMIN}" \
        --hmax      "${HMAX}" \
        --modes     mpi \
        ${ALL_FLAGS}

echo ""
echo "--- Generating report (mpi multinode) ---"
srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
    --results-dir "${RESULTS_DIR}" \
    --out         "${RESULTS_DIR}/report_mpi_multinode.txt"

echo ""
echo "================================================================="
echo " mpi (${SLURM_NNODES} nodes) complete. Results: ${RESULTS_DIR}"
echo "================================================================="
