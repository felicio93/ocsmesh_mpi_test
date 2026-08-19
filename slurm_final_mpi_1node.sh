#!/bin/bash
# =============================================================================
# FINAL benchmark — MPI mode, SINGLE node
# =============================================================================
# execution_mode='mpi': 80 ranks on one node (1 manager + 79 workers). The
# per-tile meshdata dispatch is distributed via MPIExecutor. _apply_features
# still runs on rank 0 only (known limitation, see collector.py TODO(mpi)).
#
# Uses the SAME manifest/recipe as all other final jobs (see final_config.sh).
#
# Submit: sbatch slurm_final_mpi_1node.sh
# =============================================================================

#SBATCH --job-name=ocsmesh_final_mpi_1node
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/final_mpi_1node_%j.out
#SBATCH --error=logs/final_mpi_1node_%j.err
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

RESULTS_DIR="${PROJ}/results/final_mpi_1node_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}" logs

load_ocsmesh_env
print_final_config

# MPI intermediate .npz files must live on shared FS (single node: /work2 fine).
export TMPDIR="${RESULTS_DIR}/mpi_tmp"
mkdir -p "${TMPDIR}"

echo ""
echo "--- FINAL MPI benchmark: 80 ranks on 1 node (nprocs=${NPROCS}) ---"
srun --mpi=pmi2 --ntasks=80 --nodes=1 python "${SCRIPT_DIR}/run_benchmark.py" \
    --manifest  "${MANIFEST}" \
    --shapefile "${STOFS_SHAPEFILE}" \
    --out-dir   "${RESULTS_DIR}" \
    --nprocs    "${NPROCS}" \
    --hmin      "${HMIN}" \
    --hmax      "${HMAX}" \
    --modes     mpi \
    ${ALL_FLAGS}

echo ""
echo "--- Generating report (mpi 1node) ---"
srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
    --results-dir "${RESULTS_DIR}" \
    --out         "${RESULTS_DIR}/report_mpi_1node.txt"

echo ""
echo "================================================================="
echo " mpi (1 node) complete. Results: ${RESULTS_DIR}"
echo "================================================================="
