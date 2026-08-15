#!/bin/bash
# =============================================================================
# FINAL benchmark — parallel mode (multiprocessing)
# =============================================================================
# execution_mode='parallel': all per-tile steps AND the meshdata dispatch use
# a multiprocessing.Pool with NPROCS workers on ONE node.
#
# Uses the SAME manifest/recipe as all other final jobs (see final_config.sh),
# so its output mesh must match serial_mp (numerical-equivalence check).
#
# Submit: sbatch slurm_final_parallel.sh
# =============================================================================

#SBATCH --job-name=ocsmesh_final_parallel
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/final_parallel_%j.out
#SBATCH --error=logs/final_parallel_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

RESULTS_DIR="${PROJ}/results/final_parallel_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}" logs

load_ocsmesh_env
print_final_config

export TMPDIR="/tmp/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

echo ""
echo "--- FINAL parallel benchmark (nprocs=${NPROCS}) ---"
srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/run_benchmark.py" \
    --manifest  "${MANIFEST}" \
    --shapefile "${STOFS_SHAPEFILE}" \
    --out-dir   "${RESULTS_DIR}" \
    --nprocs    "${NPROCS}" \
    --hmin      "${HMIN}" \
    --hmax      "${HMAX}" \
    --modes     parallel \
    ${LIGHT_FLAG}

echo ""
echo "--- Generating report (parallel) ---"
srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
    --results-dir "${RESULTS_DIR}" \
    --out         "${RESULTS_DIR}/report_parallel.txt"

echo ""
echo "================================================================="
echo " parallel complete. Results: ${RESULTS_DIR}"
echo "================================================================="
