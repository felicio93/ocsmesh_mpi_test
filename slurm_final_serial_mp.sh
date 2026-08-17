#!/bin/bash
# =============================================================================
# FINAL benchmark — serial_mp mode (OCSMesh 'serial' execution_mode)
# =============================================================================
# The baseline. execution_mode='serial': per-tile meshdata dispatch runs
# one-tile-at-a-time; only the Pool-based feature steps use workers.
# This is the SLOWEST mode and therefore the binding constraint on how large
# the shared workload (FINAL_MANIFEST) can be while still fitting in 8h.
#
# Uses the SAME manifest/recipe as all other final jobs (see final_config.sh).
#
# Submit: sbatch slurm_final_serial_mp.sh
# =============================================================================

#SBATCH --job-name=ocsmesh_final_serial_mp
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/final_serial_mp_%j.out
#SBATCH --error=logs/final_serial_mp_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

RESULTS_DIR="${PROJ}/results/final_serial_mp_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}" logs

load_ocsmesh_env
print_final_config

# Temp files on /work2 (Lustre) — node-local /tmp fills up under heavy raster
# clipping and aborts with 'No space left on device'.
export TMPDIR="${RESULTS_DIR}/tmp"
mkdir -p "${TMPDIR}"

echo ""
echo "--- FINAL serial_mp benchmark ---"
srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/run_benchmark.py" \
    --manifest  "${MANIFEST}" \
    --shapefile "${STOFS_SHAPEFILE}" \
    --out-dir   "${RESULTS_DIR}" \
    --nprocs    "${NPROCS}" \
    --hmin      "${HMIN}" \
    --hmax      "${HMAX}" \
    --modes     serial_mp \
    ${LIGHT_FLAG}

echo ""
echo "--- Generating report (serial_mp) ---"
srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
    --results-dir "${RESULTS_DIR}" \
    --out         "${RESULTS_DIR}/report_serial_mp.txt"

echo ""
echo "================================================================="
echo " serial_mp complete. Results: ${RESULTS_DIR}"
echo "================================================================="
