#!/bin/bash
#SBATCH --job-name=r2_F_s64
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/r2_F_serial64_%j.out
#SBATCH --error=logs/r2_F_serial64_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

MANIFEST="${PROFILE_C_FAT_MANIFEST}"
OUT="${PROJ}/results/round2/config_F/serial_mp_64"
mkdir -p "${OUT}/tmp" logs
load_ocsmesh_env

echo "=== Config F serial_mp nprocs=64 ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURM_NODELIST}  Date: $(date)"

export TMPDIR="${OUT}/tmp"
srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/run_benchmark.py" \
    --manifest  "${MANIFEST}" \
    --shapefile "${STOFS_SHAPEFILE}" \
    --out-dir   "${OUT}" \
    --nprocs    64 \
    --hmin      "${HMIN}" \
    --hmax      "${HMAX}" \
    --modes     serial_mp \
    --config-f

echo "=== DONE $(date) ==="
