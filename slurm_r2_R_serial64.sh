#!/bin/bash
# slurm_r2_R_serial64.sh
#SBATCH --job-name=r2_R_s64
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/r2_R_serial64_%j.out
#SBATCH --error=logs/r2_R_serial64_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

MANIFEST="${PROFILE_C_FAT_MANIFEST}"
OUT="${PROJ}/results/config_R/serial_mp_64"
mkdir -p "${OUT}/tmp" logs
load_ocsmesh_env

# Pre-run disk space check
AVAIL=$(df /work2 | awk 'NR==2{print $4}')
if [ "${AVAIL}" -lt 10485760 ]; then   # 10 GB in KB
    echo "ERROR: less than 10 GB available on /work2. Aborting."
    exit 1
fi

echo "=== Config R serial_mp nprocs=64 (fat baseline) ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURM_NODELIST}  Date: $(date)"
echo "Available /work2: $(df -h /work2 | awk 'NR==2{print $4}')"

# Force TMPDIR to shared Lustre before srun
export TMPDIR="${OUT}/tmp"
srun --mpi=pmi2 \
     --ntasks=1 \
     --export=ALL,TMPDIR="${OUT}/tmp" \
     python "${SCRIPT_DIR}/run_benchmark.py" \
    --manifest  "${MANIFEST}" \
    --shapefile "${STOFS_SHAPEFILE}" \
    --out-dir   "${OUT}" \
    --nprocs    64 \
    --hmin      "${HMIN}" \
    --hmax      "${HMAX}" \
    --modes     serial_mp \
    --config-r

echo "=== DONE $(date) ==="
