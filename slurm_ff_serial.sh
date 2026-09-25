#!/bin/bash
#SBATCH --job-name=ff_serial
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/ff_serial_%j.out
#SBATCH --error=logs/ff_serial_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

MANIFEST="${SCRIPT_DIR}/dem_manifest_config_f.json"
OUT="${PROJ}/results/config_ffat/serial_mp"
mkdir -p "${OUT}/tmp" logs
export TMPDIR="${OUT}/tmp"
export OCSMESH_SHARED_TMPDIR="${OUT}/tmp"
load_ocsmesh_env

AVAIL=$(df /work2 | awk 'NR==2{print $4}')
if [ "${AVAIL}" -lt 10485760 ]; then
    echo "ERROR: less than 10 GB on /work2. Aborting."; exit 1
fi

echo "=== Config F-fat serial_mp (44 tiles, nprocs=44) ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURM_NODELIST}  Date: $(date)"

srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/run_benchmark.py" \
    --manifest  "${MANIFEST}" \
    --shapefile "${STOFS_SHAPEFILE}" \
    --out-dir   "${OUT}" \
    --nprocs    44 \
    --hmin      "${HMIN}" \
    --hmax      "${HMAX}" \
    --modes     serial_mp \
    --config-ffat

echo "=== DONE $(date) ==="
