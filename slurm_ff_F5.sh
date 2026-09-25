#!/bin/bash
# slurm_ff_F5.sh — add_channel only (size=500m)
#SBATCH --job-name=ff_F5
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --ntasks=45
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/ff_F5_%j.out
#SBATCH --error=logs/ff_F5_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

MANIFEST="${SCRIPT_DIR}/dem_manifest_config_f.json"
OUT="${PROJ}/results/config_ffat/isolation/F5_channel"
mkdir -p "${OUT}/tmp" logs
export TMPDIR="${OUT}/tmp"
export OCSMESH_SHARED_TMPDIR="${OUT}/tmp"
load_ocsmesh_env

AVAIL=$(df /work2 | awk 'NR==2{print $4}')
if [ "${AVAIL}" -lt 10485760 ]; then
    echo "ERROR: less than 10 GB on /work2. Aborting."; exit 1
fi

# F5 gets full 8h — add_channel at size=500m is most expensive op
echo "=== Config F5: add_channel only (size=500m, 45 ranks) ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURM_NODELIST}  Date: $(date)"

srun --mpi=pmi2 \
     --ntasks=45 --cpus-per-task=1 \
     --export=ALL,TMPDIR="${OUT}/tmp",OCSMESH_SHARED_TMPDIR="${OUT}/tmp" \
     python "${SCRIPT_DIR}/run_benchmark.py" \
        --manifest  "${MANIFEST}" \
        --shapefile "${STOFS_SHAPEFILE}" \
        --out-dir   "${OUT}" \
        --nprocs    1 \
        --hmin      "${HMIN}" --hmax "${HMAX}" \
        --modes     mpi_no_pool --config-f5

echo "=== DONE $(date) ==="
