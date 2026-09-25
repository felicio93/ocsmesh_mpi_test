#!/bin/bash
# slurm_ff_full_hybrid.sh — full Config F-fat, mpi_hybrid
#SBATCH --job-name=ff_full_hybrid
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/ff_full_hybrid_%j.out
#SBATCH --error=logs/ff_full_hybrid_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

MANIFEST="${SCRIPT_DIR}/dem_manifest_config_f.json"
OUT="${PROJ}/results/config_ffat/full/mpi_hybrid"
mkdir -p "${OUT}/tmp" logs
export TMPDIR="${OUT}/tmp"
export OCSMESH_SHARED_TMPDIR="${OUT}/tmp"
load_ocsmesh_env

AVAIL=$(df /work2 | awk 'NR==2{print $4}')
if [ "${AVAIL}" -lt 10485760 ]; then
    echo "ERROR: less than 10 GB on /work2. Aborting."; exit 1
fi

echo "=== Config F-fat full mpi_hybrid (9 ranks x 8 cores, 44 tiles) ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURM_NODELIST}  Date: $(date)"

srun --mpi=pmi2 \
     --ntasks=9 --cpus-per-task=8 \
     --export=ALL,TMPDIR="${OUT}/tmp",OCSMESH_SHARED_TMPDIR="${OUT}/tmp" \
     python "${SCRIPT_DIR}/run_benchmark.py" \
        --manifest  "${MANIFEST}" \
        --shapefile "${STOFS_SHAPEFILE}" \
        --out-dir   "${OUT}" \
        --nprocs    8 \
        --hmin      "${HMIN}" --hmax "${HMAX}" \
        --modes     mpi_hybrid --config-ffat

echo "=== DONE $(date) ==="
