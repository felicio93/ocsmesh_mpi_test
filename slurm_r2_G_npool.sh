#!/bin/bash
#SBATCH --job-name=r2_G_npool
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/r2_G_npool_%j.out
#SBATCH --error=logs/r2_G_npool_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

MANIFEST="${PROFILE_C_FAT_MANIFEST}"
OUT="${PROJ}/results/round2/config_G/mpi_no_pool"
mkdir -p "${OUT}/tmp" logs
load_ocsmesh_env

echo "=== Config G mpi_no_pool (9 ranks, 1 core/rank) ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURM_NODELIST}  Date: $(date)"

export TMPDIR="${OUT}/tmp"
srun --mpi=pmi2 \
     --ntasks=9 \
     --cpus-per-task=1 \
     bash -c "
         export TMPDIR='${OUT}/tmp'
         exec python '${SCRIPT_DIR}/run_benchmark.py' \
             --manifest  '${MANIFEST}' \
             --shapefile '${STOFS_SHAPEFILE}' \
             --out-dir   '${OUT}' \
             --nprocs    1 \
             --hmin      '${HMIN}' \
             --hmax      '${HMAX}' \
             --modes     mpi_no_pool \
             --config-g"

echo "=== DONE $(date) ==="
