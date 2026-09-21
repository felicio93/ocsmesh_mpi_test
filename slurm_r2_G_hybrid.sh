#!/bin/bash
#SBATCH --job-name=r2_G_hybrid
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/r2_G_hybrid_%j.out
#SBATCH --error=logs/r2_G_hybrid_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

MANIFEST="${PROFILE_C_FAT_MANIFEST}"
OUT="${PROJ}/results/round2/config_G/mpi_hybrid"
mkdir -p "${OUT}/tmp" logs
load_ocsmesh_env

echo "=== Config G mpi_hybrid (9 ranks, 8 cores/rank) ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURM_NODELIST}  Date: $(date)"

export TMPDIR="${OUT}/tmp"
srun --mpi=pmi2 \
     --ntasks=9 \
     --cpus-per-task=8 \
     --overcommit \
     bash -c "
         export TMPDIR='${OUT}/tmp'
         exec python '${SCRIPT_DIR}/run_benchmark.py' \
             --manifest  '${MANIFEST}' \
             --shapefile '${STOFS_SHAPEFILE}' \
             --out-dir   '${OUT}' \
             --nprocs    8 \
             --hmin      '${HMIN}' \
             --hmax      '${HMAX}' \
             --modes     mpi_hybrid \
             --config-g"

echo "=== DONE $(date) ==="
