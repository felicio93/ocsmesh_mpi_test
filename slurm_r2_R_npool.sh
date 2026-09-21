#!/bin/bash
# slurm_r2_R_npool.sh
#SBATCH --job-name=r2_R_npool
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=6
#SBATCH --ntasks=452
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/r2_R_npool_%j.out
#SBATCH --error=logs/r2_R_npool_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

MANIFEST="${SCRIPT_DIR}/dem_manifest_full_split.json"
OUT="${PROJ}/results/config_R/mpi_no_pool"
mkdir -p "${OUT}/tmp" logs
load_ocsmesh_env

AVAIL=$(df /work2 | awk 'NR==2{print $4}')
if [ "${AVAIL}" -lt 10485760 ]; then
    echo "ERROR: less than 10 GB available on /work2. Aborting."
    exit 1
fi

echo "=== Config R mpi_no_pool (452 ranks, 6 nodes, 451 tiles) ==="
echo "Job: ${SLURM_JOB_ID}  Nodes: ${SLURM_NODELIST}  Date: $(date)"
echo "Available /work2: $(df -h /work2 | awk 'NR==2{print $4}')"

# Export TMPDIR to shared Lustre so all ranks on all nodes use it
export TMPDIR="${OUT}/tmp"
srun --mpi=pmi2 \
     --ntasks=452 \
     --cpus-per-task=1 \
     --export=ALL,TMPDIR="${OUT}/tmp" \
     python "${SCRIPT_DIR}/run_benchmark.py" \
        --manifest  "${MANIFEST}" \
        --shapefile "${STOFS_SHAPEFILE}" \
        --out-dir   "${OUT}" \
        --nprocs    1 \
        --hmin      "${HMIN}" \
        --hmax      "${HMAX}" \
        --modes     mpi_no_pool \
        --config-r

echo "=== DONE $(date) ==="
