#!/bin/bash
# slurm_r2_R_hybrid.sh
# mpi_hybrid: 452 ranks, each with 1 core (80 cores/node x 6 nodes = 480,
# but only 452 used). Auto core detection in run_benchmark.py will see
# 1 core per rank via SLURM_CPUS_PER_TASK and use nprocs=1.
# For true hybrid with multiple cores/rank, reduce ntasks and increase
# cpus-per-task, e.g. ntasks=57 cpus-per-task=8 (56 workers x 8 = 448 cores).
#SBATCH --job-name=r2_R_hybrid
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=6
#SBATCH --ntasks=57
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/r2_R_hybrid_%j.out
#SBATCH --error=logs/r2_R_hybrid_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

MANIFEST="${SCRIPT_DIR}/dem_manifest_full_split.json"
OUT="${PROJ}/results/config_R/mpi_hybrid"
mkdir -p "${OUT}/tmp" logs
export TMPDIR="${OUT}/tmp"
export OCSMESH_SHARED_TMPDIR="${OUT}/tmp"
load_ocsmesh_env

AVAIL=$(df /work2 | awk 'NR==2{print $4}')
if [ "${AVAIL}" -lt 10485760 ]; then
    echo "ERROR: less than 10 GB available on /work2. Aborting."
    exit 1
fi

echo "=== Config R mpi_hybrid (57 ranks x 8 cores, 6 nodes) ==="
echo "  56 worker ranks x 8 cores = 448 cores total"
echo "  451 tiles assigned round-robin across 56 workers"
echo "Job: ${SLURM_JOB_ID}  Nodes: ${SLURM_NODELIST}  Date: $(date)"
echo "Available /work2: $(df -h /work2 | awk 'NR==2{print $4}')"

export TMPDIR="${OUT}/tmp"
export OCSMESH_SHARED_TMPDIR="${OUT}/tmp"
srun --mpi=pmi2 \
     --ntasks=57 \
     --cpus-per-task=8 \
     --export=ALL,TMPDIR="${OUT}/tmp",OCSMESH_SHARED_TMPDIR="${OUT}/tmp" \
     python "${SCRIPT_DIR}/run_benchmark.py" \
        --manifest  "${MANIFEST}" \
        --shapefile "${STOFS_SHAPEFILE}" \
        --out-dir   "${OUT}" \
        --nprocs    8 \
        --hmin      "${HMIN}" \
        --hmax      "${HMAX}" \
        --modes     mpi_hybrid \
        --config-r

echo "=== DONE $(date) ==="
