#!/bin/bash
#SBATCH --job-name=cpu_affinity
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20          # 20 MPI ranks for CPU affinity test
#SBATCH --cpus-per-task=3 
#SBATCH --time=00:01:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Paths (edit only if your layout differs) ─────────────────────────────────
PROJ=${PROJ:-"/work2/noaa/nos-surge/felicioc/OCSMesh_MPI"}
CONDA_ENV="ocsmesh_mpi_test" #"3.12"
NPROCS=4   # = ntasks_per_node - 1 (1 MPI manager + 79 workers)

set -euo pipefail
mkdir -p logs


# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load intel-oneapi-compilers/2022.2.1
module load intel-oneapi-mpi/2021.7.1
module load hdf5/1.12.2
module load netcdf-c/4.9.0
module load netcdf-fortran/4.6.0
# module load contrib
# module load rdhpcs-conda
source "/work2/noaa/nos-surge/felicioc/envs/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# source $REPO/.venv/bin/activate


srun --mpi=pmi2 -n 1 python ${SCRIPT_DIR}/cpu_affinity.py
#srun --mpi=pmi2 --cpu-bind=none -n 1 python ${SCRIPT_DIR}/cpu_affinity.py
