#!/bin/bash
#SBATCH --job-name=chk_tiles
#SBATCH --account=nos-surge
#SBATCH --partition=batch
#SBATCH --clusters=c6
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs_gaea/check_tiles_%j.out
#SBATCH --error=logs_gaea/check_tiles_%j.err
set -uo pipefail
module load PrgEnv-intel
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
source "/ncrc/proj/nos-surge/Felicio.Cassalho/miniconda3/etc/profile.d/conda.sh"
conda activate ocsmesh_mpi_test
cd "${SLURM_SUBMIT_DIR}"; mkdir -p logs_gaea results_gaea
python -u check_tiles_clip.py
