#!/bin/bash
# =============================================================================
# Channel-refinement diagnosis (NJ tests 4/5 give a uniform 50 m hfun).
# One job = one diagnose_channels.py run, so several can run side by side.
#
# Usage:  sbatch --job-name=diagA --time=01:00:00 slurm_gaea_diagnose_channels.sh \
#             <tiles> <tolerance|none> <label>
# e.g.    sbatch ... slurm_gaea_diagnose_channels.sh gebco:1,third:3,ninth:3 50 A
# Follow: tail -f logs_gaea/diagnose_<label>_<jobid>.out
# =============================================================================
#SBATCH --account=nos-surge
#SBATCH --partition=batch
#SBATCH --clusters=c6
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs_gaea/diagnose_%x_%j.out
#SBATCH --error=logs_gaea/diagnose_%x_%j.err

set -uo pipefail
TILES="${1:?tiles, e.g. gebco:1,third:3,ninth:3}"
TOL="${2:-none}"
LABEL="${3:-run}"

module load PrgEnv-intel
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
source "/ncrc/proj/nos-surge/Felicio.Cassalho/miniconda3/etc/profile.d/conda.sh"
conda activate ocsmesh_mpi_test
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs_gaea results_gaea/diagnose

TOL_ARG=""
[ "${TOL}" != "none" ] && TOL_ARG="--tolerance ${TOL}"

echo "=== $(date +%T) label=${LABEL} tiles=${TILES} tolerance=${TOL}"
SECONDS=0
python -u diagnose_channels.py --tiles "${TILES}" --domain ${TOL_ARG}
echo "=== exit=$?  elapsed=${SECONDS}s"
