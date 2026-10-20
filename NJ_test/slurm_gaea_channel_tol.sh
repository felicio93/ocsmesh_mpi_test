#!/bin/bash
# =============================================================================
# Channel-detection cost: tolerance test.
#
# Usage: sbatch --job-name=tolA slurm_gaea_channel_tol.sh <mode> <tolerance|none> <label>
#   mode: hfun   -> build_nj_mesh_gaea.py --test 4 serial exact --hfun-only
#                   with --channel-tolerance (compare hfun stats with the
#                   tolerance=None run: min 500 / mean 1865.37 / max 3000 /
#                   5037 nodes)
#         tiles  -> time_channels_per_tile.py (per-tile get_channels time)
# Results: results_gaea_dbg/<label>/...   logs_gaea/chtol_<label>_<jobid>.out
# =============================================================================
#SBATCH --account=nos-surge
#SBATCH --partition=batch
#SBATCH --clusters=c6
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs_gaea/chtol_%x_%j.out
#SBATCH --error=logs_gaea/chtol_%x_%j.err

set -uo pipefail
MODE="${1:?hfun|tiles}"; TOL="${2:?tolerance or none}"; LABEL="${3:?label}"

module load PrgEnv-intel
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
source "/ncrc/proj/nos-surge/Felicio.Cassalho/miniconda3/etc/profile.d/conda.sh"
conda activate ocsmesh_mpi_test
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR}"
OUT="${SLURM_SUBMIT_DIR}/results_gaea_dbg/${LABEL}"
mkdir -p logs_gaea "${OUT}"
echo "=== $(date +%T) mode=${MODE} tolerance=${TOL} label=${LABEL}"
SECONDS=0
if [ "${MODE}" = hfun ]; then
  TOL_ARG=""; [ "${TOL}" != none ] && TOL_ARG="--channel-tolerance ${TOL}"
  python -u build_nj_mesh_gaea.py --test 4 --mode serial --method exact --hfun-only \
      ${TOL_ARG} --results-root "${OUT}"
  echo "=== exit=$?  elapsed=${SECONDS}s"
  python - <<PY
import json
s = json.load(open("${OUT}/test4_channels/serial/exact/summary.json"))
print("=== status", s.get("status"), "| hfun min/mean/max", s.get("hfun_min"), s.get("hfun_mean"), s.get("hfun_max"), "| hfun_nodes", s.get("hfun_nodes"), "| meshdata_s", s.get("meshdata_time_s"))
PY
else
  python -u time_channels_per_tile.py --tolerance "${TOL}" --out "${OUT}"
  echo "=== exit=$?  elapsed=${SECONDS}s"
fi
