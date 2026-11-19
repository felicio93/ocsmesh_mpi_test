#!/bin/bash
# =============================================================================
# Alaska large-domain mesh (0-360 lon, crosses the dateline)  |  NOAA Gaea (c6)
#
#   sbatch slurm_ak_mesh.sh                                   # m5, exact, bnd 1.5
#   sbatch --export=ALL,METHOD=fast slurm_ak_mesh.sh
#   sbatch --export=ALL,AK_OCSMESH=/path/to/OCSMesh,WORKAROUNDS=0 slurm_ak_mesh.sh
#
# AK_OCSMESH is the OCSMesh checkout put first on PYTHONPATH:
#   ak/lon360-patched  -> reference run (WORKAROUNDS=1, the default)
#   antimeridian branch -> comparison run (WORKAROUNDS=0)
# =============================================================================
#SBATCH --job-name=ak_mesh
#SBATCH --account=nos-surge
#SBATCH --partition=batch
#SBATCH --clusters=c6
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --output=logs/ak_mesh_%j.out
#SBATCH --error=logs/ak_mesh_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail

# ── Configuration (edit or override via --export) ────────────────────────────
ROOT=/gpfs/f6/nos-surge/scratch/Felicio.Cassalho/OCSMesh_MPI
AK_OCSMESH="${AK_OCSMESH:-${ROOT}/OCSMesh_ak_patched}"
CASE="${CASE:-m5_no_center}"
METHOD="${METHOD:-exact}"          # exact | fast
ENGINE="${ENGINE:-gmsh}"           # gmsh | triangle
BND_FACTOR="${BND_FACTOR:-1.5}"
WORKAROUNDS="${WORKAROUNDS:-1}"    # 1 = force EPSG:3338 (patched baseline)
NPROCS="${NPROCS:-${SLURM_CPUS_PER_TASK:-1}}"
DATA_DIR="${DATA_DIR:-${ROOT}/AK_LargeDomain_Mesh/2026_10_15}"
TAG="${CASE}_${METHOD}_bnd${BND_FACTOR}_w${WORKAROUNDS}_${SLURM_JOB_ID:-manual}"
OUT_DIR="${OUT_DIR:-${ROOT}/AK_LargeDomain_Mesh/results/${TAG}}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

# ── Environment ──────────────────────────────────────────────────────────────
module load PrgEnv-intel
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
source "/ncrc/proj/nos-surge/Felicio.Cassalho/miniconda3/etc/profile.d/conda.sh"
conda activate "ocsmesh_mpi_test"
export PYTHONPATH="${AK_OCSMESH}${PYTHONPATH:+:$PYTHONPATH}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

EXTRA=()
if [ "${WORKAROUNDS}" = "1" ]; then
    # read by ak/lon360-patched; exported so worker processes see it too
    export OCSMESH_METRIC_CRS=EPSG:3338
else
    unset OCSMESH_METRIC_CRS
    EXTRA+=(--no-workarounds)
fi

# Temporary files on the parallel filesystem (node-local /tmp is too small)
mkdir -p "${OUT_DIR}/tmp" "${SCRIPT_DIR}/logs"
export TMPDIR="${OUT_DIR}/tmp"
export OCSMESH_SHARED_TMPDIR="${OUT_DIR}/tmp"

cd "${SCRIPT_DIR}"

echo "============================================================="
echo "  Alaska 0-360 mesh   case=${CASE}  method=${METHOD}  engine=${ENGINE}"
echo "  bnd_factor=${BND_FACTOR}  workarounds=${WORKAROUNDS}  nprocs=${NPROCS}"
echo "  OCSMesh: ${AK_OCSMESH} ($(git -C "${AK_OCSMESH}" log -1 --format='%h %s'))"
echo "  Job    : ${SLURM_JOB_ID:-n/a}   Node: ${SLURM_NODELIST:-n/a}"
echo "  Data   : ${DATA_DIR}"
echo "  Output : ${OUT_DIR}"
echo "  Date   : $(date)"
echo "============================================================="

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --cpu-bind=cores \
    python build_ak_mesh.py \
        --data-dir   "${DATA_DIR}" \
        --out-dir    "${OUT_DIR}" \
        --save-shifted "${OUT_DIR}/shapes_360" \
        --case       "${CASE}" \
        --method     "${METHOD}" \
        --engine     "${ENGINE}" \
        --bnd-factor "${BND_FACTOR}" \
        --nprocs     "${NPROCS}" \
        ${EXTRA[@]+"${EXTRA[@]}"}

echo "=== DONE $(date) ==="
