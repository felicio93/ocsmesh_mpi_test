#!/bin/bash
# =============================================================================
# Fast reproduction of the uniform-50 m hfun (NJ test 4) on a tile subset,
# running the REAL build_nj_mesh_gaea.py code path.
#
# Usage: sbatch --job-name=rA --time=02:00:00 slurm_gaea_repro_t4.sh \
#            <label> <tiles-spec> [--hfun-only] [--no-shared-tmp]
#   tiles-spec e.g. gebco:1,third:2,ninth:2   (AOI tiles only)
#   --hfun-only      skip Geom + MeshDriver (tests whether Geom-first matters)
#   --no-shared-tmp  do not set TMPDIR/OCSMESH_SHARED_TMPDIR to the results dir
# Look at:  results_gaea_dbg/<label>/test4_channels/serial/exact/summary.json
#           -> hfun_min / hfun_mean / hfun_max / hfun_nodes
# =============================================================================
#SBATCH --account=nos-surge
#SBATCH --partition=batch
#SBATCH --clusters=c6
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs_gaea/repro_%x_%j.out
#SBATCH --error=logs_gaea/repro_%x_%j.err

set -uo pipefail
LABEL="${1:?label}"; SPEC="${2:?tiles spec}"; shift 2
HFUN_ONLY=""; SHARED_TMP=1
for a in "$@"; do
  [ "$a" = "--hfun-only" ] && HFUN_ONLY="--hfun-only"
  [ "$a" = "--no-shared-tmp" ] && SHARED_TMP=0
done

module load PrgEnv-intel
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
source "/ncrc/proj/nos-surge/Felicio.Cassalho/miniconda3/etc/profile.d/conda.sh"
conda activate ocsmesh_mpi_test
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR}"
OUT="${SLURM_SUBMIT_DIR}/results_gaea_dbg/${LABEL}"
mkdir -p logs_gaea "${OUT}/tmp"
if [ "${SHARED_TMP}" = 1 ]; then
  export TMPDIR="${OUT}/tmp"; export OCSMESH_SHARED_TMPDIR="${OUT}/tmp"
fi

echo "=== $(date +%T) label=${LABEL} spec=${SPEC} hfun_only='${HFUN_ONLY}' shared_tmp=${SHARED_TMP}"
echo "=== OCSMesh: $(python -c 'import ocsmesh;print(ocsmesh.__file__)')"
python make_subset_manifest.py "${SPEC}" "${OUT}/manifest_subset.json" || exit 1

SECONDS=0
python -u build_nj_mesh_gaea.py --test 4 --mode serial --method exact ${HFUN_ONLY} \
    --manifest "${OUT}/manifest_subset.json" --results-root "${OUT}"
echo "=== exit=$?  elapsed=${SECONDS}s"
python - <<PY
import json
s = json.load(open("${OUT}/test4_channels/serial/exact/summary.json"))
print("=== status", s.get("status"), "| hfun min/mean/max", s.get("hfun_min"), s.get("hfun_mean"), s.get("hfun_max"), "| hfun_nodes", s.get("hfun_nodes"))
PY
