#!/bin/bash
# =============================================================================
# SLURM job script — single-node smoke test benchmark
# =============================================================================
# Validates the full pipeline end-to-end across three modes on ONE Hercules
# node. Modes run in order of importance so the most critical result arrives
# first:
#
#   Step 2 — MPI    (80 ranks; validates the MPI path in ~30 min)
#   Step 3 — parallel (79 workers; Pool path in ~15 min)
#   Step 4 — serial_mp (1 manager; slowest, runs last, ~3.5h with skip-constraints)
#
# Recipe flags (default: all ON for smoke):
#   LIGHT_FEATURES=1       skip add_contour / add_channel (O(tiles x segments))
#   SKIP_TOPOFUNC=1        skip add_topo_func_constraint (serialises constraints)
#   SKIP_CONSTRAINTS=1     skip ALL topo/courant constraints (~3h/tile serial)
#
# With all three flags, serial_mp reduces to flow_limiter + const_value
# (fast) + Gmsh meshing (~25 min/tile), fitting in 8h for 7 tiles.
#
# Walltime guide (single node, exclusive):
#   smoke (7 tiles, flags ON) : ~4.5h  [mpi~30m, parallel~15m, serial_mp~3.5h]
#   full recipe (7 tiles)     : exceeds 8h (constraints dominate; use Profile A)
#
# Submit:
#   LIGHT_FEATURES=1 SKIP_TOPOFUNC=1 SKIP_CONSTRAINTS=1 sbatch slurm_single_node.sh
# =============================================================================

#SBATCH --job-name=ocsmesh_bench_1node
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80          # 1 manager + 79 MPI workers
#SBATCH --cpus-per-task=1
#SBATCH --exclusive                   # whole node (512 GB)
#SBATCH --time=08:00:00
#SBATCH --output=logs/bench_1node_%j.out
#SBATCH --error=logs/bench_1node_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

# ── Paths (edit only if your layout differs) ─────────────────────────────────
PROJ="/work2/noaa/nos-surge/felicioc/OCSMesh_MPI"
CONDA_ENV="ocsmesh_mpi_test"
STOFS_SHAPEFILE="${PROJ}/inputs/stofs3.shp"
DEM_OUT_DIR="${PROJ}/stofs_dems"
SCRIPT_DIR="${PROJ}/ocsmesh_mpi_test"
FULL_SMOKE_MANIFEST="${SCRIPT_DIR}/dem_manifest_smoke.json"
# MANIFEST is set in Step 1b below (trimmed 7-tile smoke manifest).
RESULTS_DIR="${PROJ}/results/single_node_${SLURM_JOB_ID}"
NPROCS=79   # = ntasks_per_node - 1 (1 MPI manager + 79 workers)

# ── Recipe flags ─────────────────────────────────────────────────────────────
# Default all ON for the smoke test (removes slow constraint stages).
# Override at submit time, e.g.: LIGHT_FEATURES=0 sbatch slurm_single_node.sh
LIGHT_FEATURES="${LIGHT_FEATURES:-1}"
LIGHT_FLAG=""
if [ "${LIGHT_FEATURES}" = "1" ]; then
    LIGHT_FLAG="--light-features"
    echo "NOTE: LIGHT_FEATURES=1 -> contour/channel SKIPPED"
fi

SKIP_TOPOFUNC="${SKIP_TOPOFUNC:-1}"
SKIP_TOPOFUNC_FLAG=""
if [ "${SKIP_TOPOFUNC}" = "1" ]; then
    SKIP_TOPOFUNC_FLAG="--skip-topofunc"
    echo "NOTE: SKIP_TOPOFUNC=1 -> topo_func_constraint SKIPPED"
fi

# SKIP_CONSTRAINTS skips ALL topo/courant constraints (~3h/tile serial).
# Default ON for smoke test; turns OFF for profiling.
SKIP_CONSTRAINTS="${SKIP_CONSTRAINTS:-1}"
SKIP_CONSTRAINTS_FLAG=""
if [ "${SKIP_CONSTRAINTS}" = "1" ]; then
    SKIP_CONSTRAINTS_FLAG="--skip-constraints"
    echo "NOTE: SKIP_CONSTRAINTS=1 -> topo_bound/topo_func/courant SKIPPED"
fi

ALL_FLAGS="${LIGHT_FLAG} ${SKIP_TOPOFUNC_FLAG} ${SKIP_CONSTRAINTS_FLAG}"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p "${RESULTS_DIR}" logs

echo "================================================================="
echo " OCSMesh Single-Node Benchmark"
echo " Job ID   : ${SLURM_JOB_ID}"
echo " Node     : ${SLURM_NODELIST}"
echo " Date     : $(date)"
echo " Results  : ${RESULTS_DIR}"
echo "================================================================="

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load intel-oneapi-compilers/2022.2.1
module load intel-oneapi-mpi/2021.7.1
module load hdf5/1.12.2
module load netcdf-c/4.9.0
module load netcdf-fortran/4.6.0
source "/work2/noaa/nos-surge/felicioc/envs/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

srun --mpi=pmi2 -n 1 python -c "from mpi4py import MPI; print('mpi4py MPI version:', MPI.Get_version())"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# TMPDIR on /work2 (Lustre). Node-local /tmp overflows when 79 workers each
# write full clipped-raster .tif files. See HERCULES_NOTES.md.
export TMPDIR="${RESULTS_DIR}/tmp"
mkdir -p "${TMPDIR}"

# ── Step 1: Download DEMs ─────────────────────────────────────────────────────
if [ ! -f "${FULL_SMOKE_MANIFEST}" ]; then
    echo ""
    echo "--- Step 1: Downloading DEMs ---"
    python "${SCRIPT_DIR}/download_dems.py" \
        --out-dir "${DEM_OUT_DIR}" \
        --manifest "${FULL_SMOKE_MANIFEST}" \
        --only MA_NH_ME
else
    echo ""
    echo "--- Step 1: DEMs already downloaded (full smoke manifest found) ---"
fi

# ── Step 1b: Trim manifest to 7 tiles (6 CUDEM = 1 per class + 1 GEBCO) ──────
# At ~25 min/tile Gmsh + ~3h/tile constraints, 7 tiles with SKIP_CONSTRAINTS=1
# gives serial_mp ~3.5h total — fits in the 8h window.
MANIFEST="${SCRIPT_DIR}/dem_manifest_smoke7.json"
if [ ! -f "${MANIFEST}" ]; then
    echo ""
    echo "--- Step 1b: Trimming smoke manifest to 7 tiles (1 per class) ---"
    srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/trim_manifest.py" \
        --in  "${FULL_SMOKE_MANIFEST}" \
        --out "${MANIFEST}" \
        --n-cudem 6
else
    echo ""
    echo "--- Step 1b: Trimmed smoke manifest found ---"
fi

# ── Step 2: MPI benchmark FIRST (80 ranks) ───────────────────────────────────
# Run MPI first so the most important validation completes in ~30 min.
# If MPI crashes, cancel and debug without waiting hours for serial_mp.
echo ""
echo "--- Step 2: MPI benchmark (80 ranks) --- [RUNS FIRST]"
MPI_TMPDIR="${RESULTS_DIR}/mpi_tmp"
mkdir -p "${MPI_TMPDIR}"
export TMPDIR="${MPI_TMPDIR}"

srun --mpi=pmi2 --ntasks=80 --nodes=1 bash -c "\
    export TMPDIR='${MPI_TMPDIR}'; \
    exec python '${SCRIPT_DIR}/run_benchmark.py' \
        --manifest '${MANIFEST}' \
        --shapefile '${STOFS_SHAPEFILE}' \
        --out-dir '${RESULTS_DIR}/mpi' \
        --nprocs '${NPROCS}' \
        --modes mpi ${ALL_FLAGS}"

# Reset TMPDIR back to the job scratch for the serial/parallel steps.
export TMPDIR="${RESULTS_DIR}/tmp"
mkdir -p "${TMPDIR}"

# ── Step 3: parallel benchmark (single rank, Pool workers) ───────────────────
echo ""
echo "--- Step 3: parallel benchmark (nprocs=${NPROCS}) ---"
srun --mpi=pmi2 -n 1 bash -c "\
    export TMPDIR='${RESULTS_DIR}/tmp'; \
    exec python '${SCRIPT_DIR}/run_benchmark.py' \
        --manifest '${MANIFEST}' \
        --shapefile '${STOFS_SHAPEFILE}' \
        --out-dir '${RESULTS_DIR}/parallel' \
        --nprocs '${NPROCS}' \
        --modes parallel ${ALL_FLAGS}"

# ── Step 4: serial_mp benchmark (slowest — runs last) ────────────────────────
echo ""
echo "--- Step 4: serial_mp benchmark (baseline, slowest) ---"
srun --mpi=pmi2 -n 1 bash -c "\
    export TMPDIR='${RESULTS_DIR}/tmp'; \
    exec python '${SCRIPT_DIR}/run_benchmark.py' \
        --manifest '${MANIFEST}' \
        --shapefile '${STOFS_SHAPEFILE}' \
        --out-dir '${RESULTS_DIR}/serial_mp' \
        --nprocs '${NPROCS}' \
        --modes serial_mp ${ALL_FLAGS}"

# ── Step 5: Profiling report ──────────────────────────────────────────────────
echo ""
echo "--- Step 5: Generating profiling report ---"
srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
    --results-dir  "${RESULTS_DIR}/mpi" \
                   "${RESULTS_DIR}/parallel" \
                   "${RESULTS_DIR}/serial_mp" \
    --out          "${RESULTS_DIR}/benchmark_report.txt"

echo ""
echo "================================================================="
echo " Benchmark complete."
echo " Results : ${RESULTS_DIR}"
echo " Report  : ${RESULTS_DIR}/benchmark_report.txt"
echo "================================================================="
