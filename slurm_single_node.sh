#!/bin/bash
# =============================================================================
# SLURM job script — single-node benchmark
# =============================================================================
# Runs four benchmark modes on ONE Hercules node (80 cores, 512 GB):
#
#   serial_true  — true single-core baseline (nprocs=1 forced)
#   serial_mp    — OCSMesh serial mode, Pool steps use 79 workers
#   parallel     — full multiprocessing mode, 79 workers
#   mpi          — MPI mode, 1 manager + 79 worker ranks
#
# serial_true/serial_mp/parallel run as srun -n 1 (single rank).
# mpi runs as srun -n 80 (1 manager + 79 workers).
#
# Walltime guide (single node, exclusive):
#   smoke run (~39 tiles, MA/NH/ME subset): 4–8 h
#   full run  (~388 tiles, full domain):    use windfall partition, 24–48 h
#
# Submit: sbatch slurm_single_node.sh
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
CONDA_ENV="ocsmesh_mpi_test"                      # conda env with OCSMesh + mpi4py
STOFS_SHAPEFILE="${PROJ}/inputs/stofs3.shp"
DEM_OUT_DIR="${PROJ}/stofs_dems"
SCRIPT_DIR="${PROJ}/ocsmesh_mpi_test"
# Smoke manifests:
#   dem_manifest_smoke.json    - 39 tiles (MA/NH/ME). serial_mp exceeds 8h.
#   dem_manifest_smoke15.json  - 14 CUDEM + 1 GEBCO. serial_mp fits in ~6h.
# For the full 388-tile production benchmark, switch to dem_manifest.json.
FULL_SMOKE_MANIFEST="${SCRIPT_DIR}/dem_manifest_smoke.json"
# MANIFEST is set in Step 1b below (trimmed 7-tile smoke manifest).
RESULTS_DIR="${PROJ}/results/single_node_${SLURM_JOB_ID}"

# Set LIGHT_FEATURES=1 to skip global add_contour/add_channel (fast MPI-path
# debugging). Leave unset/0 for the full realistic recipe.
LIGHT_FEATURES="${LIGHT_FEATURES:-0}"
LIGHT_FLAG=""
if [ "${LIGHT_FEATURES}" = "1" ]; then
    LIGHT_FLAG="--light-features"
    echo "NOTE: LIGHT_FEATURES=1 -> global contour/channel refinements SKIPPED"
fi

# Set SKIP_TOPOFUNC=1 to skip add_topo_func_constraint. That constraint forces
# OCSMesh's _apply_constraints to run SERIALLY (~44 min/tile) even in
# parallel/mpi modes. Skipping it removes the dominant serial bottleneck and
# lets the constraint stage actually parallelize. Recommended for the smoke
# test so serial_mp + parallel + mpi all fit in 8h. Default ON here.
SKIP_TOPOFUNC="${SKIP_TOPOFUNC:-1}"
SKIP_TOPOFUNC_FLAG=""
if [ "${SKIP_TOPOFUNC}" = "1" ]; then
    SKIP_TOPOFUNC_FLAG="--skip-topofunc"
    echo "NOTE: SKIP_TOPOFUNC=1 -> topo_func_constraint SKIPPED"
fi
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

# Activate conda
source "/work2/noaa/nos-surge/felicioc/envs/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Verify mpi4py sees the right MPI
# Must use srun --mpi=pmi2 even for single-rank calls: inside a SLURM allocation
# 'import ocsmesh' triggers MPI_Init, which aborts without a PMI server. See HERCULES_NOTES #7b.
srun --mpi=pmi2 -n 1 python -c "from mpi4py import MPI; print('mpi4py MPI version:', MPI.Get_version())"

# Thread pinning (belt-and-suspenders; ocsmesh also sets these in MPI mode)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Temp files (OCSMesh HfunCollector._work_dir, clipped rasters, .npz) go here.
# MUST be on /work2 (Lustre, 6.3PB) NOT node-local /tmp: with 79 workers each
# writing full clipped-raster .tif files, a compute node's local disk fills up
# and OCSMesh aborts with 'No space left on device' (see job 9575590 MPI step).
export TMPDIR="${RESULTS_DIR}/tmp"
mkdir -p "${TMPDIR}"

# ── Step 1: Download DEMs (only if the full smoke manifest is missing) ────────
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

# ── Step 1b: Trim smoke manifest to 7 tiles (1 per refinement class) ──────────
# The topo_bound/topo_func/courant constraints run SERIALLY at ~44 min/tile
# (this is the dominant cost, measured on job 9585234). To fit serial_mp in
# the 8h wall clock while still exercising EVERY refinement type exactly once,
# cut to 6 CUDEM tiles (modulo-6 -> 1 tile per class) + 1 GEBCO = 7 tiles.
#   estimated serial_mp: ~3 constraint tiles x 44 min + fast steps ~= 3h
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

# ── Step 2: serial_true + serial_mp + parallel benchmarks (single rank) ───────
echo ""
echo "--- Step 2: serial_true / serial_mp / parallel benchmarks ---"
NPROCS=79   # = ntasks_per_node - 1 (leave one for the MPI manager)

srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/run_benchmark.py" \
    --manifest "${MANIFEST}" \
    --shapefile "${STOFS_SHAPEFILE}" \
    --out-dir   "${RESULTS_DIR}/serial_parallel" \
    --nprocs    "${NPROCS}" \
    --modes     serial_mp parallel \
    ${LIGHT_FLAG} ${SKIP_TOPOFUNC_FLAG}

# ── Step 3: MPI benchmark (all 80 ranks) ─────────────────────────────────────
echo ""
echo "--- Step 3: MPI benchmark (80 ranks) ---"

# Shared filesystem for MPI intermediate files
MPI_TMPDIR="${RESULTS_DIR}/mpi_tmp"
mkdir -p "${MPI_TMPDIR}"
export TMPDIR="${MPI_TMPDIR}"

# --mpi=pmi2 is required on Hercules (confirmed working; a bare launch or
# the wrong PMI aborts with 'PMI2_Job_GetId returned 14'). See HERCULES_NOTES #3/#4.
srun --mpi=pmi2 --ntasks=80 --nodes=1 python "${SCRIPT_DIR}/run_benchmark.py" \
    --manifest "${MANIFEST}" \
    --shapefile "${STOFS_SHAPEFILE}" \
    --out-dir   "${RESULTS_DIR}/mpi" \
    --nprocs    "${NPROCS}" \
    --modes     mpi \
    ${LIGHT_FLAG} ${SKIP_TOPOFUNC_FLAG}

# ── Step 4: Profiling report ──────────────────────────────────────────────────
echo ""
echo "--- Step 4: Generating profiling reports ---"
srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
    --results-dir  "${RESULTS_DIR}" \
    --out          "${RESULTS_DIR}/benchmark_report.txt"

echo ""
echo "================================================================="
echo " Benchmark complete."
echo " Results : ${RESULTS_DIR}"
echo " Report  : ${RESULTS_DIR}/benchmark_report.txt"
echo "================================================================="
