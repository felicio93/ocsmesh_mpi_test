# OCSMesh MPI Benchmark — STOFS-3D-Atlantic

End-to-end test of the new OCSMesh MPI parallelization against the serial
and multiprocessing modes, using real-world DEMs for the STOFS-3D-Atlantic
domain on the NOAA RDHPC (Hercules).

All paths in this README and in the SLURM scripts are hard-wired to:

```
/work2/noaa/nos-surge/felicioc/OCSMesh_MPI
```

If you use a different location, edit the `PROJ=` line below and the
`PROJ=` line at the top of each SLURM script.

---

## Files

```
ocsmesh_mpi_test/
├── download_dems.py         # Step 3 – download GEBCO + CUDEM 1/9" tiles
├── build_geom_and_hfun.py   # Geom/Hfun recipe (raster order + refinements)
├── run_benchmark.py         # Step 5 – run serial/parallel/MPI + profile
├── analyze_profile.py       # Step 7 – build the human-readable report
├── slurm_single_node.sh     # SLURM: single node, all 3 modes
├── slurm_multi_node.sh      # SLURM: multi-node MPI scaling
├── HERCULES_NOTES.md        # running log of HPC gotchas + OCSMesh feedback
└── README.md
```

`build_geom_and_hfun.py` is the module you edit to change *what* refinement
goes *where*. `run_benchmark.py` just imports it.

---

## What is being tested

**Global mesh size bounds:** `hmin = 1000 m` (1 km, finest),
`hmax = 7000 m` (7 km, coarsest / background).

**DEMs:** 1 GEBCO tile (full domain, 15", lowest priority) + ~375 CUDEM
1/9" tiles (every other tile per subfolder). CUDEM tiles always override
GEBCO where they overlap. The gaps left by "every other tile" let you see
GEBCO show through, so the priority mechanism is visually verifiable.

**Per-source refinements — assigned by list index (modulo 6, CUDEM only):**

| `(cudem_pos) % 6` | Refinement |
|---|---|
| 0 | `add_subtidal_flow_limiter` (hmin=1000, hmax=7000, −200→0 m) |
| 1 | `add_constant_value` (1000 m, −5→0 m) |
| 2 | `add_topo_bound_constraint` (1500 m, −2→1 m, min) |
| 3 | `add_topo_func_constraint` (depth/2, −3000→0 m, min) |
| 4 | `add_courant_num_constraint` (0.9, dt=150 s, amp=2 m) |
| 5 | *skipped* (leaves some tiles refinement-free) |

**Global refinements (all rasters):** `add_contour` (0 m + −200 m),
`add_channel`.

**Box-based refinements (fixed lat/lon):**
- Box1 `(-85..-82, 25..31)` → `add_region_constraint` (max 3500 m)
- Box2 `(-80..-77, 31..35)` → `add_patch` (1000 m) + `add_feature` (line)

Every `HfunCollector` refinement method is exercised at least once.

---

## Step 1 — Set up shell environment (once per session)

```bash
export PROJ=/work2/noaa/nos-surge/felicioc/OCSMesh_MPI
export REPO=$PROJ/ocsmesh_mpi_test
export DEMS=$PROJ/stofs_dems
export SHP=$PROJ/inputs/stofs3.shp
export MANIFEST=$REPO/dem_manifest.json

mkdir -p $PROJ
```

---

## Step 2 — Clone and install OCSMesh (editable) + clone this repo

Editable install (`-e`) means any later `git pull` in the OCSMesh repo is
picked up immediately with no reinstall.

```bash
# Load the toolchain FIRST so mpi4py builds against the right MPI.
module purge
module load intel-oneapi-compilers/2022.2.1
module load intel-oneapi-mpi/2021.7.1
module load hdf5/1.12.2
module load netcdf-c/4.9.0
module load netcdf-fortran/4.6.0

# Create and activate the conda env (named ocsmesh_mpi_test).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n ocsmesh_mpi_test python=3.10
conda activate ocsmesh_mpi_test

# --- Clone OCSMesh into $PROJ and check out the dev (MPI) branch ---
# The MPI implementation lives on the 'dev' branch. A default clone
# (main) has NO ocsmesh/mpi.py and imports fail with
# "ModuleNotFoundError: No module named 'ocsmesh.mpi'". See HERCULES_NOTES #7.
cd $PROJ
git clone https://github.com/noaa-ocs-modeling/OCSMesh.git
cd OCSMesh
git checkout dev                     # <-- REQUIRED: MPI code is on dev
git pull
git submodule update --init --recursive   # (if any submodules)

# --- Editable install of ocsmesh (core deps only) ---
# NOTE: do NOT rely on the ".[mpi]" extra on HPC. It pulls a PREBUILT
# mpi4py wheel that is linked against a generic/bundled MPI, which then
# fails under srun with Intel MPI. See HERCULES_NOTES.md #1 and #2.
pip install -e .

# --- Build mpi4py FROM SOURCE against the loaded Intel MPI module ---
# MPICC points the build at the site MPI compiler wrapper; --no-binary
# forces a source build (no cached/prebuilt wheel).
which mpicc                                  # sanity: must resolve to Intel MPI
MPICC=$(which mpicc) pip install --no-binary=mpi4py --no-cache-dir mpi4py

# Verify ocsmesh + the MPI module import:
python -c "import ocsmesh; print('ocsmesh at', ocsmesh.__file__)"
python -c "from ocsmesh.mpi import MPIExecutor; print('ocsmesh.mpi OK')"

# Verify mpi4py — MUST be run under srun, NOT as a bare 'python -c'.
# A bare run aborts with 'PMI2_Job_GetId returned 14' because there is no
# PMI server outside a launcher. See HERCULES_NOTES.md #3.
srun --mpi=pmi2 -n 2 python -c \
  "from mpi4py import MPI; c=MPI.COMM_WORLD; print('rank', c.Get_rank(), 'of', c.Get_size())"

# --- Clone this benchmark repo into $PROJ ---
cd $PROJ
git clone https://github.com/felicio93/ocsmesh_mpi_test.git
```

> The editable install means `$PROJ/OCSMesh` is your live source tree. To
> update: `cd $PROJ/OCSMesh && git pull` — your Python env uses the new
> code immediately, no reinstall needed.
>
> If `srun --mpi=pmi2` errors, list the PMI types your Slurm supports with
> `srun --mpi=list` and use the matching one (e.g. `pmix`). See
> `HERCULES_NOTES.md` for the full troubleshooting log.

---

## Step 3 — Download the DEMs

Run on a login node (network access required). Re-runnable: existing files
are skipped.

**GEBCO note:** the NCEI ETOPO2022 THREDDS server has no WCS, so the
deep-ocean tile is NOT auto-downloaded (see HERCULES_NOTES.md #5). Provide
it yourself: download a GEBCO grid for the domain from
https://download.gebco.net/ (GeoTIFF) and either drop it in `$DEMS/gebco/`
or point `GEBCO_LOCAL` at it. The download script auto-detects a `*.tif`
in `$DEMS/gebco/`.

```bash
cd $REPO

# Deep-ocean background: point at your GEBCO GeoTIFF (or just place it in
# $DEMS/gebco/ and skip this — the script auto-detects it there).
export GEBCO_LOCAL=$DEMS/gebco/gebco_2024_n56.0_s5.0_w-100.0_e-50.0.tif

# Dry run first — prints how many tiles / how much data, no download:
python download_dems.py --out-dir $DEMS --manifest $MANIFEST --dry-run

# Real download (~50 GB CUDEM, takes a while):
python download_dems.py --out-dir $DEMS --manifest $MANIFEST

# Or grab just one subfolder to test the pipeline quickly:
python download_dems.py --out-dir $DEMS --manifest $MANIFEST --only MA_NH_ME
```

This writes `$MANIFEST` (consumed by the benchmark).

---

## Step 4 — Sanity-check the refinement assignment

Confirms which tiles map to which refinement class. No meshing, instant.

```bash
cd $REPO
python build_geom_and_hfun.py --manifest $MANIFEST
```

You should see counts per modulo class (flow_limiter, constant_value,
topo_bound, topo_func, courant, skipped).

---

## Step 5 — Quick smoke test (small manifest, in an allocation)

Verifies the whole pipeline end-to-end (Geom → Hfun → meshdata) before
burning a full SLURM job.

**Do NOT smoke-test against the full 388-raster `$MANIFEST`** — at 1 km
resolution over the whole domain that is the real benchmark and takes
hours. Instead, build a tiny one-subfolder manifest first:

```bash
cd $REPO

# ~39 tiles (New England) instead of 388 — finishes in minutes:
python download_dems.py \
    --out-dir  $DEMS \
    --manifest $REPO/dem_manifest_smoke.json \
    --only     MA_NH_ME
```

Then run the smoke test **inside a short interactive allocation** (never
do multi-process mesh generation on a login node).

> IMPORTANT: launch the benchmark with `srun --mpi=pmi2`, NOT a bare
> `python`. Inside a Slurm allocation, `import ocsmesh` initializes MPI
> (Slurm env vars are present), so even the non-MPI `parallel` mode aborts
> at import with `PMI2_Job_GetId returned 14` if launched as a plain
> `python`. See HERCULES_NOTES.md #7.

```bash
# 1) Grab a small interactive allocation:
salloc -N 1 -n 16 -t 01:00:00 -A nos-surge -p hercules
```

`salloc` drops you into a NEW shell on the compute node — your exported
vars and modules do NOT carry over. Re-set them inside the allocation:

```bash
# 2) Inside the allocation, re-export vars + load env:
export PROJ=/work2/noaa/nos-surge/felicioc/OCSMesh_MPI
export REPO=$PROJ/ocsmesh_mpi_test
export DEMS=$PROJ/stofs_dems
export SHP=$PROJ/inputs/stofs3.shp

module purge
module load intel-oneapi-compilers/2022.2.1
module load intel-oneapi-mpi/2021.7.1
module load hdf5/1.12.2
module load netcdf-c/4.9.0
module load netcdf-fortran/4.6.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocsmesh_mpi_test
cd $REPO
```

```bash
# 3a) Test MPI mode (recommended — this is the point of the benchmark).
#     9 ranks = 1 manager + 8 workers; --nprocs is the worker count.
srun --mpi=pmi2 -n 9 python run_benchmark.py \
    --manifest  $REPO/dem_manifest_smoke.json \
    --shapefile $SHP \
    --out-dir   $REPO/results_smoke \
    --nprocs    8 \
    --modes     mpi
```

```bash
# 3b) OR test serial + parallel (multiprocessing) with a single task.
#     Use -n 1: multiprocessing spawns its own workers; srun -n 1 just
#     gives the one python process a valid PMI context so MPI_Init works.
srun --mpi=pmi2 -n 1 python run_benchmark.py \
    --manifest  $REPO/dem_manifest_smoke.json \
    --shapefile $SHP \
    --out-dir   $REPO/results_smoke \
    --nprocs    8 \
    --modes     serial parallel
```

Notes:
- New England (MA_NH_ME) is outside the box-based refinements (box1 = West
  FL, box2 = SC/GA), so the smoke test exercises the index-modulo
  per-source refinements + global contour/channel, but not
  region_constraint/patch/feature. That is still a valid pipeline check.
- Exit the allocation with `exit` when done.

---

## Step 6 — Single-node SLURM run (serial + parallel + MPI)

The paths in `slurm_single_node.sh` are already set to
`/work2/noaa/nos-surge/felicioc/OCSMesh_MPI`. Verify the `module load`
lines match Hercules, then submit:

```bash
cd $REPO
mkdir -p logs
sbatch slurm_single_node.sh

# Watch it:
squeue -u $USER
tail -f logs/bench_1node_*.out
```

Output lands in `$PROJ/results/single_node_<jobid>/` with
`benchmark_results.json`, `profile_{serial,parallel,mpi}.prof`, and
`benchmark_report.txt`.

---

## Step 7 — Multi-node SLURM run (MPI scaling)

Only after the single-node run works.

```bash
cd $REPO
sbatch slurm_multi_node.sh
```

`RESULTS_DIR` and the MPI scratch live under `/work2` (shared Lustre), so
cross-node `.npz` exchange works. `MPIExecutor.verify_shared_filesystem()`
runs at the start of each MPI dispatch and aborts early with a clear
message if any worker node can't read/write the shared scratch dir.

---

## Step 8 — Generate / combine reports

The SLURM scripts already call `analyze_profile.py`. To combine the
single-node and multi-node runs into one report:

```bash
cd $REPO
python analyze_profile.py \
    --results-dir $PROJ/results/single_node_<jobid> \
                  $PROJ/results/multi_node_<jobid> \
    --out combined_benchmark_report.txt
```

---

## Running the benchmark by hand (outside SLURM)

```bash
cd $REPO

# Serial + parallel (no mpiexec needed):
python run_benchmark.py \
    --manifest $MANIFEST --shapefile $SHP \
    --out-dir ./results --nprocs 40 --modes serial parallel

# MPI — must be launched with srun (Hercules needs --mpi=pmi2).
# Use N+1 ranks: 1 manager (rank 0) + N workers; --nprocs is the worker count.
srun --mpi=pmi2 -n 41 python run_benchmark.py \
    --manifest $MANIFEST --shapefile $SHP \
    --out-dir ./results --nprocs 40 --modes mpi
```

---

## Output reference

### `benchmark_results.json`
Per-mode wall-clock time, node/triangle counts, hfun min/max/mean, and
speedup vs the serial baseline.

### cProfile `.prof` files
```python
import pstats
p = pstats.Stats("profile_mpi.prof"); p.sort_stats("cumulative"); p.print_stats(30)
# or: pip install snakeviz && snakeviz profile_mpi.prof
```

### `benchmark_report.txt`
Timing table + speedups, mesh-quality stats, serial-vs-parallel-vs-MPI
numerical equivalence check, top cProfile hotspots per mode, and a CSV
speedup table.

---

## Notes / known limitations

- `topo_func_constraint` uses a module-level `_half_depth` function (NOT a
  lambda) so the parallel/MPI constraint path does not fall back to serial.
- `method='fast'` is not yet MPI-enabled — the benchmark uses the default
  `exact` method.
- `_apply_features` (contours, channels, flow limiters, constraints) still
  runs on Rank 0 before the MPI meshdata dispatch; only the per-tile
  `meshdata()` calls are distributed. This is the next parallelization
  target.
- On multi-node jobs, intermediate `.npz` files are exchanged via the
  shared filesystem — `RESULTS_DIR` under `/work2` handles this.
- `download_dems.py` selects every other tile per subfolder; edit the
  `_TILES` lists or `select_every_other()` to change coverage.
