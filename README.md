# OCSMesh MPI Benchmark — STOFS-3D-Atlantic

End-to-end test of the new OCSMesh MPI parallelization against the serial
and multiprocessing modes, using real-world DEMs for the STOFS-3D-Atlantic
domain on the NOAA RDHPC (Hercules).

---

## Files

```
ocsmesh_mpi_test/
├── download_dems.py         # Step 1 – download GEBCO + CUDEM 1/9" tiles
├── build_geom_and_hfun.py   # Geom/Hfun recipe (raster order + refinements)
├── run_benchmark.py         # Step 3 – run serial/parallel/MPI + profile
├── analyze_profile.py       # Step 4 – build the human-readable report
├── slurm_single_node.sh     # SLURM wrapper: single node, all 3 modes
├── slurm_multi_node.sh      # SLURM wrapper: multi-node MPI scaling
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

## Prerequisites (one-time)

### 1. OCSMesh + mpi4py in a conda env on Hercules

```bash
module purge
module load intel/2022.1.2        # match to Hercules; sets the MPI toolchain
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocsmesh            # your env name

# Install the MPI branch of OCSMesh in editable mode with the mpi extra:
pip install -e /path/to/OCSMesh[mpi]

# Verify mpi4py is linked against the loaded MPI:
python -c "from mpi4py import MPI; print('MPI', MPI.Get_version())"
```

The conda env MUST be built against the same MPI that `module load` provides,
otherwise `srun`/`mpiexec` and `mpi4py` will disagree at runtime.

### 2. Clone this repo on Hercules

```bash
cd /work/noaa/<project>/<user>
git clone https://github.com/felicio93/ocsmesh_mpi_test.git
cd ocsmesh_mpi_test
```

### 3. Have the STOFS-3D-Atlantic domain shapefile on Hercules

Note its full path — you pass it via `--shapefile`.

---

## Step-by-step (copy-paste)

Set these once per shell session (edit to your paths):

```bash
export PROJ=/work/noaa/<project>/<user>
export REPO=$PROJ/ocsmesh_mpi_test
export DEMS=$PROJ/stofs_dems
export SHP=$PROJ/stofs_domain.shp
export MANIFEST=$REPO/dem_manifest.json

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocsmesh
cd $REPO
```

### Step 1 — Download the DEMs

Run on a login node or a short interactive allocation (network access
required). Re-runnable: existing files are skipped.

```bash
# Dry run first — prints how many tiles / how much data, no download:
python download_dems.py --out-dir $DEMS --manifest $MANIFEST --dry-run

# Real download (~50 GB, takes a while):
python download_dems.py --out-dir $DEMS --manifest $MANIFEST

# Or download only one subfolder to test the pipeline quickly:
python download_dems.py --out-dir $DEMS --manifest $MANIFEST --only MA_NH_ME
```

This writes `dem_manifest.json` (consumed by the benchmark).

### Step 2 — Sanity-check the refinement assignment

Confirms which tiles map to which refinement class. No meshing, instant.

```bash
python build_geom_and_hfun.py --manifest $MANIFEST
```

You should see counts per modulo class (flow_limiter, constant_value,
topo_bound, topo_func, courant, skipped).

### Step 3 — Quick local smoke test (serial + parallel, no MPI)

Verifies the whole pipeline end-to-end before burning a SLURM allocation.
Use `--only MA_NH_ME` in Step 1 first if you want this to be fast.

```bash
python run_benchmark.py \
    --manifest  $MANIFEST \
    --shapefile $SHP \
    --out-dir   $REPO/results_smoke \
    --nprocs    8 \
    --modes     serial parallel
```

### Step 4 — Single-node SLURM run (serial + parallel + MPI)

Edit the `CHANGE_ME` placeholders at the top of `slurm_single_node.sh`
first (conda env, shapefile, DEM dir, results dir, email, module lines):

```bash
mkdir -p $REPO/logs
sbatch slurm_single_node.sh
```

Watch it:

```bash
squeue -u $USER
tail -f $REPO/logs/bench_1node_*.out
```

Output lands in `results/single_node_<jobid>/` with
`benchmark_results.json`, `profile_{serial,parallel,mpi}.prof`, and
`benchmark_report.txt`.

### Step 5 — Multi-node SLURM run (MPI scaling)

Only after the single-node run works. Edit `slurm_multi_node.sh`
placeholders — crucially point `TMPDIR`/results at a **shared**
filesystem (Lustre/GPFS), never node-local `/tmp`.

```bash
sbatch slurm_multi_node.sh
```

`MPIExecutor.verify_shared_filesystem()` runs at the start of each MPI
dispatch and aborts early with a clear message if any worker node can't
read/write the shared scratch dir.

### Step 6 — Generate / combine reports

The SLURM scripts already call `analyze_profile.py`. To combine the
single-node and multi-node runs into one report:

```bash
python analyze_profile.py \
    --results-dir results/single_node_<jobid> \
                  results/multi_node_<jobid> \
    --out combined_benchmark_report.txt
```

---

## Running the benchmark by hand (outside SLURM)

```bash
# Serial + parallel (no mpiexec needed):
python run_benchmark.py \
    --manifest $MANIFEST --shapefile $SHP \
    --out-dir ./results --nprocs 40 --modes serial parallel

# MPI — must be launched with srun/mpiexec.
# Use N+1 ranks: 1 manager (rank 0) + N workers; --nprocs is the worker count.
srun -n 41 python run_benchmark.py \
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
  shared filesystem — set `TMPDIR` to Lustre/GPFS (done in the multi-node
  SLURM script).
- `download_dems.py` selects every other tile per subfolder; edit the
  `_TILES` lists or `select_every_other()` to change coverage.
