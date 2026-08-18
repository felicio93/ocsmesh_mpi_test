# OCSMesh MPI Benchmark — STOFS-3D-Atlantic

End-to-end test of the new OCSMesh MPI parallelization against the serial
(multiprocessing) and multi-node modes, using real-world DEMs for the
STOFS-3D-Atlantic domain on the NOAA RDHPC (Hercules).

All paths in this README and in the SLURM scripts are hard-wired to:

```
/work2/noaa/nos-surge/felicioc/OCSMesh_MPI
```

If you use a different location, edit the `PROJ=` line below and the
`PROJ=` line at the top of each SLURM script (and `final_config.sh`).

---

## Files

```
ocsmesh_mpi_test/
├── download_dems.py            # download GEBCO + CUDEM 1/9" tiles
├── trim_manifest.py            # cut a manifest down to N CUDEM tiles (+GEBCO)
├── build_geom_and_hfun.py      # Geom/Hfun recipe (raster order + refinements)
├── run_benchmark.py            # run serial_mp/parallel/mpi + cProfile
├── analyze_profile.py          # build the human-readable report + stage breakdown
├── slurm_single_node.sh        # SLURM: single node, smoke test (all modes)
├── slurm_multi_node.sh         # SLURM: multi-node MPI (legacy)
├── final_config.sh             # shared config sourced by all final jobs
├── slurm_final_serial_mp.sh    # FINAL: serial_mp baseline (8h)
├── slurm_final_parallel.sh     # FINAL: parallel/multiprocessing (8h)
├── slurm_final_mpi_1node.sh    # FINAL: MPI single node (8h)
├── slurm_final_mpi_multinode.sh# FINAL: MPI multi node (8h)
├── slurm_final_mpi_stress.sh   # FINAL: MPI max-scale stress test (size after smoke)
├── HERCULES_NOTES.md           # running log of HPC gotchas + OCSMesh feedback
└── README.md
```

`build_geom_and_hfun.py` is the module you edit to change *what* refinement
goes *where*. `run_benchmark.py` just imports it.

---

## Benchmark modes

`run_benchmark.py --modes` accepts any of:

| Mode         | ocsmesh `execution_mode` | nprocs | What is parallelized |
|--------------|--------------------------|--------|----------------------|
| `serial_mp`  | `serial`                 | N      | Pool-based feature steps only (contour/channel/flow/const); meshdata dispatch is serial |
| `parallel`   | `parallel`               | N      | All per-tile steps + meshdata dispatch via multiprocessing.Pool |
| `mpi`        | `mpi`                    | N      | meshdata dispatch via MPIExecutor across ranks (1 manager + N workers) |

(There is also a `serial_true` mode — true single core, `nprocs=1` — but it
is not used in the current benchmark.)

All modes run the IDENTICAL manifest + recipe, so the resulting meshes must
match (numerical-equivalence check in the report).

---

## Required OCSMesh branch + Gmsh

The MPI implementation and the fixes this benchmark depends on live on the
**`felicio/mpi-fixes`** branch of `noaa-ocs-modeling/OCSMesh` (branched off
`dev`). A plain `main`/`dev` clone will NOT work.

Gmsh is the meshing engine and **must be installed in the conda env**, or
the meshdata stage aborts with `ImportError: Gmsh library not installed`:

```bash
conda activate ocsmesh_mpi_test
python -c "import gmsh" 2>&1 || pip install gmsh
python -c "import gmsh; print('gmsh OK')"
```

---

## What is being tested

**Global mesh size bounds:** `hmin = 1000 m` (finest), `hmax = 7000 m`
(coarsest / background).

**DEMs:** 1 GEBCO tile (full domain, 15", lowest priority) + CUDEM 1/9"
tiles (every other tile per subfolder). CUDEM tiles override GEBCO where
they overlap.

**Per-source refinements — assigned by list index (modulo 6, CUDEM only):**

| `(cudem_pos) % 6` | Refinement |
|---|---|
| 0 | `add_subtidal_flow_limiter` (hmin=1000, hmax=7000, −200→0 m) |
| 1 | `add_constant_value` (1000 m, −5→0 m) |
| 2 | `add_topo_bound_constraint` (1500 m, −2→1 m, min) |
| 3 | `add_topo_func_constraint` (depth/2, −3000→0 m, min) |
| 4 | `add_courant_num_constraint` (0.9, dt=150 s, amp=2 m) |
| 5 | *skipped* |

**Global refinements (all rasters):** `add_contour` (0 m + −200 m),
`add_channel`.

**Box-based refinements (fixed lat/lon):**
- Box1 `(-85..-82, 25..31)` → `add_region_constraint` (max 3500 m)
- Box2 `(-80..-77, 31..35)` → `add_patch` (1000 m) + `add_feature` (line)

---

## Recipe cost knobs (IMPORTANT for fitting the 8h wall clock)

Profiling on Hercules revealed two dominant, un-MPI-parallelized costs in
the `exact` method. Two flags let you trim them:

- **`--light-features`** (env `LIGHT_FEATURES=1`): skip the global
  `add_contour` + `add_channel`. These are an O(tiles × contour-segments)
  bottleneck; on the full recipe `_apply_features` alone can take many
  hours even on a handful of tiles.

- **`--skip-topofunc`** (env `SKIP_TOPOFUNC=1`, default ON in the smoke
  script): skip `add_topo_func_constraint`. That constraint carries a
  callable which forces OCSMesh's `_apply_constraints` to run **serially
  even in parallel/mpi modes** (~44 min/tile). Skipping it removes the
  single largest serial bottleneck AND lets the constraint stage actually
  parallelize (so parallel/mpi show a real speedup on constraints).

The per-tile `topo_bound` / `topo_func` / `courant` constraints are the
most expensive step: each transforms the full-resolution raster
coordinates through pyproj and does windowed KDTree/array math over tens of
millions of points.

---

## Step 1 — Set up shell environment (once per session)

```bash
export PROJ=/work2/noaa/nos-surge/felicioc/OCSMesh_MPI
export REPO=$PROJ/ocsmesh_mpi_test
export DEMS=$PROJ/stofs_dems
export SHP=$PROJ/inputs/stofs3.shp

mkdir -p $PROJ
```

---

## Step 2 — Clone and install OCSMesh (editable) + clone this repo

Editable install (`-e`) means any later `git pull`/branch switch in the
OCSMesh repo is picked up immediately with no reinstall.

```bash
# Load the toolchain FIRST so mpi4py builds against the right MPI.
module purge
module load intel-oneapi-compilers/2022.2.1
module load intel-oneapi-mpi/2021.7.1
module load hdf5/1.12.2
module load netcdf-c/4.9.0
module load netcdf-fortran/4.6.0

# Activate the conda env. NOTE: source conda's init directly by its
# hardcoded path — 'conda info --base' fails inside SLURM jobs because
# conda is not on PATH at job start. The SLURM scripts use:
#   source /work2/noaa/nos-surge/felicioc/envs/miniconda3/etc/profile.d/conda.sh
source /work2/noaa/nos-surge/felicioc/envs/miniconda3/etc/profile.d/conda.sh
conda create -y -n ocsmesh_mpi_test python=3.10
conda activate ocsmesh_mpi_test

# --- Clone OCSMesh and check out the felicio/mpi-fixes branch ---
# The MPI code + benchmark fixes live on felicio/mpi-fixes (off dev).
cd $PROJ
git clone https://github.com/noaa-ocs-modeling/OCSMesh.git
cd OCSMesh
git checkout felicio/mpi-fixes        # <-- REQUIRED
git pull

# --- Editable install of ocsmesh (core deps only) ---
# Do NOT use the ".[mpi]" extra on HPC — it pulls a prebuilt mpi4py wheel
# linked to a generic MPI that fails under srun. See HERCULES_NOTES #1/#2.
pip install -e .

# --- Gmsh meshing engine (REQUIRED) ---
python -c "import gmsh" 2>&1 || pip install gmsh

# --- Build mpi4py FROM SOURCE against the loaded Intel MPI module ---
which mpicc                                  # must resolve to Intel MPI
MPICC=$(which mpicc) pip install --no-binary=mpi4py --no-cache-dir mpi4py

# Verify imports:
python -c "import ocsmesh; print('ocsmesh at', ocsmesh.__file__)"
python -c "from ocsmesh.mpi import MPIExecutor; print('ocsmesh.mpi OK')"

# Verify mpi4py — MUST run under srun (bare 'python -c' aborts with
# 'PMI2_Job_GetId returned 14'; see HERCULES_NOTES #3).
srun --mpi=pmi2 -n 2 python -c \
  "from mpi4py import MPI; c=MPI.COMM_WORLD; print('rank', c.Get_rank(), 'of', c.Get_size())"

# --- Clone this benchmark repo into $PROJ ---
cd $PROJ
git clone https://github.com/felicio93/ocsmesh_mpi_test.git
```

> Editable install: `$PROJ/OCSMesh` is your live source tree. Update with
> `cd $PROJ/OCSMesh && git pull` — no reinstall needed.

---

## Step 3 — Download the DEMs

Run on a login node (network required). Re-runnable: existing files are
skipped.

**GEBCO note:** the NCEI ETOPO2022 THREDDS server has no WCS, so the
deep-ocean tile is NOT auto-downloaded (HERCULES_NOTES #5). Provide it:
download a GEBCO grid for the domain from https://download.gebco.net/
(GeoTIFF) and drop it in `$DEMS/gebco/` (auto-detected) or point
`GEBCO_LOCAL` at it.

```bash
cd $REPO
export GEBCO_LOCAL=$DEMS/gebco/gebco_2024_n56.0_s5.0_w-100.0_e-50.0.tif

# The smoke SLURM script downloads a New-England subset automatically, but
# you can also do it by hand:
python download_dems.py --out-dir $DEMS \
    --manifest $REPO/dem_manifest_smoke.json --only MA_NH_ME
```

---

## Step 4 — Single-node smoke test (fits 8h)

`slurm_single_node.sh` is the validation run. It:

1. Downloads the New-England smoke manifest if missing.
2. Trims it to **7 tiles** (6 CUDEM = 1 per refinement class + 1 GEBCO)
   via `trim_manifest.py`, so `serial_mp` fits in 8h.
3. Runs `serial_mp` + `parallel` (single rank), then `mpi` (80 ranks).
4. Writes results, `.2dm` meshes, cProfile, and a report.

Env toggles (both applied to all modes so inputs stay identical):

- `LIGHT_FEATURES=1` — skip contour/channel (recommended for fast smoke).
- `SKIP_TOPOFUNC=1`  — skip topo_func_constraint (default ON; removes the
  ~44 min/tile serial bottleneck).

```bash
cd $REPO
mkdir -p logs
LIGHT_FEATURES=1 sbatch slurm_single_node.sh

# Watch it:
squeue -u $USER
tail -f logs/bench_1node_*.err     # OCSMesh progress is on stderr
```

Output: `$PROJ/results/single_node_<jobid>/` with
`serial_parallel/` and `mpi/` subdirs, each containing
`benchmark_results.json`, `profile_*.prof`, `hfun_*.2dm`, and the report.

> TMPDIR is set to `${RESULTS_DIR}/tmp` (on `/work2`, Lustre) for ALL
> modes. Do NOT use node-local `/tmp` — 79 workers writing clipped rasters
> overflow it and OCSMesh aborts with 'No space left on device'.

---

## Step 5 — Final per-mode runs (8h each)

Once the smoke test passes, the final benchmark runs one mode per SLURM
job, all sourcing `final_config.sh` (single source of truth for manifest,
recipe flags, hmin/hmax). Edit `final_config.sh` to set the workload
(`FINAL_MANIFEST` / `FINAL_N_CUDEM`) and recipe knobs, then:

```bash
cd $REPO
sbatch slurm_final_serial_mp.sh
sbatch slurm_final_parallel.sh
sbatch slurm_final_mpi_1node.sh
sbatch slurm_final_mpi_multinode.sh    # edit --nodes for the scaling study
```

Size the workload so `serial_mp` (the slowest, binding mode) fits in ~7h;
the same manifest is then used by every mode for a valid comparison.

---

## Step 6 — MPI stress test (push the implementation to its limits)

`slurm_final_mpi_stress.sh` runs the full recipe on as many tiles / nodes
as possible (MPI only — serial could never finish this in 8h). **Sizing is
a placeholder**: set `STRESS_N_CUDEM` and `--nodes` after the MPI smoke
test measures per-tile MPI cost and the rank-0-only `_apply_features`
fraction (which caps the achievable speedup — see limitations).

---

## Step 7 — Generate / combine reports

The SLURM scripts already call `analyze_profile.py`. To combine multiple
per-mode runs into one report (timing table, speedups, numerical
equivalence check, and the pipeline stage breakdown):

```bash
cd $REPO
python analyze_profile.py \
    --results-dir $PROJ/results/final_serial_mp_<jobid> \
                  $PROJ/results/final_parallel_<jobid> \
                  $PROJ/results/final_mpi_1node_<jobid> \
    --out combined_report.txt
```

---

## Output reference

### `benchmark_results.json`
Per-mode wall-clock time, node/triangle counts, hfun min/max/mean,
`ocsmesh_execution_mode`, `effective_nprocs`, and speedup vs baseline.

### `hfun_<mode>.2dm`
The generated mesh for each mode, for visual comparison in QGIS. All modes
should produce matching meshes.

### cProfile `.prof` files
```python
import pstats
p = pstats.Stats("profile_mpi.prof"); p.sort_stats("cumulative"); p.print_stats(30)
```

### report `.txt`
Timing table + speedups, mesh-quality stats, numerical-equivalence check,
top cProfile hotspots per mode, a CSV speedup table, and a **Pipeline Stage
Breakdown** (self-time bucketed into geom_build / hfun_construct /
apply_features / calc_write_meshdata / composite / mesh_write_2dm /
raster_io / kdtree).

---

## Notes / known limitations

- **`add_topo_func_constraint` forces serial constraint application.** Its
  stored callable can't be pickled for the Pool, so OCSMesh's
  `_apply_constraints` falls back to serial even in parallel/mpi modes
  (~44 min/tile). Use `--skip-topofunc` for tractable runs. (This is a real
  OCSMesh limitation worth fixing.)
- **The topo/courant constraints are the dominant cost** in the `exact`
  method — full-resolution coordinate transforms + KDTree per tile.
- **`_apply_features` runs on Rank 0 only** even in MPI mode (contours,
  channels, flow limiters, constraints). Only the per-tile `meshdata()`
  dispatch is MPI-distributed (`_calculate_and_write_hfun_to_disk`). This
  caps the achievable MPI speedup (Amdahl) and is the next parallelization
  target — see the `TODO(mpi)` markers in `ocsmesh/hfun/collector.py`.
- **`method='fast'` is not MPI-enabled** — the benchmark uses `exact`.
- **TMPDIR must be on shared `/work2`** (Lustre), never node-local `/tmp`.
- On multi-node jobs, intermediate `.npz`/raster files are exchanged via
  the shared filesystem; `MPIExecutor.verify_shared_filesystem()` checks
  this at dispatch start and aborts early with a clear message otherwise.
```
