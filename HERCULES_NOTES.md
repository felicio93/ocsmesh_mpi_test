# Hercules Gotchas & OCSMesh MPI Feedback Log

Running log of issues hit while testing the OCSMesh MPI implementation on
the NOAA RDHPC (Hercules), with root cause, workaround, and whether it is
something the OCSMesh side can fix/improve.

Share the "OCSMesh-side?" column with the OCSMesh MPI developer.

Environment (as tested):
- Cluster: Hercules (Rocky 9, Slurm)
- Modules: `intel-oneapi-compilers/2022.2.1`, `intel-oneapi-mpi/2021.7.1`,
  `hdf5/1.12.2`, `netcdf-c/4.9.0`, `netcdf-fortran/4.6.0`
- Conda env: `ocsmesh_mpi_test` (Python 3.10)
- OCSMesh: editable install from `$PROJ/OCSMesh`

---

## #1 — `pip install -e .` does not install mpi4py

**Symptom**
After `pip install -e .`, `import mpi4py` fails with
`ModuleNotFoundError: No module named 'mpi4py'`, even though ocsmesh
imports fine.

**Root cause**
`mpi4py` is declared as an *optional* dependency in OCSMesh's
`pyproject.toml`:

```toml
dependencies = [ ... ]          # core deps — mpi4py NOT included
[project.optional-dependencies]
mpi = ["mpi4py"]                # only installed via the ".[mpi]" extra
```

A plain `pip install -e .` installs only core deps by design.

**Workaround**
Install mpi4py explicitly (see #2 — do NOT just add `.[mpi]` on HPC).

**OCSMesh-side?**
Docs only. The optional-extra design is intentional. Worth documenting
clearly in the OCSMesh MPI setup guide that mpi4py is not auto-installed.

---

## #2 — `.[mpi]` extra pulls a prebuilt mpi4py wheel linked to the wrong MPI

**Symptom**
`pip install -e ".[mpi]"` succeeds and `import mpi4py` works on the login
node, but under `srun` the MPI layer misbehaves / does not see the Intel
MPI ranks (silent wrong-MPI, or init failures).

**Root cause**
On PyPI, `pip` grabs a prebuilt mpi4py wheel that bundles / links a
generic MPI (often MPICH), NOT the site's Intel MPI. mpi4py must be
compiled against the exact MPI that `srun` uses.

**Workaround**
Build mpi4py from source against the loaded Intel MPI module:

```bash
module load intel-oneapi-mpi/2021.7.1
which mpicc                                   # must resolve to Intel MPI
MPICC=$(which mpicc) pip install --no-binary=mpi4py --no-cache-dir mpi4py
```

The build log should say "Building wheel for mpi4py ... done" (compiled),
not "Downloading mpi4py-...whl".

**OCSMesh-side?**
Docs mainly. Could optionally add a note / helper in the OCSMesh MPI docs
recommending the `MPICC=... --no-binary=mpi4py` pattern for HPC installs.

---

## #3 — `PMI2_Job_GetId returned 14` when verifying mpi4py

**Symptom**
```
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
Abort(1090831) on node 0 (rank 0 in comm 0): Fatal error in PMPI_Init_thread:
Other MPI error, error stack:
MPIR_Init_thread(176): ...
MPIR_pmi_init(167)...: PMI2_Job_GetId returned 14
```

**Root cause**
Running an MPI program as a bare `python -c ...` on a login / interactive
shell. `MPI_Init` tries to contact a PMI (process management interface)
server that only exists when the job is launched under `srun`/`mpirun`.
No launcher → no PMI → abort at init. This is NOT an OCSMesh bug and NOT
an install problem — mpi4py is fine.

**Workaround**
Always launch MPI code through Slurm with a matching PMI:

```bash
# Inside an allocation:
salloc -N 1 -n 2 -t 00:10:00 -A nos-surge -p hercules

# Verify with srun (pick the PMI your Slurm supports):
srun --mpi=pmi2 -n 2 python -c \
  "from mpi4py import MPI; c=MPI.COMM_WORLD; print('rank', c.Get_rank(), 'of', c.Get_size())"

# If pmi2 is not supported, list options and try another (e.g. pmix):
srun --mpi=list
```

If Intel MPI + Slurm still disagree, point Intel MPI at the Slurm PMI lib:
```bash
export I_MPI_PMI_LIBRARY=/opt/slurm/lib/libpmi2.so   # path is site-specific
```

**OCSMesh-side?**
Partially. The environment plumbing is outside OCSMesh, BUT
`MPIExecutor` could catch a failed `MPI_Init` / PMI error and emit a
friendly message such as:
> "MPI failed to initialize. Are you running under srun/mpirun with the
>  correct --mpi=pmiX? A bare 'python script.py' cannot initialize MPI."
instead of surfacing a raw `PMI2_Job_GetId` abort. Good UX improvement.

---

## #4 — Confirmed working: `srun --mpi=pmi2` on Hercules

**Symptom / status**
Resolution of #3. With mpi4py built from source against Intel MPI (#2),
this launches correctly:

```bash
srun --mpi=pmi2 -n 2 python -c \
  "from mpi4py import MPI; c=MPI.COMM_WORLD; print('rank', c.Get_rank(), 'of', c.Get_size())"
# rank 1 of 2
# rank 0 of 2
```

**Takeaway**
`--mpi=pmi2` is the confirmed working PMI on Hercules. Both benchmark
SLURM scripts now pass `--mpi=pmi2` to every `srun` MPI launch. If a
future Hercules change breaks this, run `srun --mpi=list` and switch to
the supported type (e.g. `pmix`).

**Confirmed Slurm/allocation values (from a working nos-surge job card):**
- `--account=nos-surge`
- `--partition=hercules`
- Node size: **80 cores / 512 GB per node** (use `--exclusive`)
- Windfall queue note: max 450 nodes, no walltime limit but lower priority;
  for quick turnaround keep walltime <= 8 hrs.

**OCSMesh-side?**
No — site config. Recorded here so the SLURM scripts and the OCSMesh MPI
docs can cite a known-good Hercules launch recipe.

---

## #5 — ETOPO2022 THREDDS WCS returns HTTP 400 (no WCS for that dataset)

**Symptom**
`download_dems.py` GEBCO step fails on every attempt:
```
400 Client Error: 400 for url:
https://www.ngdc.noaa.gov/thredds/wcs/global/etopo2022/ETOPO_2022_v1_15s_N90W180_bed.nc?...WCS...
WARNING: GEBCO download failed.
```

**Root cause**
The NCEI THREDDS server does NOT expose a WCS service for the ETOPO2022
global dataset (and the single global `N90W180` NetCDF the URL assumed
does not exist there). The dataset is served as **per-tile 15° NetCDF
files** via fileServer / OPeNDAP / NCSS only — not WCS. So the WCS
`GetCoverage` request is invalid and returns 400.

Confirmed catalog layout:
```
/thredds/catalog/global/ETOPO2022/15s/15s_bed_elev_netcdf/catalog.html
  -> ETOPO_2022_v1_15s_N60W105_bed.nc, ...N60W090..., etc. (15°x15° tiles)
Plain-file download base:
  /thredds/fileServer/global/ETOPO2022/15s/15s_bed_elev_netcdf/<tile>.nc
```

**Workaround (recommended)**
Provide the deep-ocean background GeoTIFF yourself — no WCS needed:

1. Download a GEBCO grid for the domain from https://download.gebco.net/
   (GeoTIFF), e.g. `gebco_2024_n56.0_s5.0_w-100.0_e-50.0.tif`.
2. Drop it in `$DEMS/gebco/` OR set `GEBCO_LOCAL` to its full path:
   ```bash
   export GEBCO_LOCAL=$DEMS/gebco/gebco_2024_n56.0_s5.0_w-100.0_e-50.0.tif
   ```
`download_dems.py` now (a) honors `GEBCO_LOCAL`, (b) auto-detects any
`*.tif` already sitting in `$DEMS/gebco/`, and only then (c) falls back
to downloading + merging the ETOPO2022 15" NetCDF tiles with rasterio.

**OCSMesh-side?**
No — this is a data-sourcing issue in the benchmark's download helper,
now fixed. Not an OCSMesh code concern.

---

## #6 — CUDEM tile downloads 404 (hardcoded filenames were stale/guessed)

**Symptom**
Many CUDEM tiles 404 during download, e.g.:
```
404 Client Error: Not Found for url:
https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483/FL/ncei19_n25x00_w082x00_2020v1.tif
```
Some subfolders (MA_NH_ME, rima, chesapeake_bay, NC) downloaded fine;
others (southeast, FL, LA_MS, TX) had many misses.

**Root cause**
`download_dems.py` originally carried HARDCODED tile-name lists. For the
subfolders that were scraped completely up-front they were correct; for
the larger ones (FL ~109 tiles, LA_MS ~119, TX ~97, southeast ~75) the
lists were partially hand-guessed, so the filenames (wrong year/version
suffix, wrong lon/lat step) did not exist on S3 → 404.

**Fix**
Scrape each subfolder's `index.html` at RUNTIME and regex out the real
`ncei19_*.tif` names, then pick every other tile:
```
https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/<subfolder>/index.html
```
Implemented as `get_subfolder_tiles(subfolder)` with a compiled regex
`ncei19_[ns]\\d+[xX]\\d+_[ew]\\d+[xX]\\d+_\\d{4}v\\d+\\.tif` (uppercase X
allowed — some FL tiles use it). No filenames are hardcoded anymore, so
the lists can never go stale.

**OCSMesh-side?**
No — benchmark data-sourcing only. Not an OCSMesh code concern.

---

## #7 — `ocsmesh.mpi` missing, and MPI init on import inside an allocation

Two related findings hit in sequence during the first smoke test.

### 7a — `ModuleNotFoundError: No module named 'ocsmesh.mpi'`

**Symptom**
```
from ocsmesh.mpi import (...)
ModuleNotFoundError: No module named 'ocsmesh.mpi'
```
even though `ocsmesh` imports fine and is the editable install at
`$PROJ/OCSMesh/ocsmesh/__init__.py`.

**Root cause**
The MPI implementation (`ocsmesh/mpi.py`, `MPIExecutor`) lives on the
**`dev`** branch of OCSMesh. A default clone checks out `main`/`master`,
which does not contain `mpi.py`.

**Workaround**
```bash
cd $PROJ/OCSMesh
git checkout dev
git pull
ls -la ocsmesh/mpi.py                       # confirm it exists
python -c "from ocsmesh.mpi import MPIExecutor; print('mpi OK')"
```
Editable install picks up the branch switch immediately — no reinstall.

**OCSMesh-side?**
Docs / release. Until `dev` merges to `main`, the MPI feature isn't on
the default branch, so any "clone + install" following the main README
will lack it. Worth a prominent note in the OCSMesh MPI docs (and this
benchmark's README now does `git checkout dev` explicitly).

### 7b — `PMI2_Job_GetId returned 14` even in non-MPI (`parallel`) mode

**Symptom**
Inside a `salloc` allocation, launching with a bare `python` (not srun)
aborts at import, even for `--modes parallel` (which uses
multiprocessing, not MPI):
```
Abort(1090831) ... PMIR_pmi_init(167)...: PMI2_Job_GetId returned 14
```

**Root cause**
`import ocsmesh` runs `ocsmesh/__init__.py`, which calls
`_configure_mpi_environment()`. Under a Slurm allocation the MPI env
vars are present, so `mpi4py`/`MPI_Init` fires on plain import — but a
bare `python` process has no PMI server, so init aborts. (On a login
node with no allocation this does not happen, because the Slurm env vars
are absent and MPI stays dormant.)

**Workaround**
Always launch through `srun --mpi=pmi2`, even for non-MPI modes:
```bash
# parallel / serial (single task; multiprocessing spawns its own workers):
srun --mpi=pmi2 -n 1 python run_benchmark.py ... --modes parallel

# mpi (N+1 ranks = 1 manager + N workers):
srun --mpi=pmi2 -n 9 python run_benchmark.py ... --nprocs 8 --modes mpi
```

**OCSMesh-side?**
Yes — UX improvement candidate. Having `MPI_Init` fire as a side effect
of plain `import ocsmesh` (inside an allocation) is surprising and makes
non-MPI use awkward under Slurm. The developer may want MPI
initialization to stay lazy until an MPI feature is actually invoked
(e.g. only inside `MPIExecutor`), so `import ocsmesh` and
`multiprocessing`-based runs don't require a launcher.

---

## #8 — `in_item.clip()` AttributeError on str/Path DEM inputs (OCSMesh bug)

**Symptom**
```
AttributeError: 'str' object has no attribute 'clip'
  File ".../ocsmesh/hfun/collector.py", line 1012, in __init__
    in_item.clip(clip_shape)
```

**Root cause**
In `HfunCollector.__init__`, when the input is a str/Path .tif, `in_item` is
reassigned to `str(in_item)` and a `Raster` object is created separately as
`raster = Raster(in_item)`. The base_shape clip branch then called
`in_item.clip()` (on the plain string) instead of `raster.clip()`.

**Workaround / fix**
Fixed on `felicio/mpi-fixes`: call `raster.clip(clip_shape)` (matches the
base_mesh branch just below). Every str/Path + base_shape call hit this.

**OCSMesh-side?** Code fix needed (done on felicio/mpi-fixes).

---

## #9 — MPI failures cascade: mp start method + Pool workers initializing MPI

**Symptom**
Under a SLURM allocation, `--modes parallel` (multiprocessing) aborted with
`PMI_Init returned 14` flooding stderr — one abort per Pool worker.

**Root cause (two linked bugs)**
1. `run_benchmark.py` imported `from ocsmesh.mpi import ...` BEFORE `import
   ocsmesh`, bypassing `ocsmesh/__init__.py` which calls
   `_configure_mpi_environment()`. So the multiprocessing start method was
   never set to 'spawn'.
2. Even after fixing (1), `set_start_method('spawn', force=False)` is a no-op
   if the method was already locked to 'fork' by an earlier import. And the
   spawned/forked Pool workers re-import ocsmesh, which (because SLURM sets
   `SLURM_NTASKS`) made `_get_mpi_comm()` attempt `MPI_Init` in every worker —
   but workers have no PMI server → abort.

**Workaround / fix**
- `run_benchmark.py`: `import ocsmesh` FIRST (runs `_configure_mpi_environment`).
- `ocsmesh/mpi.py`: `set_start_method('spawn', force=True)`.
- `ocsmesh/mpi.py`: `_get_mpi_comm()` returns None for non-'MainProcess'
  processes (Pool workers), so MPI is never initialized in a worker.
All on `felicio/mpi-fixes`.

**OCSMesh-side?** Code fix needed (done). Also a UX note: MPI init as a side
effect of import under a SLURM allocation is surprising.

---

## #10 — SLURM forces TMPDIR to node-local scratch inside srun → disk full

**Symptom**
MPI mode aborted early in `raster.clip()`:
```
_tiffWriteProc: No space left on device.
rasterio.errors.RasterioIOError: Write failed.
```
`/work2` had 6.3 PB free, and the job's `${RESULTS_DIR}/mpi_tmp` was empty.

**Root cause**
OCSMesh sets its temp dir ONCE at import via `tempfile.gettempdir()` (reads
`TMPDIR`). Hercules' SLURM prolog forces `TMPDIR=/local/scratch/$USER/$JOBID`
(a small node-local disk) inside EVERY srun — and this overrides any exported
`TMPDIR`, even `srun --export=ALL,TMPDIR=...`. With 80 MPI ranks each writing
full-resolution clipped-raster .tif files to that small local disk, it fills
up. (parallel mode with a single rank happened to fit, which is why only MPI
failed.)

**Proof (on Hercules):**
```
srun --export=ALL,TMPDIR=/work2/... bash -c 'python -c "import tempfile; print(tempfile.gettempdir())"'
  -> /local/scratch/...           # overridden!
srun bash -c 'export TMPDIR=/work2/...; python -c "import tempfile; print(tempfile.gettempdir())"'
  -> /work2/...                   # sticks
```

**Workaround**
Set `TMPDIR` INSIDE the rank shell, after the prolog runs:
```bash
srun --mpi=pmi2 --ntasks=80 bash -c "export TMPDIR='${MPI_TMPDIR}'; exec python run_benchmark.py ..."
```
All job scripts now wrap the python call this way.

**OCSMesh-side?** Partially. Site config is the cause, but OCSMesh could
expose a way to set its working/temp directory explicitly (rather than reading
`TMPDIR` once at import), which would make HPC temp management robust.

---

## #11 — Gmsh not installed (meshing engine)

**Symptom**
```
ImportError: Gmsh library not installed.
  File ".../ocsmesh/engines/gmsh.py", line 31
```
Occurs only after `_apply_features` completes, at the `_calculate_and_write_
hfun_to_disk` (meshing) stage — so a run can burn hours before hitting it.

**Root cause**
`gmsh` is a separate pip package and was not in the conda env.

**Workaround**
`pip install gmsh` in the env. Verify: `python -c "import gmsh"`.

**OCSMesh-side?** Docs — Gmsh should be listed as a required runtime dep for
the meshing engine in the OCSMesh install guide.

---

## #12 — `add_feature` crashes on empty channel points (OCSMesh bug)

**Symptom**
```
ValueError: data must be of shape (n, m), where there are n points of dimension m
  File ".../ocsmesh/hfun/raster.py", line 1229, in add_feature
    tree = cKDTree(np.array(points))
```
Hit inside `_apply_channels` when `add_channel` found no channels on a tile.

**Root cause**
`add_feature` built a KDTree from an empty `points` list without guarding.

**Workaround / fix**
Fixed on `felicio/mpi-fixes`: skip the window when `len(points) == 0`.

**OCSMesh-side?** Code fix needed (done).

---

## #13 — Runtime cost: constraints and contours dominate; MPI covers one stage

**Observation (from this benchmark study, not a bug)**
- `add_topo_bound_constraint`, `add_topo_func_constraint`, and
  `add_courant_num_constraint` each cost ~3 h/tile in serial. They transform
  the full-resolution raster coordinates through pyproj and do windowed KDTree
  work per tile.
- `add_topo_func_constraint` additionally forces `_apply_constraints` to run
  SERIALLY even in parallel/mpi modes (its callable can't be pickled for the
  Pool).
- Global `add_contour` + `add_channel` are an O(tiles × contour-segments)
  cost, also on rank 0.
- The MPI implementation parallelizes `hfun.meshdata()`'s per-tile
  triangulation (`_calculate_and_write_hfun_to_disk_mpi`) — NOT
  `_apply_features` (constraints/contours) and NOT `MeshDriver.run()` (final
  mesh). Both remain serial/rank-0.

**Implication**
Amdahl's law caps the overall speedup MPI can deliver on the realistic recipe.
The benchmark therefore uses recipe-trimming flags (`--light-features`,
`--skip-topofunc`, `--skip-constraints`) to keep runs tractable, and profiles
two things separately:
  - Profile A: the serial constraint cost (full recipe, serial_mp).
  - Profile B: the MPI speedup on the meshdata stage (constraints skipped).

**OCSMesh-side?** The next parallelization targets are `_apply_features`
(constraints/contours) and possibly `MeshDriver.run()`. `TopoFuncConstraint`
should use a picklable callable so the parallel constraint path isn't disabled.

---

## #14 — `RegionConstraint._apply_rate` is the remaining serial bottleneck (not topo/courant)

**Observation (measured on job 9600559, MPI mode, 7 tiles, --skip-constraints active)**

Even with `--skip-constraints` (which skips topo_bound/topo_func/courant), the
`add_region_constraint(value=3500, rate=0.05)` on BOX1 and the
`add_patch`/`add_feature` on BOX2 ran serially on rank 0 in MPI mode and
dominated the total runtime:

```
cProfile top-20 (mpi, 3031s total):
    1    _apply_features           2295s  (rank 0 only)
   30    apply_constraints         1770s  (HfunRaster.apply_constraints)
    1    _apply_constraints_serial 1718s
    7    RegionConstraint.apply    1658s
    7    _apply_rate                756s  ← expensive: distance expansion
   14    add_feature                497s  ← KDTree distance calc
    1    _calculate_write_mpi       659s  ← MPI dispatch (the good part)
    1    _dispatch                  616s  ← actual MPI parallelism
```

**Root cause**

The `rate=0.05` parameter on `add_region_constraint` triggers
`_apply_rate`: a per-window KDTree distance expansion across the full-
resolution raster grid. Same expensive path as topo/courant constraints.
Per-tile cost: ~107s (tottime) × 7 tiles = ~750s. With overhead: ~1718s
serial in MPI mode (rank 0 only).

`add_patch` and `add_feature` also go through `add_feature()` →
`cKDTree` distance query → expensive (497s for add_feature, 452s for
to_crs reprojection).

**Comparison: parallel mode ran constraints in parallel**
In the same job's parallel mode run (15:54 → 16:08):
  `_apply_constraints_parallel`: 7 Pool workers → ~14 min (vs 29 min serial in MPI)
This confirms Soroosh's point: Pool workers CAN parallelize constraints; the
MPI mode currently does NOT use Pool workers for this step on rank 0.

**Workaround**
Use `--skip-box-refinements` to skip region_constraint, patch, and feature
for smoke test and Profile B runs. This isolates the pure Gmsh meshdata
dispatch (the stage MPI actually parallelizes). Keep OFF for Profile A
(full realistic recipe — these box refinements are physically meaningful for
the STOFS domain).

Note: the box refinements belong OUTSIDE the MA/NH/ME smoke test region
(BOX1 = West FL shelf, BOX2 = SC/GA coast), so they produce no visible
effect on the smoke meshes but still cost ~29 min serial on rank 0.

**OCSMesh-side?** Yes — `_apply_features` on rank 0 is the next
parallelization target. Running `add_region_constraint`/`add_patch`/
`add_feature` via Pool workers on rank 0 (as parallel mode already does
for flow_limiter/const_value) would cut this ~2×; MPI-distributing it
across ranks (the TODO(mpi) in collector.py) would cut it further.

---

---

## #15 — OCSMesh branch changes: mpi now uses parallel refinement path + gmsh boundary defaults to 'adapt'

**Context (session 2025-08-20)**

Two changes were made to `felicio/mpi-fixes` as a result of the smoke-matrix
design work with Soroosh. Both fix performance/quality issues that the smoke
matrix (HERCULES_NOTES #16) was designed to measure.

### 15a — MPI now routes _apply_constraints / _apply_flow_limiters / _apply_const_val through the Pool-based parallel path

**File:** `ocsmesh/hfun/collector.py` (three dispatch methods)

**Before:**
```python
if self.execution_mode == 'parallel' and self._nprocs > 1:
    self._apply_constraints_parallel()   # (and similarly for flow_limiters, const_val)
else:
    self._apply_constraints_serial()     # ← MPI fell here!
```

**After:**
```python
if self.execution_mode in ('parallel', 'mpi') and self._nprocs > 1:
    self._apply_constraints_parallel()   # now reached in MPI mode too
else:
    self._apply_constraints_serial()
```

**Why this matters:**
Prior to this fix, MPI mode fell through to the serial path for ALL three
refinement dispatchers, even though OCSMesh already had a perfectly good
Pool-based parallel implementation. The `_apply_features` call is guarded by
`if is_manager:` upstream (collector.py:1077), so only rank 0 ever reaches
these dispatchers — spawning a Pool there is safe. The net result:

  MPI mode before: refinements serial on rank 0, meshdata MPI-distributed
  MPI mode after:  refinements Pool-parallel on rank 0, meshdata MPI-distributed

This makes MPI faster overall than pure `parallel` (which has Pool-parallel
refinements but single-threaded per-tile meshdata dispatch), not just for the
meshdata stage. The smoke-matrix Config C vs B (HERCULES_NOTES #16) directly
measures this gain.

**Caveat:** `TopoFuncConstraint` still auto-falls back to serial (its lambda
can't be pickled for Pool). That's why topofunc is excluded from all
smoke-matrix configs — see #16.

### 15b — gmsh engine now defaults boundary representation to 'adapt'

**File:** `ocsmesh/hfun/raster.py` (~line 298)

**Before:**
```python
engine = get_mesh_engine(mesh_engine, **mesh_options)
# → gmsh GmshOptions defaults to bnd_representation='fixed'
```

**After:**
```python
if mesh_engine == 'gmsh':
    mesh_options.setdefault('bnd_representation', 'adapt')
engine = get_mesh_engine(mesh_engine, **mesh_options)
```

**Why this matters:**
`'adapt'` causes Gmsh to resample the tile boundary vertices to match the
hfun resolution (via `utils.resample_geom_by_hfun`) BEFORE meshing. `'fixed'`
locks the original dense boundary vertices as hard points, which can generate
sliver triangles where the boundary resolution is much finer than the
requested element size.

`setdefault` means:
- Callers that don't pass `bnd_representation` get `'adapt'` (the new default).
- Callers that explicitly pass a different value are not affected.
- Gated on `mesh_engine == 'gmsh'` because `TriangleOptions` does not accept
  this kwarg and would raise `TypeError`.

The smoke matrix (Config A/B: pure meshdata runs) is the first opportunity to
measure whether this changes the hfun mesh quality metrics (node count, size
field distribution, hfun_min/max).

**OCSMesh-side?** Yes (done on felicio/mpi-fixes, commit 8d98df1).

---

## #16 — Smoke-test matrix: 4-config cost ladder for isolating pipeline stage costs

**Context (session 2025-08-20)**

After the OCSMesh branch changes (#15), a structured smoke-test matrix was
designed to measure the effect of each change in isolation and to characterize
the cost of each pipeline stage. Each config adds exactly one cost class:

```
Config   flow+const   constraints(no topofunc)   contour/channel+boxes   topo_func
-------  -----------  ------------------------   ---------------------   ---------
A         1/tile            -                              -                 -
B         2/tile            -                              -                 -
C         2/tile           yes                             -                 -
D         2/tile           yes                            yes                -
(full)    2/tile           yes                            yes               yes ← excluded (forces serial)
```

**What each comparison tells you:**

- **B vs A:** Does applying 2 fast refinements per tile (instead of 1) change
  the hfun quality or the meshdata dispatch time? Measures pure per-tile
  meshdata overhead with maximum fast-refinement work.

- **C vs B:** Did the OCSMesh #15a change (MPI now uses parallel constraints)
  actually speed up the constraint stage? In C, `_apply_constraints` /
  `_apply_flow_limiters` / `_apply_const_val` use the Pool path under MPI —
  this comparison isolates whether that change is effective.

- **D vs C:** What does global `add_contour` + `add_channel` + box refinements
  add? These still run serially on rank 0 via `_apply_features` (the TODO(mpi)
  in collector.py). This measures the still-un-parallelized serial bottleneck
  and justifies the next OCSMesh development step.

- **Within each config, mpi vs parallel vs serial_mp:** Direct mode comparison
  on identical work. Config A/B (all-fast, no constraints) gives the cleanest
  MPI speedup number because nothing routes through the slow/serial paths.

**Config details:**

| Config | `run_benchmark.py` flags | Description |
|--------|--------------------------|-------------|
| A | `--skip-constraints --skip-box-refinements --light-features` | modulo scheme, 1 fast ref/tile |
| B | `--all-fast-refinements` | flow+const on EVERY tile (new flag) |
| C | `--skip-topofunc --light-features` | adds topo_bound + courant via parallel Pool |
| D | `--skip-topofunc` | adds global contour/channel + box refinements (serial) |

**`--all-fast-refinements` (new flag, added this session):**
Bypasses the index-modulo scheme and applies BOTH `add_subtidal_flow_limiter`
AND `add_constant_value` to EVERY CUDEM tile. Forces all slow stages off
(constraints, contour/channel, boxes) regardless of other skip flags. Both
refinements are pickle-safe (no lambda) so parallel/mpi never fall back to
serial. This is the cleanest possible "maximum fast-path" baseline.

**Submitted via:** `slurm_smoke_matrix.sh` (new script, added this session).

**To run all 4 configs × 3 modes (12 runs, 1 node, 8h budget):**
```bash
sbatch slurm_smoke_matrix.sh
# or a subset:
CONFIGS="A B" MODES="mpi parallel" sbatch slurm_smoke_matrix.sh
```

Results land in `$PROJ/results/smoke_matrix_<jobid>/config_<A|B|C|D>/<mode>/`.
Per-config reports: `config_<X>/report_config_<X>.txt`.

**OCSMesh-side?** The gap between Config C and D (serial `_apply_features` on
rank 0) is the next parallelization target in OCSMesh. Config D vs C
will quantify exactly how much it costs.

---

## Template for new entries

```
## #N — <one-line symptom>

**Symptom**
<paste exact error>

**Root cause**
<why>

**Workaround**
<commands>

**OCSMesh-side?**
<Docs only | UX improvement possible | Code fix needed | No>
```
