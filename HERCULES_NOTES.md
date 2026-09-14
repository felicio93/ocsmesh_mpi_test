# Hercules Gotchas & OCSMesh MPI Feedback Log

Running log of issues hit while testing the OCSMesh MPI implementation on
the NOAA RDHPC (Hercules), with root cause, workaround, and whether it is
something the OCSMesh side can fix/improve.

Share the "OCSMesh-side?" column with the OCSMesh MPI developer.

Environment (as tested):
- Cluster: Hercules (Rocky 9, Slurm)
- Modules: intel-oneapi-compilers/2022.2.1, intel-oneapi-mpi/2021.7.1,
  hdf5/1.12.2, netcdf-c/4.9.0, netcdf-fortran/4.6.0
- Conda env: ocsmesh_mpi_test (Python 3.10)
- OCSMesh: editable install from $PROJ/OCSMesh, branch test/pr251

---

## #1 — pip install -e . does not install mpi4py

**Symptom**
After pip install -e ., import mpi4py fails with
ModuleNotFoundError: No module named mpi4py, even though ocsmesh
imports fine.

**Root cause**
mpi4py is declared as an optional dependency in OCSMesh pyproject.toml.
A plain pip install -e . installs only core deps by design.

**Workaround**
Install mpi4py explicitly (see #2 — do NOT just add .[mpi] on HPC).

**OCSMesh-side?**
Docs only. Worth documenting clearly in the OCSMesh MPI setup guide.

---

## #2 — .[mpi] extra pulls a prebuilt mpi4py wheel linked to the wrong MPI

**Symptom**
pip install -e ".[mpi]" succeeds and import mpi4py works on the login
node, but under srun the MPI layer misbehaves.

**Root cause**
PyPI grabs a prebuilt mpi4py wheel linking a generic MPI (often MPICH),
NOT the site Intel MPI. mpi4py must be compiled against the exact MPI
that srun uses.

**Workaround**

    module load intel-oneapi-mpi/2021.7.1
    which mpicc
    MPICC=$(which mpicc) pip install --no-binary=mpi4py --no-cache-dir mpi4py

**OCSMesh-side?**
Docs mainly. Recommend the MPICC=... --no-binary=mpi4py pattern for HPC.

---

## #3 — PMI2_Job_GetId returned 14 when verifying mpi4py

**Symptom**

    python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
    Abort(1090831) on node 0 (rank 0 in comm 0): Fatal error in PMPI_Init_thread:
    MPIR_pmi_init(167)...: PMI2_Job_GetId returned 14

**Root cause**
Running an MPI program as a bare python -c on a login shell. MPI_Init
tries to contact a PMI server that only exists under srun/mpirun.
This is NOT an OCSMesh bug.

**Workaround**

    srun --mpi=pmi2 -n 2 python -c \
      "from mpi4py import MPI; c=MPI.COMM_WORLD; print(c.Get_rank(), c.Get_size())"

**OCSMesh-side?**
Partially. MPIExecutor could catch a failed MPI_Init and emit a friendly
message instead of a raw abort.

---

## #4 — Confirmed working: srun --mpi=pmi2 on Hercules

**Symptom / status**
Resolution of #3. With mpi4py built from source against Intel MPI (#2),
launches correctly.

**Confirmed Slurm/allocation values:**
- --account=nos-surge
- --partition=hercules
- Node size: 80 cores / 512 GB per node (use --exclusive)
- Windfall queue: no walltime limit but lower priority.

**OCSMesh-side?**
No — site config.

---

## #5 — ETOPO2022 THREDDS WCS returns HTTP 400

**Symptom**

    400 Client Error: 400 for url: https://www.ngdc.noaa.gov/thredds/wcs/...

**Root cause**
The NCEI THREDDS server does NOT expose WCS for ETOPO2022. Served as
per-tile 15 degree NetCDF files only.

**Workaround**
Provide the GEBCO GeoTIFF yourself:
1. Download from https://download.gebco.net/
2. Drop in $DEMS/gebco/ or set GEBCO_LOCAL env var.

Already have DEMs but no manifest? Run dry-run on the head node:

    python download_dems.py --out-dir $DEMS \
        --manifest $REPO/dem_manifest_smoke.json --only MA_NH_ME --dry-run

**OCSMesh-side?**
No — data-sourcing issue in the benchmark helper, now fixed.

---

## #6 — CUDEM tile downloads 404 (hardcoded filenames were stale/guessed)

**Symptom**
Many CUDEM tiles 404 during download.

**Root cause**
download_dems.py originally carried hardcoded tile-name lists that were
partially guessed and did not exist on S3.

**Fix**
Scrape each subfolder index.html at runtime and regex out the real tile
names. No filenames are hardcoded anymore.

**OCSMesh-side?**
No — benchmark data-sourcing only.

---

## #7 — ocsmesh.mpi missing, and MPI init on import inside an allocation

### 7a — ModuleNotFoundError: No module named ocsmesh.mpi

**Symptom**

    ModuleNotFoundError: No module named ocsmesh.mpi

**Root cause**
The MPI implementation lives on the dev branch. A default clone checks
out main/master, which does not contain mpi.py.

**Workaround**

    cd $PROJ/OCSMesh
    git checkout dev
    git pull

**OCSMesh-side?**
Docs / release.

### 7b — PMI2_Job_GetId returned 14 even in non-MPI (parallel) mode

**Symptom**
Inside a salloc allocation, launching with bare python aborts at import
even for --modes parallel.

**Root cause**
import ocsmesh calls _configure_mpi_environment(). Under a Slurm
allocation the MPI env vars are present, so MPI_Init fires on plain
import — but a bare python process has no PMI server.

**Workaround**
Always launch through srun --mpi=pmi2, even for non-MPI modes.

**OCSMesh-side?**
Yes — UX improvement candidate.

---

## #8 — in_item.clip() AttributeError on str/Path DEM inputs (OCSMesh bug)

**Symptom**

    AttributeError: str object has no attribute clip
      File .../ocsmesh/hfun/collector.py, line 1012, in __init__
        in_item.clip(clip_shape)

**Root cause**
HfunCollector.__init__ reassigned in_item = str(in_item) and created a
separate raster object, but the clip branch called in_item.clip() on the
plain string instead of raster.clip().

**Fix**
Fixed in PR #250: call raster.clip(clip_shape).

**OCSMesh-side?** Code fix (done, PR #250).

---

## #9 — MPI failures cascade: mp start method + Pool workers initializing MPI

**Symptom**
Under a SLURM allocation, --modes parallel aborted with PMI_Init returned
14 flooding stderr — one abort per Pool worker.

**Root cause**
1. run_benchmark.py imported from ocsmesh.mpi BEFORE import ocsmesh,
   bypassing _configure_mpi_environment().
2. Forked Pool workers re-imported ocsmesh, triggering MPI_Init in workers
   that have no PMI server.

**Fix**
- run_benchmark.py: import ocsmesh FIRST.
- ocsmesh/mpi.py: set_start_method(spawn, force=True).
- ocsmesh/mpi.py: _get_mpi_comm() returns None for non-MainProcess.
All fixed in PR #250.

**OCSMesh-side?** Code fix (done, PR #250).

---

## #10 — SLURM forces TMPDIR to node-local scratch inside srun — disk full

**Symptom**

    _tiffWriteProc: No space left on device.
    rasterio.errors.RasterioIOError: Write failed.

**Root cause**
Hercules SLURM prolog forces TMPDIR=/local/scratch/$USER/$JOBID inside
every srun. With 80 MPI ranks each writing full-resolution clipped-raster
tif files, the small node-local disk fills up.

**Workaround**
Set TMPDIR INSIDE the rank shell, after the prolog runs:

    srun --mpi=pmi2 --ntasks=80 bash -c "export TMPDIR=${RESULTS_DIR}/mpi_tmp; exec python ..."

All job scripts already handle this.

**OCSMesh-side?** Partially. OCSMesh could expose a way to set its temp
directory explicitly rather than reading TMPDIR once at import.

---

## #11 — Gmsh not installed (meshing engine)

**Symptom**

    ImportError: Gmsh library not installed.
      File .../ocsmesh/engines/gmsh.py, line 31

**Root cause**
gmsh is a separate pip package and was not in the conda env.

**Workaround**
pip install gmsh in the env. Verify: python -c "import gmsh"

**OCSMesh-side?** Docs — Gmsh should be listed as a required runtime dep.

---

## #12 — add_feature crashes on empty channel points (OCSMesh bug)

**Symptom**

    ValueError: data must be of shape (n, m), where there are n points of dimension m
      File .../ocsmesh/hfun/raster.py, line 1229, in add_feature
        tree = cKDTree(np.array(points))

**Root cause**
add_feature built a KDTree from an empty points list without guarding.

**Fix**
Fixed in PR #250: skip the window when len(points) == 0.

**OCSMesh-side?** Code fix (done, PR #250).

---

## #13 — Runtime cost: constraints and contours dominate; MPI covers one stage

**Observation**
- add_topo_bound_constraint, add_topo_func_constraint, and
  add_courant_num_constraint each cost ~3 h/tile in serial.
- add_topo_func_constraint additionally forces _apply_constraints to run
  SERIALLY even in parallel/mpi modes (its callable cannot be pickled).
- Global add_contour + add_channel are an O(tiles x contour-segments) cost.
  add_contour is now parallelized by PR #251.
- The MPI implementation parallelizes hfun.meshdata() per-tile triangulation
  only. MeshDriver.run() is not MPI-parallelized.

**OCSMesh-side?** Next targets: _apply_channels (PR after #251),
MeshDriver.run().

---

## #14 — RegionConstraint._apply_rate is the remaining serial bottleneck

**Observation (measured on job 9600559, MPI mode, 7 tiles)**

Even with --skip-constraints, add_region_constraint and add_patch/add_feature
ran serially on rank 0:

    _apply_features           2295s
    _apply_constraints_serial 1718s
    RegionConstraint.apply    1658s
    _apply_rate                756s  <- expensive: distance expansion
    add_feature                497s  <- KDTree distance calc
    _calculate_write_mpi       659s  <- MPI dispatch

**Root cause**
rate=0.05 on add_region_constraint triggers _apply_rate: a per-window
KDTree distance expansion across the full-resolution raster grid.

**Workaround**
Use --skip-box-refinements for smoke test and Profile B runs.

**OCSMesh-side?** Yes — _apply_features on rank 0 is the next
parallelization target after channels.

---

## #15 — OCSMesh PR #250: mpi now uses parallel refinement path + gmsh adapt

**Context (PR #250, merged to dev 2025-08-31)**

### 15a — MPI now routes through Pool-based parallel refinements

Before PR #250, MPI fell through to serial for _apply_constraints,
_apply_flow_limiters, and _apply_const_val. Fixed by adding mpi to the
dispatch condition:

    if self.execution_mode in (parallel, mpi) and self._nprocs > 1:

Config C vs B in the smoke matrix measures this gain.

### 15b — gmsh engine now defaults boundary representation to adapt

Causes Gmsh to resample tile boundary vertices to match the hfun resolution
before meshing, reducing sliver triangles. Uses setdefault so callers can
override.

**OCSMesh-side?** Done (PR #250).

---

## #16 — Smoke-test matrix: 5-config cost ladder

**Context (sessions 2025-08-20, updated 2026-09-14)**

    Config   flow+const   constraints(no topofunc)   contour/channel+boxes   topo_func
    -------  -----------  ------------------------   ---------------------   ---------
    A         1/tile            -                              -                 -
    B         2/tile            -                              -                 -
    C         2/tile           yes                             -                 -
    D         2/tile           yes                            yes                -
    E         -                -                             contour only        -
    (full)    2/tile           yes                            yes               yes  excluded

What each comparison tells you:
- B vs A: Effect of doubling per-tile refinement work.
- C vs B: Did PR #250 parallel constraint routing help?
- D vs C: What does global contour/channel add? Now addressed by PR #251.
- E: Isolates PR #251 contour parallelization with no other noise.

Config details:

| Config | Flags | Description |
|--------|-------|-------------|
| A | --skip-constraints --skip-box-refinements --light-features | modulo scheme, 1 fast ref/tile |
| B | --all-fast-refinements | flow+const on EVERY tile |
| C | --skip-topofunc --light-features | adds topo_bound + courant via parallel Pool |
| D | --skip-topofunc | adds global contour/channel + box refinements |
| E | --skip-constraints --skip-box-refinements | contours only (isolates PR #251) |

**OCSMesh-side?** Config D vs C quantifies the PR #251 gain.
Config E isolates it cleanly.

---

## #17 — PR #251: _apply_contours parallelized

**Context (PR #251 Draft, 2026-09-14)**

PR #251 by Anas (anas-ibrahem/feat/parallize_apply_contours) parallelizes
_apply_contours in HfunCollector using the same 3-phase file-based worker
pattern as _apply_flow_limiters and _apply_const_val.

### What changed

ocsmesh/hfun/collector.py:
- Added _contours_task_worker(): self-contained worker that rebuilds a
  HfunRaster inside the child process, replays all contour lines, and
  saves the result to a file.
- _apply_contours() now dispatches to serial or parallel based on
  execution_mode and _nprocs.
- _apply_contours_parallel() uses the 3-phase pattern:
  Phase 1: extract contours to feather files on coordinator
  Phase 2: Pool.map(_contours_task_worker, tasks)
  Phase 3: replace _hfun_list entries with worker outputs

ocsmesh/hfun/raster.py and ocsmesh/hfun/mesh.py:
- pool argument made Optional[Pool] on add_feature, add_patch, add_channel
  — allows pool=None (sequential) inside daemon workers.

ocsmesh/utils.py:
- Added run_starmap(pool, func, iterable): sequential if pool=None,
  otherwise pool.starmap().
- Fixed add_pool_args(): nprocs <= 1 passes pool=None instead of spawning
  a Pool — safe inside daemon workers.

### How to test

    cd $PROJ/OCSMesh
    git checkout test/pr251
    python -c "from ocsmesh.hfun.collector import _contours_task_worker; print(OK)"

Run Config D (exercises _apply_contours):

    cd $PROJ/ocsmesh_mpi_test
    MODES="mpi" sbatch slurm_smoke_config_D.sh

Run Config E (isolates contour parallelization cleanly):

    CONFIGS="E" MODES="mpi parallel serial_mp" sbatch slurm_smoke_matrix.sh

### What to measure

In Config D cProfile, previously (PR #250 baseline):

    _apply_contours    8025 s   30% of total
    _apply_channels    6689 s   25% of total

After PR #251, _apply_contours should drop to approximately:

    8025 / min(79, n_contour_tiles) seconds

_apply_channels is unchanged — the next follow-up PR.

### Recipe fix: EXPANSION_RATE = 0.15

Config D previously produced 25M nodes because expansion_rate=0.05 creates
~50 km transition zones, flooding the entire domain. Fixed in
build_geom_and_hfun.py:

    EXPANSION_RATE = 0.15  # was 0.05 — gives ~8 km transition zone

### Known issues

- Pylint format check failing in PR #251 — cosmetic only, no runtime effect.
- _apply_channels still serial — next follow-up PR.
- TopoFuncConstraint still forces serial fallback (unpicklable lambda).

**OCSMesh-side?** Code improvement (PR #251, Draft).

---

## Template for new entries

    ## #N — <one-line symptom>

    **Symptom**
    <paste exact error>

    **Root cause**
    <why>

    **Workaround**
    <commands>

    **OCSMesh-side?**
    <Docs only | UX improvement possible | Code fix needed | No>
