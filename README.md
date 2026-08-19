# OCSMesh MPI Benchmark — STOFS-3D-Atlantic

## What this is (read this first)

This repository **benchmarks the new MPI parallelization in OCSMesh** against
the older serial and multiprocessing execution modes, using real
STOFS-3D-Atlantic Digital Elevation Models (DEMs) on the NOAA RDHPC
(Hercules).

**Why:** OCSMesh mesh generation can take many hours on large domains. A new
MPI implementation distributes part of that work across many CPU cores /
compute nodes. This benchmark answers two questions:

1. **How much faster is MPI** than serial/multiprocessing for the stage it
   parallelizes?
2. **Which stages of the whole "DEM → mesh" pipeline are the real
   bottlenecks** (i.e. what should be parallelized next)?

If you have never seen this project before, read the "Mental model" and
"How the benchmark works" sections below before running anything.

---

## Mental model: how OCSMesh builds a mesh

OCSMesh turns raster elevation data (DEMs) into an unstructured triangular
mesh in three conceptual stages:

```
   DEMs ─┬─► GEOM   (where to mesh: the land/water boundary polygon)
         │
         └─► HFUN   (how big each element should be: a "size function")
                        │
              GEOM + HFUN ─► MeshDriver ─► FINAL MESH (triangulated .2dm)
```

- **Geom** (`GeomCollector`): extracts the domain boundary from the DEMs.
- **Hfun** (`HfunCollector`): builds a *size function* — a field telling the
  mesher how fine/coarse elements should be at each location. It is assembled
  from many "refinements" (see below). Computing it (`hfun.meshdata()`) is the
  expensive part, and **this is the stage the OCSMesh MPI implementation
  parallelizes**.
- **MeshDriver**: takes the geom + hfun and runs the meshing engine (Gmsh) to
  produce the final triangulated mesh. This stage is **serial / global** in
  OCSMesh (not MPI-parallelized) — profiling it tells us if it is a
  bottleneck worth parallelizing next.

### The refinements (what the Hfun recipe applies)

`build_geom_and_hfun.py` is the "recipe" — it decides *what* size-function
refinement is applied *where*. Per-CUDEM-tile refinements are assigned by tile
index modulo 6:

| `(cudem_pos) % 6` | Refinement | Cost |
|---|---|---|
| 0 | `add_subtidal_flow_limiter` | cheap |
| 1 | `add_constant_value` | cheap |
| 2 | `add_topo_bound_constraint` | **expensive (~3h/tile serial)** |
| 3 | `add_topo_func_constraint` | **expensive + forces serial fallback** |
| 4 | `add_courant_num_constraint` | **expensive (~3h/tile serial)** |
| 5 | *skipped* | — |

Plus global refinements applied to all tiles: `add_contour` (0 m + −200 m) and
`add_channel` — these are an **O(tiles × contour-segments) bottleneck**. Plus
fixed box refinements: `add_region_constraint`, `add_patch`, `add_feature`.

**Key finding from this study:** the `topo_*`/`courant` constraints and the
`contour`/`channel` steps dominate runtime and are largely serial. The
recipe-trimming flags below exist to make the benchmark tractable.

---

## The execution modes being compared

`run_benchmark.py --modes` accepts:

| Mode | OCSMesh `execution_mode` | nprocs | What runs in parallel |
|---|---|---|---|
| `serial_mp` | `serial` | N | Only the Pool-based feature steps; per-tile meshdata is serial. This is the "baseline" — the slowest. |
| `parallel` | `parallel` | N | All per-tile steps + meshdata via a local `multiprocessing.Pool` (single node). |
| `mpi` | `mpi` | N | Per-tile meshdata dispatched across MPI ranks (can span nodes). **This is the implementation under test.** |

(There is also `serial_true` = true single core, `nprocs=1`. It is available
but not used in the current benchmark.)

All modes run the **identical manifest + recipe**, so their output meshes must
match — `analyze_profile.py` includes a numerical-equivalence check.

---

## How the benchmark works

`run_benchmark.py` for each mode:

1. (full pipeline only) builds the **Geom**
2. builds the **Hfun** recipe
3. runs `hfun.meshdata()` — **wrapped in cProfile**, because this is the
   MPI-parallelized stage we most want to profile
4. (full pipeline only) runs **`MeshDriver.run()`** to produce the final mesh
5. writes outputs:
   - `hfun_<mode>.2dm` — the size-function field (always)
   - `mesh_<mode>.2dm` — the final triangulated mesh (full pipeline only)
   - `profile_<mode>.prof` — cProfile binary (read with `pstats`/`snakeviz`)
   - `benchmark_results.json` — timings, per-stage wall times, mesh stats
6. `analyze_profile.py` merges the per-mode results into one
   `benchmark_report.txt` with: timing table, speedups, per-stage wall-clock
   breakdown, cProfile hotspots, and the equivalence check.

### Recipe / pipeline flags (control cost)

Because the full realistic recipe on many tiles takes far longer than an 8h
SLURM job, these flags trim it:

- `--light-features` — skip global `add_contour` + `add_channel`
  (the O(tiles×segments) bottleneck).
- `--skip-topofunc` — skip `add_topo_func_constraint` only. (That constraint
  stores a callable that forces OCSMesh's constraint stage to run **serially
  even in parallel/mpi**, so skipping it lets constraints parallelize.)
- `--skip-constraints` — skip ALL of `topo_bound` + `topo_func` + `courant`.
  Leaves only the two cheap per-tile refinements. Used for the smoke test.
- `--full-pipeline` — run the COMPLETE workflow (geom + hfun + MeshDriver
  final mesh) and record per-stage wall times. Off = hfun-only (faster).

In the SLURM scripts these map to env toggles: `LIGHT_FEATURES`,
`SKIP_TOPOFUNC`, `SKIP_CONSTRAINTS`, `FULL_PIPELINE`.

---

## Files

```
ocsmesh_mpi_test/
├── download_dems.py            # download GEBCO + CUDEM 1/9" tiles
├── trim_manifest.py            # cut a manifest to N CUDEM tiles (+GEBCO)
├── build_geom_and_hfun.py      # THE RECIPE: geom + hfun + refinements
├── run_benchmark.py            # run modes + cProfile + per-stage timers
├── analyze_profile.py          # build the human-readable report
│
├── slurm_single_node.sh        # SMOKE TEST (validate all modes, 1 node, 8h)
├── final_config.sh             # shared config for all FINAL jobs
├── slurm_profile_a_serial.sh   # PROFILE A: serial constraint cost (full recipe)
├── slurm_final_serial_mp.sh    # PROFILE B: serial_mp baseline (meshdata)
├── slurm_final_parallel.sh     # PROFILE B: multiprocessing
├── slurm_final_mpi_1node.sh    # PROFILE B: MPI, 1 node
├── slurm_final_mpi_multinode.sh# PROFILE B: MPI, multi-node scaling
├── slurm_multi_node.sh         # (legacy multi-node MPI script)
│
├── HERCULES_NOTES.md           # running log of HPC gotchas + OCSMesh feedback
└── README.md
```

---

## Requirements (READ — non-obvious)

1. **OCSMesh branch `felicio/mpi-fixes`** (off `dev`). The MPI code and the
   bug fixes this benchmark depends on live there. A plain `main`/`dev` clone
   will NOT work.
2. **Gmsh must be installed** in the conda env, or the meshing stage aborts
   with `ImportError: Gmsh library not installed`.
3. **mpi4py built from source** against the site Intel MPI (not a PyPI wheel).
4. **TMPDIR must be forced to /work2 INSIDE each srun** — Hercules overrides
   `TMPDIR` to a small node-local disk for every srun, which fills up when 80
   ranks write raster temp files. See HERCULES_NOTES #10. The job scripts
   already handle this.

---

## Step 1 — Environment (once)

```bash
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

# Source conda by its hardcoded path (conda is not on PATH in SLURM jobs).
source /work2/noaa/nos-surge/felicioc/envs/miniconda3/etc/profile.d/conda.sh
conda create -y -n ocsmesh_mpi_test python=3.10
conda activate ocsmesh_mpi_test

# OCSMesh: clone and check out the required branch.
cd $PROJ
git clone https://github.com/noaa-ocs-modeling/OCSMesh.git
cd OCSMesh
git checkout felicio/mpi-fixes      # REQUIRED
pip install -e .

# Gmsh meshing engine (REQUIRED).
python -c "import gmsh" 2>&1 || pip install gmsh

# mpi4py FROM SOURCE against Intel MPI.
MPICC=$(which mpicc) pip install --no-binary=mpi4py --no-cache-dir mpi4py

# Verify (must run under srun, not bare python — see HERCULES_NOTES #3).
srun --mpi=pmi2 -n 2 python -c \
  "from mpi4py import MPI; c=MPI.COMM_WORLD; print('rank', c.Get_rank(), 'of', c.Get_size())"

cd $PROJ
git clone https://github.com/felicio93/ocsmesh_mpi_test.git
```

## Step 2 — Download DEMs

```bash
cd $REPO
export GEBCO_LOCAL=$DEMS/gebco/gebco_2024_n56.0_s5.0_w-100.0_e-50.0.tif
# (place the GEBCO GeoTIFF in $DEMS/gebco/ — see HERCULES_NOTES #5)
python download_dems.py --out-dir $DEMS \
    --manifest $REPO/dem_manifest_smoke.json --only MA_NH_ME
```

## Step 3 — Smoke test (validate everything works, ~fits 8h)

The smoke test runs all modes on a tiny 7-tile workload with the slow
refinements skipped, in the order **MPI → parallel → serial_mp** (so the most
important result arrives first). It trims the manifest automatically.

```bash
cd $REPO
mkdir -p logs
sbatch slurm_single_node.sh          # SKIP_CONSTRAINTS=1, LIGHT_FEATURES=1 by default

squeue -u $USER
tail -f logs/bench_1node_*.err       # OCSMesh progress prints to stderr
```

Success = you see `[mpi] DONE`, `[parallel] DONE`, `[serial_mp] DONE`, the
equivalence check passes, and `hfun_<mode>.2dm` files are written under
`$PROJ/results/single_node_<jobid>/`.

## Step 4 — Final profiling (after smoke passes)

Two profiles, because MPI only accelerates ONE stage:

### Profile A — the serial constraint cost (why serial is slow)
Full recipe, few tiles, serial_mp only, end-to-end. Shows the ~3h/tile
constraint cost and the un-parallelized `_apply_features` / MeshDriver stages.

```bash
sbatch slurm_profile_a_serial.sh
```

### Profile B — the MPI speedup (what MPI accelerates)
`--light-features --skip-constraints`, ~18 tiles, all modes. Isolates the
per-tile meshdata stage where MPI shines. Run one job per mode; they all
`source final_config.sh` so they share the identical workload.

```bash
sbatch slurm_final_serial_mp.sh
sbatch slurm_final_parallel.sh
sbatch slurm_final_mpi_1node.sh
sbatch slurm_final_mpi_multinode.sh   # edit --nodes for the scaling study
```

Then combine into one report:

```bash
python analyze_profile.py \
    --results-dir $PROJ/results/final_serial_mp_<jobid> \
                  $PROJ/results/final_parallel_<jobid> \
                  $PROJ/results/final_mpi_1node_<jobid> \
                  $PROJ/results/final_mpi_multinode_<jobid> \
    --out combined_report.txt
```

---

## Reading the outputs

- **`benchmark_report.txt`** — the human-readable summary: timing table,
  speedups vs baseline, **per-stage wall-clock breakdown** (geom / hfun build /
  hfun meshdata / MeshDriver), cProfile hotspots, and numerical-equivalence
  check.
- **`benchmark_results.json`** — machine-readable: per-mode wall time,
  `stage_times_s`, node/triangle counts, hfun stats.
- **`hfun_<mode>.2dm`** — the size-function field (open in QGIS).
- **`mesh_<mode>.2dm`** — the final triangulated mesh (full pipeline; open in
  QGIS). All modes should produce matching meshes.
- **`profile_<mode>.prof`** — cProfile binary. Read with:
  ```bash
  python -c "import pstats; pstats.Stats('profile_mpi.prof').sort_stats('cumulative').print_stats(30)"
  # or: snakeviz profile_mpi.prof
  ```

---

## Known limitations / findings

- **MPI parallelizes only `hfun.meshdata()`** (the per-tile size-function
  computation). Geom build, `_apply_features` (contours/channels/constraints),
  and `MeshDriver.run()` are **not** MPI-parallelized — they run on rank 0 /
  serially. Amdahl's law therefore caps the achievable overall speedup; the
  per-stage breakdown quantifies this.
- **`add_topo_func_constraint` forces serial constraint application** even in
  parallel/mpi (its callable can't be pickled for the Pool). ~3h/tile.
- **The topo/courant constraints and contour/channel steps dominate** the
  `exact` method's cost. These are the next parallelization targets.
- **`method='fast'` is not MPI-enabled** — the benchmark uses `exact`.
- **TMPDIR must be forced to /work2 inside each srun** (Hercules node-local
  scratch is small and fills up). See HERCULES_NOTES #10.
