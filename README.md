# OCSMesh MPI Benchmark — STOFS-3D-Atlantic

## What this is (read this first)

This repository **benchmarks the MPI parallelization in OCSMesh** against
the older serial and multiprocessing execution modes, using real
STOFS-3D-Atlantic Digital Elevation Models (DEMs) on the NOAA RDHPC
(Hercules).

**Why:** OCSMesh mesh generation can take many hours on large domains. The
MPI implementation distributes part of that work across many CPU cores /
compute nodes. This benchmark answers two questions:

1. **How much faster is MPI** than serial/multiprocessing for the stages it
   parallelizes?
2. **Which stages of the whole "DEM to mesh" pipeline are the real
   bottlenecks** (i.e. what should be parallelized next)?

If you have never seen this project before, read the "Mental model" and
"How the benchmark works" sections below before running anything.

---

## OCSMesh branch history

| PR | Branch | Status | What it does |
|---|---|---|---|
| #250 | felicio/mpi-fixes merged to dev | Merged | MPI bug fixes + performance improvements |
| #251 | anas-ibrahem/feat/parallize_apply_contours | Draft | Parallelizes _apply_contours |

The benchmark test branch **test/pr251** combines both. It is based on
dev (which contains PR #250) with PR #251s two commits on top.

---

## Mental model: how OCSMesh builds a mesh

OCSMesh turns raster elevation data (DEMs) into an unstructured triangular
mesh in three conceptual stages:

    DEMs -> GEOM   (where to mesh: the land/water boundary polygon)
         -> HFUN   (how big each element should be: a size function)
                        |
              GEOM + HFUN -> MeshDriver -> FINAL MESH (triangulated .2dm)

- **Geom** (GeomCollector): extracts the domain boundary from the DEMs.
- **Hfun** (HfunCollector): builds a size function — a field telling the
  mesher how fine/coarse elements should be at each location. Computing it
  (hfun.meshdata()) is the expensive part, and this is the stage the OCSMesh
  MPI implementation parallelizes.
- **MeshDriver**: takes the geom + hfun and runs the meshing engine (Gmsh) to
  produce the final triangulated mesh. This stage is serial / global in
  OCSMesh (not MPI-parallelized).

### The refinements (what the Hfun recipe applies)

build_geom_and_hfun.py is the recipe. Per-CUDEM-tile refinements are assigned
by tile index modulo 6:

| (cudem_pos) % 6 | Refinement | Cost |
|---|---|---|
| 0 | add_subtidal_flow_limiter | cheap |
| 1 | add_constant_value | cheap |
| 2 | add_topo_bound_constraint | expensive (~3h/tile serial) |
| 3 | add_topo_func_constraint | expensive + forces serial fallback |
| 4 | add_courant_num_constraint | expensive (~3h/tile serial) |
| 5 | skipped | — |

Plus global refinements: add_contour (0 m + -200 m) and add_channel.
add_contour is now parallelized by PR #251. add_channel is still serial.
Plus fixed box refinements: add_region_constraint, add_patch, add_feature.

### Recipe parameter: EXPANSION_RATE

EXPANSION_RATE controls the width of the transition zone around contours,
channels, and box refinements. The default was 0.05, which creates ~50 km
transition zones — flooding the entire STOFS domain and producing 25M+ nodes
in Config D. It has been increased to 0.15, giving ~8 km transition zones
and a realistic ~500K-1M node mesh.

---

## The execution modes being compared

run_benchmark.py --modes accepts:

| Mode | OCSMesh execution_mode | nprocs | What runs in parallel |
|---|---|---|---|
| serial_mp | serial | N | Only Pool-based feature steps; meshdata is serial. Slowest baseline. |
| parallel | parallel | N | All per-tile steps + meshdata via multiprocessing.Pool (single node). |
| mpi | mpi | N | Per-tile meshdata dispatched across MPI ranks (can span nodes). |

There is also serial_true = true single core, nprocs=1. Available but not
used in the current benchmark.

All modes run the identical manifest + recipe, so their output meshes must
match — analyze_profile.py includes a numerical-equivalence check.

---

## How the benchmark works

run_benchmark.py for each mode:

1. (full pipeline only) builds the Geom
2. builds the Hfun recipe
3. runs hfun.meshdata() wrapped in cProfile
4. (full pipeline only) runs MeshDriver.run() to produce the final mesh
5. writes outputs:
   - hfun_<mode>.2dm — the size-function field (always)
   - mesh_<mode>.2dm — the final triangulated mesh (full pipeline only)
   - profile_<mode>.prof — cProfile binary
   - benchmark_results.json — timings, per-stage wall times, mesh stats
6. analyze_profile.py merges results into benchmark_report.txt

### Recipe / pipeline flags (control cost)

- --light-features — skip global add_contour + add_channel.
- --skip-topofunc — skip add_topo_func_constraint only.
- --skip-constraints — skip ALL topo/courant constraints.
- --skip-box-refinements — skip add_region_constraint / add_patch / add_feature.
- --all-fast-refinements — apply both fast per-tile refinements to every CUDEM
  tile, bypassing the index-modulo scheme. Forces all slow stages OFF.
- --full-pipeline — run the complete workflow including MeshDriver final mesh.

---

## Files

    ocsmesh_mpi_test/
    |-- download_dems.py            # download GEBCO + CUDEM 1/9" tiles
    |-- trim_manifest.py            # cut a manifest to N CUDEM tiles (+GEBCO)
    |-- build_geom_and_hfun.py      # THE RECIPE: geom + hfun + refinements
    |-- run_benchmark.py            # run modes + cProfile + per-stage timers
    |-- analyze_profile.py          # build the human-readable report
    |-- cpu_affinity.py             # CPU pinning diagnostic (MPI + multiprocessing)
    |
    |-- slurm_single_node.sh        # SMOKE TEST (validate all modes, 1 node, 8h)
    |-- slurm_smoke_matrix.sh       # SMOKE MATRIX (5-config cost ladder x 3 modes)
    |-- slurm_smoke_config_A.sh     # Config A dedicated job
    |-- slurm_smoke_config_B.sh     # Config B dedicated job
    |-- slurm_smoke_config_C.sh     # Config C dedicated job
    |-- slurm_smoke_config_D.sh     # Config D dedicated job
    |-- slurm_cpu_affinity.sh       # CPU affinity diagnostic job
    |-- final_config.sh             # shared config for all FINAL jobs
    |-- slurm_profile_a_serial.sh   # PROFILE A: serial constraint cost
    |-- slurm_final_serial_mp.sh    # PROFILE B: serial_mp baseline
    |-- slurm_final_parallel.sh     # PROFILE B: multiprocessing
    |-- slurm_final_mpi_1node.sh    # PROFILE B: MPI, 1 node
    |-- slurm_final_mpi_multinode.sh# PROFILE B: MPI, multi-node scaling
    |-- slurm_multi_node.sh         # (legacy multi-node MPI script)
    |
    |-- HERCULES_NOTES.md           # running log of HPC gotchas + OCSMesh feedback
    |-- README.md

---

## Requirements (READ — non-obvious)

1. OCSMesh branch test/pr251 — combines PR #250 (merged to dev) and
   PR #251 (parallel contours, Draft). See setup instructions below.
2. Gmsh must be installed in the conda env.
3. mpi4py built from source against the site Intel MPI (not a PyPI wheel).
4. TMPDIR must be forced to /work2 INSIDE each srun — Hercules overrides
   TMPDIR to a small node-local disk. See HERCULES_NOTES #10.
5. DEMs already downloaded? Run download_dems.py --dry-run on the head
   node to generate the manifest without re-downloading files.

---

## Step 1 — Environment (once)

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

    source /work2/noaa/nos-surge/felicioc/envs/miniconda3/etc/profile.d/conda.sh
    conda create -y -n ocsmesh_mpi_test python=3.10
    conda activate ocsmesh_mpi_test

    cd $PROJ
    git clone https://github.com/noaa-ocs-modeling/OCSMesh.git
    cd OCSMesh
    git checkout dev
    git pull origin dev
    git remote add anas https://github.com/anas-ibrahem/OCSMesh.git
    git fetch anas
    git checkout -b test/pr251 anas/feat/parallize_apply_contours

    python -c "from ocsmesh.hfun.collector import _contours_task_worker; print('PR #251 OK')"
    python -c "from ocsmesh.mpi import MPIExecutor; print('PR #250 OK')"
    python -c "from ocsmesh.utils import run_starmap; print('run_starmap OK')"

    pip install -e .
    python -c "import gmsh" 2>&1 || pip install gmsh
    MPICC=$(which mpicc) pip install --no-binary=mpi4py --no-cache-dir mpi4py

    srun --mpi=pmi2 -n 2 python -c \
      "from mpi4py import MPI; c=MPI.COMM_WORLD; print('rank', c.Get_rank(), 'of', c.Get_size())"

    cd $PROJ
    git clone https://github.com/felicio93/ocsmesh_mpi_test.git

---

## Step 2 — Download DEMs

    cd $REPO
    export GEBCO_LOCAL=$DEMS/gebco/gebco_2024_n56.0_s5.0_w-100.0_e-50.0.tif
    python download_dems.py --out-dir $DEMS \
        --manifest $REPO/dem_manifest_smoke.json --only MA_NH_ME

Already have the DEMs? Run in dry-run mode on the head node to generate
the manifest without downloading anything:

    python download_dems.py --out-dir $DEMS \
        --manifest $REPO/dem_manifest_smoke.json --only MA_NH_ME --dry-run

---

## Step 3 — Smoke test (validate everything works)

    cd $REPO
    mkdir -p logs
    sbatch slurm_single_node.sh

    squeue -u $USER
    tail -f logs/bench_1node_*.err

Success = you see [mpi] DONE, [parallel] DONE, [serial_mp] DONE, the
equivalence check passes, and hfun_<mode>.2dm files are written under
$PROJ/results/single_node_<jobid>/.

---

## Step 3b — Smoke-test matrix (5-config cost ladder)

    Config   What runs                                  Compared to previous config
    ------   ----------------------------------------   ---------------------------
    A        1 fast ref/tile (modulo)                   baseline
    B        2 fast refs/tile (flow+const, every tile)  adds per-tile ref density
    C        + constraints (topo_bound + courant)        adds constraint stage via parallel Pool
    D        + global contour/channel + boxes            adds _apply_contours (parallelized by PR #251)
    E        no constraints, no box, contours only       isolates PR #251 contour parallelization

topo_func_constraint is excluded from all configs (its lambda forces serial
fallback, defeating the mode comparison).

    sbatch slurm_smoke_matrix.sh

    CONFIGS="A B C" sbatch slurm_smoke_matrix.sh
    MODES="mpi" sbatch slurm_smoke_config_D.sh
    CONFIGS="E" sbatch slurm_smoke_matrix.sh

Results: $PROJ/results/smoke_matrix_<jobid>/config_<A|B|C|D|E>/<mode>/

---

## Step 4 — Final profiling (after smoke passes)

### Profile A — the serial constraint cost

    sbatch slurm_profile_a_serial.sh

### Profile B — the MPI speedup

    sbatch slurm_final_serial_mp.sh
    sbatch slurm_final_parallel.sh
    sbatch slurm_final_mpi_1node.sh
    sbatch slurm_final_mpi_multinode.sh

    python analyze_profile.py \
        --results-dir $PROJ/results/final_serial_mp_<jobid> \
                      $PROJ/results/final_parallel_<jobid> \
                      $PROJ/results/final_mpi_1node_<jobid> \
                      $PROJ/results/final_mpi_multinode_<jobid> \
        --out combined_report.txt

---

## Reading the outputs

- benchmark_report.txt — timing table, speedups, per-stage wall-clock
  breakdown, cProfile hotspots, and numerical-equivalence check.
- benchmark_results.json — machine-readable timings and mesh stats.
- hfun_<mode>.2dm — the size-function field (open in QGIS).
- mesh_<mode>.2dm — the final triangulated mesh (full pipeline).
- profile_<mode>.prof — cProfile binary. Read with:
    python -c "import pstats; pstats.Stats('profile_mpi.prof').sort_stats('cumulative').print_stats(30)"
    or: snakeviz profile_mpi.prof

---

## Known limitations / findings

- MPI parallelizes hfun.meshdata() only. Geom build and MeshDriver.run()
  are not MPI-parallelized. Amdahls law caps the achievable overall speedup.
- _apply_contours is now parallelized (PR #251). Previously 8,025 s (30%
  of Config D MPI runtime). Config D measures this improvement.
- _apply_channels is still serial — the next follow-up PR after #251.
  Previously 6,689 s (25% of Config D MPI runtime).
- add_topo_func_constraint forces serial constraint application even in
  parallel/mpi. ~3h/tile. Excluded from all smoke-matrix configs.
- MPI now uses Pool-based parallel refinements on rank 0 (PR #250 commit
  8d98df1). Config C vs B measures this gain.
- gmsh boundary defaults to adapt (PR #250 commit 8d98df1), reducing
  sliver triangles.
- EXPANSION_RATE increased to 0.15 — the original 0.05 created 50 km
  transition zones producing 25M+ nodes in Config D.
- method=fast is not MPI-enabled — the benchmark uses exact.
- TMPDIR must be forced to /work2 inside each srun. See HERCULES_NOTES #10.
- PROJ is overridable in slurm_smoke_matrix.sh and slurm_single_node.sh
  via PROJ=/your/path sbatch ...
