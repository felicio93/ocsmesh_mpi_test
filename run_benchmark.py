"""STOFS-3D-Atlantic OCSMesh benchmark: serial, parallel, and MPI modes.

This script is the heart of the Hercules benchmark. It:

1.  Loads the DEM manifest produced by ``download_dems.py``.
2.  Builds an ``HfunCollector`` that covers the full STOFS-3D-Atlantic
    domain with one representative mesh refinement *per geographic region*
    (one refinement type per bounding box).
3.  Runs four back-to-back executions:
        a. serial_true   (execution_mode='serial',   nprocs=1)
                         True single-core baseline. All Pool calls use
                         1 worker, so nothing runs in parallel.
        b. serial_mp     (execution_mode='serial',   nprocs=<N>)
                         OCSMesh serial mode but Pool-based steps
                         (contours, channels, patches, linefeatures)
                         use N workers. Meshdata dispatch is single-
                         threaded. Isolates the effect of those Pool
                         steps vs. serial_true.
        c. parallel      (execution_mode='parallel', nprocs=<N>)
                         Full multiprocessing mode. All per-tile steps
                         and meshdata dispatch use N workers.
        d. mpi           (execution_mode='mpi')
                         MPI mode via MPIExecutor. Must be launched
                         under srun/mpiexec with N+1 ranks total
                         (1 manager + N workers).
    and times each with both wall-clock and cProfile.
4.  Saves per-mode timing, stats, .2dm mesh files, and cProfile `.prof`
    files alongside a JSON summary (``benchmark_results.json``).

Mode → OCSMesh mapping
-----------------------
    Benchmark mode   execution_mode   nprocs passed to Hfun
    serial_true      'serial'         1
    serial_mp        'serial'         <N>
    parallel         'parallel'       <N>
    mpi              'mpi'            <N>   (N = MPI size - 1)

MPI awareness
-------------
Under ``mpiexec`` / ``srun`` only Rank 0 runs serial_true, serial_mp,
and parallel benchmarks. All ranks participate in the MPI benchmark.
Worker ranks call ``hfun.meshdata()`` and return ``None``; only Rank 0
writes output.

Usage (interactive / pre-test)
-------------------------------
    # True serial only (no mpiexec needed):
    python run_benchmark.py --manifest dem_manifest.json \\
                            --shapefile /path/to/stofs_domain.shp \\
                            --out-dir ./results \\
                            --nprocs 40 \\
                            --modes serial_true serial_mp parallel

    # Full four-mode run under MPI:
    mpiexec -n 41 python run_benchmark.py \\
        --manifest dem_manifest.json \\
        --shapefile /path/to/stofs_domain.shp \\
        --out-dir ./results \\
        --nprocs 40 \\
        --modes serial_true serial_mp parallel mpi

Profile output files (Rank 0 only):
    <out-dir>/profile_serial_true.prof
    <out-dir>/profile_serial_mp.prof
    <out-dir>/profile_parallel.prof
    <out-dir>/profile_mpi.prof
    <out-dir>/hfun_serial_true.2dm
    <out-dir>/hfun_serial_mp.2dm
    <out-dir>/hfun_parallel.2dm
    <out-dir>/hfun_mpi.2dm
    <out-dir>/benchmark_results.json
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import logging
import os
import pstats
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# MPI bootstrap (must precede any ocsmesh import)
# ---------------------------------------------------------------------------
# ocsmesh.__init__ calls _configure_mpi_environment() at import time, which
# pins thread pools and sets multiprocessing start method to 'spawn'.
# IMPORTANT: import ocsmesh FIRST so __init__ runs _configure_mpi_environment()
# before any Pool() is created. Importing ocsmesh.mpi directly first would
# bypass __init__ and leave multiprocessing start method as 'fork', causing
# PMI_Init aborts in Pool worker processes under a SLURM allocation.

import ocsmesh  # noqa: F401 — triggers _configure_mpi_environment() in __init__
from ocsmesh.mpi import (
    MPIExecutor,
    _get_mpi,
    _is_mpi_active,
    _is_mpi_env_detected,
)

# Determine rank early — before any heavy imports so log output is clean.
_MPI_ACTIVE = _is_mpi_env_detected() and _is_mpi_active()
_RANK = 0
_SIZE = 1
if _MPI_ACTIVE:
    try:
        from mpi4py import MPI as _MPI
        _RANK = _MPI.COMM_WORLD.Get_rank()
        _SIZE = _MPI.COMM_WORLD.Get_size()
    except ImportError:
        _MPI_ACTIVE = False

_IS_MANAGER = (_RANK == 0)

# ---------------------------------------------------------------------------
# Logging — only rank 0 prints INFO; workers print WARNING+ to stderr.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO if _IS_MANAGER else logging.WARNING,
    format=f"[Rank {_RANK}] %(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
_logger = logging.getLogger("stofs_benchmark")

from ocsmesh import Geom, Hfun, Mesh, MeshDriver, Raster
from shapely.geometry import box, MultiPolygon, Polygon
import geopandas as gpd

# The Geom/Hfun recipe (raster ordering, index-modulo refinements,
# global contour/channel, box-based region/patch/feature) lives in
# build_geom_and_hfun.py so it can be inspected/edited independently.
import build_geom_and_hfun as recipe
from build_geom_and_hfun import GLOBAL_HMIN, GLOBAL_HMAX

# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------
# Maps benchmark mode name -> (ocsmesh execution_mode, nprocs_override)
# nprocs_override=None means "use args.nprocs as-is"
# nprocs_override=1    means "force nprocs=1 regardless of args.nprocs"

_ALL_MODES = ["serial_true", "serial_mp", "parallel", "mpi"]

_MODE_CONFIG = {
    #  benchmark_mode : (ocsmesh_execution_mode, nprocs_override)
    "serial_true": ("serial",   1),     # true single-core: nprocs forced to 1
    "serial_mp":   ("serial",   None),  # serial mode, Pool steps use N workers
    "parallel":    ("parallel", None),  # full multiprocessing
    "mpi":         ("mpi",      None),  # MPI via MPIExecutor
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest(manifest_path: Path) -> Dict:
    with open(manifest_path) as fh:
        return json.load(fh)


def _load_domain_shape(shapefile: Optional[Path]):
    """Return a Shapely geometry (Polygon or MultiPolygon) for the domain.

    If *shapefile* is None or missing, a broad default bounding box covering
    the STOFS-3D-Atlantic extent is used so the script can run without the
    shapefile (useful for local testing).
    """
    if shapefile and Path(shapefile).exists():
        _logger.info(f"Loading domain shapefile: {shapefile}")
        gdf = gpd.read_file(shapefile)
        if gdf.crs and not gdf.crs.equals("EPSG:4326"):
            gdf = gdf.to_crs("EPSG:4326")
        geom = gdf.union_all()
        _logger.info(f"Domain geometry type: {geom.geom_type}")
        return geom
    else:
        _logger.warning(
            "No shapefile provided (or file not found). "
            "Using default STOFS-3D-Atlantic bounding box."
        )
        return box(-100.0, 7.0, -50.0, 47.0)


def _build_geom(manifest, domain_shape, nprocs):
    """Build the GeomCollector (domain boundary) from the DEM manifest.

    Only needed for the full end-to-end pipeline (MeshDriver.run()). The geom
    defines WHERE to mesh (the land/water boundary); the hfun defines the
    element SIZES. MeshDriver combines them to triangulate the final mesh.
    """
    raster_paths, _ = recipe.load_ordered_rasters(manifest)
    if not raster_paths:
        raise RuntimeError("No DEM files found for geom build.")
    return recipe.build_geom(raster_paths, domain_shape, nprocs)


def _build_hfun(
    manifest: Dict,
    domain_shape,
    nprocs: int,
    execution_mode: str,
    light_features: bool = False,
    skip_topofunc: bool = False,
    skip_constraints: bool = False,
) -> Hfun:
    """Construct the HfunCollector with all DEMs and all refinements."""
    raster_paths, raster_metas = recipe.load_ordered_rasters(manifest)
    if not raster_paths:
        raise RuntimeError(
            "No DEM files found. Run download_dems.py first, or check "
            "--manifest path."
        )
    return recipe.build_hfun(
        raster_paths,
        raster_metas,
        domain_shape,
        nprocs,
        execution_mode,
        light_features=light_features,
        skip_topofunc=skip_topofunc,
        skip_constraints=skip_constraints,
    )


# ---------------------------------------------------------------------------
# Profiled runner
# ---------------------------------------------------------------------------

def _run_mode(
    manifest: Dict,
    domain_shape,
    nprocs: int,
    mode: str,
    out_dir: Path,
    light_features: bool = False,
    skip_topofunc: bool = False,
    skip_constraints: bool = False,
    full_pipeline: bool = False,
) -> Dict:
    """Run meshdata() for one benchmark mode.

    Parameters
    ----------
    nprocs : int
        Number of parallel workers as requested by the user (args.nprocs).
        For 'serial_true' this is overridden to 1 internally.
    mode : str
        One of serial_true, serial_mp, parallel, mpi.

    Returns a result dict with timing stats and output mesh stats.
    Only meaningful on Rank 0; worker ranks return an empty dict.
    """
    log = _logger
    result: Dict = {"mode": mode, "status": "pending"}

    ocsmesh_mode, nprocs_override = _MODE_CONFIG[mode]
    effective_nprocs = nprocs_override if nprocs_override is not None else nprocs

    prof = cProfile.Profile()
    t0 = time.perf_counter()
    # Per-stage wall-clock timers. These directly answer benchmark goal #2:
    # which pipeline stage dominates and therefore deserves parallelization.
    stage_times = {}

    try:
        log.info(f"{'='*60}")
        log.info(f"Starting mode: {mode.upper()}")
        log.info(
            f"  ocsmesh execution_mode={ocsmesh_mode!r}  "
            f"nprocs={effective_nprocs}  full_pipeline={full_pipeline}"
        )
        log.info(f"{'='*60}")

        # ── Stage 1: (optional) build the Geom (domain boundary) ─────
        # Only needed for the full end-to-end pipeline. The geom is the
        # land/water boundary the final mesh is triangulated inside.
        geom = None
        if full_pipeline:
            t_geom = time.perf_counter()
            geom = _build_geom(manifest, domain_shape, effective_nprocs)
            stage_times["geom_build_s"] = round(time.perf_counter() - t_geom, 3)

        # ── Stage 2: build the Hfun (size-function recipe) ───────────
        t_hfun_build = time.perf_counter()
        hfun = _build_hfun(
            manifest, domain_shape, effective_nprocs, ocsmesh_mode,
            light_features=light_features,
            skip_topofunc=skip_topofunc,
            skip_constraints=skip_constraints,
        )
        stage_times["hfun_build_s"] = round(time.perf_counter() - t_hfun_build, 3)

        # ── Stage 3: hfun.meshdata() — THE MPI-PARALLELIZED STAGE ────
        # This is where OCSMesh's MPIExecutor distributes per-tile work.
        # cProfile wraps only this call so the .prof isolates the stage
        # the MPI implementation actually accelerates.
        t_meshdata = time.perf_counter()
        prof.enable()
        meshdata = hfun.meshdata()
        prof.disable()
        stage_times["hfun_meshdata_s"] = round(time.perf_counter() - t_meshdata, 3)

        # MPI worker ranks return None from meshdata(); only rank 0 continues.
        if meshdata is None:
            return {}  # worker rank — caller handles

        # ── Stats on the hfun size-function field ────────────────────
        n_nodes = len(meshdata.coords)
        n_tria = len(meshdata.tria) if meshdata.tria is not None else 0
        vals = meshdata.values

        # ── Stage 4: (optional) MeshDriver.run() — FINAL MESH ────────
        # Global triangulation of the whole domain using geom + hfun.
        # This is NOT MPI-parallelized in OCSMesh — it is a single serial
        # engine call. Profiling it shows whether it is a bottleneck that
        # would deserve parallelization next.
        final_mesh = None
        if full_pipeline and geom is not None:
            t_driver = time.perf_counter()
            log.info(f"[{mode}] Running MeshDriver (final mesh generation)...")
            driver = MeshDriver(
                geom, hfun, engine_name="gmsh",
                bnd_representation="exact",
            )
            final_mesh = driver.run()
            stage_times["meshdriver_run_s"] = round(
                time.perf_counter() - t_driver, 3)

        wall_time = time.perf_counter() - t0

        result = {
            "mode": mode,
            "ocsmesh_execution_mode": ocsmesh_mode,
            "effective_nprocs": effective_nprocs,
            "full_pipeline": full_pipeline,
            "status": "success",
            "wall_time_s": round(wall_time, 3),
            "n_nodes": n_nodes,
            "n_triangles": n_tria,
            "hfun_min": float(np.min(vals)),
            "hfun_max": float(np.max(vals)),
            "hfun_mean": float(np.mean(vals)),
            "hfun_std": float(np.std(vals)),
            "stage_times_s": stage_times,
        }
        log.info(
            f"[{mode}] DONE in {wall_time:.1f}s  "
            f"nodes={n_nodes:,}  tria={n_tria:,}  "
            f"hfun=[{result['hfun_min']:.0f}, {result['hfun_max']:.0f}]"
        )
        log.info(f"[{mode}] stage times (s): {stage_times}")

        # ── Save cProfile (isolates the MPI-parallelized meshdata stage) ─
        prof_path = out_dir / f"profile_{mode}.prof"
        prof.dump_stats(str(prof_path))
        log.info(f"cProfile saved to {prof_path}")

        # ── Save the HFUN size-function field as .2dm (always) ───────
        hfun_path = out_dir / f"hfun_{mode}.2dm"
        try:
            Mesh(meshdata).write(str(hfun_path), overwrite=True, format='2dm')
            log.info(f"Hfun size-field saved to {hfun_path}")
        except Exception as mesh_exc:  # pylint: disable=broad-exception-caught
            log.warning(f"Could not save hfun .2dm: {mesh_exc}")

        # ── Save the FINAL MESH as .2dm (full pipeline only) ─────────
        if final_mesh is not None:
            final_path = out_dir / f"mesh_{mode}.2dm"
            try:
                final_mesh.write(str(final_path), format="2dm", overwrite=True)
                log.info(f"Final mesh saved to {final_path}")
            except Exception as mesh_exc:  # pylint: disable=broad-exception-caught
                log.warning(f"Could not save final mesh .2dm: {mesh_exc}")

        # Also print top-20 cumulative to stdout
        sio = io.StringIO()
        ps = pstats.Stats(prof, stream=sio)
        ps.sort_stats("cumulative")
        ps.print_stats(20)
        log.info(f"\ncProfile top-20 ({mode}):\n{sio.getvalue()}")

    except Exception as exc:  # pylint: disable=broad-exception-caught
        wall_time = time.perf_counter() - t0
        tb = traceback.format_exc()
        log.error(f"[{mode}] FAILED after {wall_time:.1f}s: {exc}\n{tb}")
        result = {
            "mode": mode,
            "ocsmesh_execution_mode": ocsmesh_mode,
            "effective_nprocs": effective_nprocs,
            "full_pipeline": full_pipeline,
            "status": "failed",
            "wall_time_s": round(wall_time, 3),
            "stage_times_s": stage_times,
            "error": repr(exc),
            "traceback": tb,
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCSMesh STOFS-3D-Atlantic benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes
-----
  serial_true   True single-core baseline (nprocs=1 forced).
                All Pool-based steps use exactly 1 worker.
  serial_mp     OCSMesh serial mode with multiprocessing pools.
                Pool-based steps (contours, channels, patches,
                linefeatures) use --nprocs workers; meshdata
                dispatch is single-threaded.
  parallel      Full multiprocessing mode (execution_mode=parallel).
                All per-tile steps and meshdata use --nprocs workers.
  mpi           MPI mode via MPIExecutor.
                Must be launched under srun/mpiexec.
        """,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent / "dem_manifest.json",
        help="DEM manifest JSON produced by download_dems.py",
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=None,
        help="STOFS-3D-Atlantic domain shapefile (optional; bbox fallback used if absent)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./benchmark_results"),
        help="Directory for results, profiles, logs, and .2dm meshes",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=max(os.cpu_count() or 1, 1),
        help=(
            "Number of workers for serial_mp, parallel, and mpi modes. "
            "Ignored for serial_true (always 1). Default: all CPUs."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=_ALL_MODES,
        default=_ALL_MODES,
        help=f"Benchmark modes to run (default: all four). Choices: {_ALL_MODES}",
    )
    parser.add_argument(
        "--light-features",
        action="store_true",
        help=(
            "Skip global add_contour / add_channel refinements. These are the "
            "O(tiles x segments) bottleneck of the exact method. Use for fast "
            "MPI-path debugging; the per-tile and box refinements still run."
        ),
    )
    parser.add_argument(
        "--skip-topofunc",
        action="store_true",
        help=(
            "Skip add_topo_func_constraint. That constraint forces OCSMesh's "
            "_apply_constraints to run SERIALLY even in parallel/mpi modes "
            "(~44 min/tile). Skipping it lets the constraint stage parallelize "
            "and removes the dominant serial bottleneck."
        ),
    )
    parser.add_argument(
        "--skip-constraints",
        action="store_true",
        help=(
            "Skip ALL topo/courant constraints (topo_bound, topo_func, courant). "
            "Leaves only fast flow_limiter + const_value refinements. Use for "
            "the smoke test so serial_mp fits in 8h while still exercising "
            "the full Gmsh meshdata pipeline end-to-end."
        ),
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help=(
            "Run the COMPLETE end-to-end workflow: build Geom, build Hfun, "
            "hfun.meshdata() (MPI-parallelized), then MeshDriver.run() to "
            "triangulate the FINAL mesh, and write both hfun_<mode>.2dm and "
            "mesh_<mode>.2dm. Records per-stage wall times (geom / hfun build / "
            "hfun meshdata / MeshDriver run). Without this flag only the hfun "
            "stage runs (faster; used for the smoke test)."
        ),
    )
    parser.add_argument(
        "--hmin",
        type=float,
        default=GLOBAL_HMIN,
        help=f"Global minimum mesh size in metres (default: {GLOBAL_HMIN})",
    )
    parser.add_argument(
        "--hmax",
        type=float,
        default=GLOBAL_HMAX,
        help=f"Global maximum mesh size in metres (default: {GLOBAL_HMAX})",
    )
    args = parser.parse_args()

    recipe.GLOBAL_HMIN = args.hmin
    recipe.GLOBAL_HMAX = args.hmax

    # Only rank 0 does setup I/O; workers skip straight to meshdata() calls.
    if _IS_MANAGER:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Output directory : {args.out_dir.resolve()}")
        _logger.info(f"Manifest         : {args.manifest.resolve()}")
        _logger.info(f"Shapefile        : {args.shapefile}")
        _logger.info(f"nprocs           : {args.nprocs}  (serial_true always uses 1)")
        _logger.info(f"Modes            : {args.modes}")
        _logger.info(f"Light features   : {args.light_features}  (skip contour/channel if True)")
        _logger.info(f"Skip topofunc    : {args.skip_topofunc}  (skip topo_func_constraint if True)")
        _logger.info(f"Skip constraints : {args.skip_constraints}  (skip all topo/courant constraints if True)")
        _logger.info(f"Full pipeline    : {args.full_pipeline}  (geom + hfun + MeshDriver final mesh if True)")
        _logger.info(f"MPI active       : {_MPI_ACTIVE}  (size={_SIZE})")
        _logger.info(f"hmin={recipe.GLOBAL_HMIN} m  hmax={recipe.GLOBAL_HMAX} m")

    # ── Load inputs ──────────────────────────────────────────────────
    if _IS_MANAGER:
        try:
            manifest = _load_manifest(args.manifest)
        except FileNotFoundError:
            _logger.error(
                f"Manifest not found: {args.manifest}. "
                "Run download_dems.py first."
            )
            sys.exit(1)

        domain_shape = _load_domain_shape(args.shapefile)
    else:
        manifest = None
        domain_shape = None

    # Broadcast to workers so all ranks have identical inputs
    if _MPI_ACTIVE:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        manifest = comm.bcast(manifest, root=0)
        domain_shape = comm.bcast(domain_shape, root=0)

    # ── Run each mode ────────────────────────────────────────────────
    all_results: List[Dict] = []

    for mode in args.modes:
        if mode == "mpi" and not _MPI_ACTIVE:
            if _IS_MANAGER:
                _logger.warning(
                    "Skipping MPI mode — not running under mpiexec/srun. "
                    "Re-run with: srun --mpi=pmi2 -n <N+1> python run_benchmark.py --modes mpi"
                )
            continue

        # Non-MPI modes: only rank 0 runs them; workers have nothing to do.
        if mode in ("serial_true", "serial_mp", "parallel") and not _IS_MANAGER:
            continue

        result = _run_mode(
            manifest,
            domain_shape,
            args.nprocs,
            mode,
            args.out_dir,
            light_features=args.light_features,
            skip_topofunc=args.skip_topofunc,
            skip_constraints=args.skip_constraints,
            full_pipeline=args.full_pipeline,
        )

        # Worker ranks return empty dict from _run_mode when meshdata()=None
        if result:
            all_results.append(result)

    # ── Write summary JSON (rank 0 only) ─────────────────────────────
    if _IS_MANAGER and all_results:
        # Speedup relative to serial_true baseline (if present), else serial_mp
        baseline_time = next(
            (r["wall_time_s"] for r in all_results
             if r["mode"] == "serial_true" and r["status"] == "success"),
            None,
        )
        if baseline_time is None:
            baseline_time = next(
                (r["wall_time_s"] for r in all_results
                 if r["mode"] == "serial_mp" and r["status"] == "success"),
                None,
            )
        for r in all_results:
            if baseline_time and r.get("status") == "success":
                r["speedup_vs_baseline"] = round(baseline_time / r["wall_time_s"], 3)

        summary = {
            "hostname": os.uname().nodename,
            "mpi_size": _SIZE,
            "nprocs_requested": args.nprocs,
            "hmin": recipe.GLOBAL_HMIN,
            "hmax": recipe.GLOBAL_HMAX,
            "n_dems": sum(
                1 for v in manifest.values() if v.get("available")
            ),
            "results": all_results,
        }
        out_json = args.out_dir / "benchmark_results.json"
        out_json.write_text(json.dumps(summary, indent=2))
        _logger.info(f"\nResults written to {out_json}")

        # ── Print summary table ──────────────────────────────────────
        _logger.info("\n" + "=" * 72)
        _logger.info("  BENCHMARK SUMMARY")
        _logger.info("=" * 72)
        _logger.info(
            f"  {'Mode':<14} {'nprocs':>6}  {'Status':<10} "
            f"{'Time (s)':>10}  {'Speedup':>9}  {'Nodes':>10}"
        )
        _logger.info(
            f"  {'-'*14} {'-'*6}  {'-'*10} "
            f"{'-'*10}  {'-'*9}  {'-'*10}"
        )
        for r in all_results:
            sp = f"{r.get('speedup_vs_baseline', 1.0):.2f}x"
            nd = f"{r.get('n_nodes', 0):,}"
            np_ = r.get('effective_nprocs', '?')
            _logger.info(
                f"  {r['mode']:<14} {np_:>6}  {r['status']:<10} "
                f"{r.get('wall_time_s', 0):>10.2f}  {sp:>9}  {nd:>10}"
            )
        _logger.info("=" * 72)


if __name__ == "__main__":
    main()
