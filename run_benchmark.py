"""STOFS-3D-Atlantic OCSMesh benchmark: serial, parallel, and MPI modes.

This script is the heart of the Hercules benchmark. It:

1.  Loads the DEM manifest produced by ``download_dems.py``.
2.  Builds an ``HfunCollector`` that covers the full STOFS-3D-Atlantic
    domain with one representative mesh refinement *per geographic region*
    (one refinement type per bounding box).
3.  Runs three back-to-back executions:
        a. serial          (execution_mode='serial')
        b. parallel        (execution_mode='parallel', nprocs=<N>)
        c. MPI             (execution_mode='mpi')
    and times each with both wall-clock and cProfile.
4.  Saves per-mode timing, stats, and cProfile `.prof` files alongside a
    JSON summary (``benchmark_results.json``).

MPI awareness
-------------
Under ``mpiexec`` / ``srun`` only Rank 0 runs the serial and parallel
benchmarks. All ranks participate in the MPI benchmark. Worker ranks
call ``hfun.meshdata()`` and return ``None``; only Rank 0 writes output.

Usage (interactive / pre-test)
-------------------------------
    # Serial + parallel only (no mpiexec needed):
    python run_benchmark.py --manifest dem_manifest.json \\
                            --shapefile /path/to/stofs_domain.shp \\
                            --out-dir ./results \\
                            --nprocs 40 \\
                            --modes serial parallel

    # Full three-mode run under MPI:
    mpiexec -n 41 python run_benchmark.py \\
        --manifest dem_manifest.json \\
        --shapefile /path/to/stofs_domain.shp \\
        --out-dir ./results \\
        --nprocs 40 \\
        --modes serial parallel mpi

Profile output files (Rank 0 only):
    <out-dir>/profile_serial.prof
    <out-dir>/profile_parallel.prof
    <out-dir>/profile_mpi.prof
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
# pins thread pools and sets multiprocessing start method.  We just need to
# be careful not to import MPI ourselves before ocsmesh does.

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

# Now safe to import ocsmesh
from ocsmesh import Hfun, Raster
from shapely.geometry import box, MultiPolygon, Polygon
import geopandas as gpd

# The Geom/Hfun recipe (raster ordering, index-modulo refinements,
# global contour/channel, box-based region/patch/feature) lives in
# build_geom_and_hfun.py so it can be inspected/edited independently.
import build_geom_and_hfun as recipe
from build_geom_and_hfun import GLOBAL_HMIN, GLOBAL_HMAX

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
        # Broad bounding box: roughly the STOFS-3D-Atlantic extent
        return box(-100.0, 7.0, -50.0, 47.0)


def _build_hfun(
    manifest: Dict,
    domain_shape,
    nprocs: int,
    execution_mode: str,
    logger=None,
) -> Hfun:
    """Construct the HfunCollector with all DEMs and all refinements.

    Thin wrapper around ``build_geom_and_hfun.build_hfun``. The actual
    recipe (raster ordering, index-modulo per-source refinements, global
    contour/channel, box-based region/patch/feature) lives in
    build_geom_and_hfun.py so it can be edited/inspected independently.

    Parameters
    ----------
    manifest : dict
        Output of download_dems.py — maps DEM name → metadata.
    domain_shape :
        Shapely geometry (used as base_shape for spatial clipping of DEMs).
    nprocs : int
        Number of workers for parallel / MPI mode.
    execution_mode : {'serial', 'parallel', 'mpi'}
    logger : logging.Logger, optional

    Returns
    -------
    Hfun (HfunCollector)
    """
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
) -> Dict:
    """Run meshdata() for one execution mode.

    Returns a result dict with timing stats and output mesh stats.
    Only meaningful on Rank 0; worker ranks return an empty dict.
    """
    log = _logger
    result: Dict = {"mode": mode, "status": "pending"}

    prof = cProfile.Profile()
    t0 = time.perf_counter()

    try:
        log.info(f"{'='*60}")
        log.info(f"Starting mode: {mode.upper()}")
        log.info(f"{'='*60}")

        hfun = _build_hfun(manifest, domain_shape, nprocs, mode)

        # ── Profile + run ────────────────────────────────────────────
        prof.enable()
        meshdata = hfun.meshdata()
        prof.disable()

        wall_time = time.perf_counter() - t0

        # Workers return None from meshdata()
        if meshdata is None:
            return {}  # worker rank — caller handles

        # ── Stats ────────────────────────────────────────────────────
        n_nodes = len(meshdata.coords)
        n_tria = len(meshdata.tria) if meshdata.tria is not None else 0
        vals = meshdata.values
        result = {
            "mode": mode,
            "status": "success",
            "wall_time_s": round(wall_time, 3),
            "n_nodes": n_nodes,
            "n_triangles": n_tria,
            "hfun_min": float(np.min(vals)),
            "hfun_max": float(np.max(vals)),
            "hfun_mean": float(np.mean(vals)),
            "hfun_std": float(np.std(vals)),
        }
        log.info(
            f"[{mode}] DONE in {wall_time:.1f}s  "
            f"nodes={n_nodes:,}  tria={n_tria:,}  "
            f"hfun=[{result['hfun_min']:.0f}, {result['hfun_max']:.0f}]"
        )

        # ── Save cProfile ────────────────────────────────────────────
        prof_path = out_dir / f"profile_{mode}.prof"
        prof.dump_stats(str(prof_path))
        log.info(f"cProfile saved to {prof_path}")

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
            "status": "failed",
            "wall_time_s": round(wall_time, 3),
            "error": repr(exc),
            "traceback": tb,
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCSMesh STOFS-3D-Atlantic benchmark (serial / parallel / MPI)."
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
        help="Directory for results, profiles, and logs",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=max(os.cpu_count() or 1, 1),
        help="Number of processes for parallel mode (default: all CPUs)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["serial", "parallel", "mpi"],
        default=["serial", "parallel", "mpi"],
        help="Execution modes to benchmark (default: all three)",
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

    # Apply CLI overrides to the recipe module (build_geom_and_hfun), which
    # is the single source of truth for the size bounds used when building
    # the Hfun. run_benchmark's module-level GLOBAL_HMIN/HMAX are just the
    # imported defaults for the argparse help text.
    recipe.GLOBAL_HMIN = args.hmin
    recipe.GLOBAL_HMAX = args.hmax

    # Only rank 0 does setup I/O; workers skip straight to meshdata() calls.
    if _IS_MANAGER:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Output directory : {args.out_dir.resolve()}")
        _logger.info(f"Manifest         : {args.manifest.resolve()}")
        _logger.info(f"Shapefile        : {args.shapefile}")
        _logger.info(f"nprocs           : {args.nprocs}")
        _logger.info(f"Modes            : {args.modes}")
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
                    "Re-run with: mpiexec -n <N+1> python run_benchmark.py --modes mpi"
                )
            continue

        if mode in ("serial", "parallel") and not _IS_MANAGER:
            # Workers have nothing to do for non-MPI modes
            continue

        result = _run_mode(
            manifest,
            domain_shape,
            args.nprocs,
            mode,
            args.out_dir,
        )

        # Worker ranks return empty dict from _run_mode when meshdata()=None
        if result:
            all_results.append(result)

    # ── Write summary JSON (rank 0 only) ─────────────────────────────
    if _IS_MANAGER and all_results:
        # Compute speedups relative to serial baseline (if present)
        serial_time = next(
            (r["wall_time_s"] for r in all_results
             if r["mode"] == "serial" and r["status"] == "success"),
            None,
        )
        for r in all_results:
            if serial_time and r.get("status") == "success":
                r["speedup_vs_serial"] = round(serial_time / r["wall_time_s"], 3)

        summary = {
            "hostname": os.uname().nodename,
            "mpi_size": _SIZE,
            "nprocs_parallel": args.nprocs,
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
        _logger.info("\n" + "=" * 65)
        _logger.info("  BENCHMARK SUMMARY")
        _logger.info("=" * 65)
        _logger.info(f"  {'Mode':<12} {'Status':<10} {'Time (s)':>10}  {'Speedup':>9}  {'Nodes':>10}")
        _logger.info(f"  {'-'*12} {'-'*10} {'-'*10}  {'-'*9}  {'-'*10}")
        for r in all_results:
            sp = f"{r.get('speedup_vs_serial', 1.0):.2f}x"
            nd = f"{r.get('n_nodes', 0):,}"
            _logger.info(
                f"  {r['mode']:<12} {r['status']:<10} "
                f"{r.get('wall_time_s', 0):>10.2f}  {sp:>9}  {nd:>10}"
            )
        _logger.info("=" * 65)


if __name__ == "__main__":
    main()
