"""STOFS-3D-Atlantic OCSMesh benchmark: serial, parallel, MPI, and hybrid modes.

This script is the heart of the Hercules benchmark. It:

1.  Loads the DEM manifest produced by ``download_dems.py``.
2.  Builds an ``HfunCollector`` that covers the full STOFS-3D-Atlantic
    domain with refinements controlled by recipe flags.
3.  Runs back-to-back executions across the requested modes:

        serial_true   (execution_mode='serial',   nprocs=1)
        serial_mp     (execution_mode='serial',   nprocs=N)
        parallel      (execution_mode='parallel', nprocs=N)
        mpi           (execution_mode='mpi',      nprocs=N)
        mpi_no_pool   (execution_mode='mpi',      nprocs=1 per rank)
        mpi_hybrid    (execution_mode='mpi',      nprocs=auto per rank)

    mpi_no_pool and mpi_hybrid both use OCSMesh execution_mode='mpi'.
    The difference is the nprocs passed per rank:
        mpi_no_pool  -> nprocs=1  (pure MPI, no internal Pool)
        mpi_hybrid   -> nprocs=auto (each rank spawns a Pool sized to
                        available cores / worker ranks on the node)

4.  Times each mode with wall-clock, CPU seconds, and cProfile.
5.  Records CPU utilization (busy_cores, utilization %) per mode.
6.  After all modes complete, runs a pixel-exact correctness check
    comparing meshdata values between serial_mp and each MPI mode.
7.  Saves per-mode timing, stats, .2dm mesh files, cProfile .prof files,
    and a JSON summary (benchmark_results.json).

Mode -> OCSMesh mapping
-----------------------
    Benchmark mode   execution_mode   nprocs passed to Hfun
    serial_true      'serial'         1
    serial_mp        'serial'         N
    parallel         'parallel'       N
    mpi              'mpi'            N
    mpi_no_pool      'mpi'            1         (per rank)
    mpi_hybrid       'mpi'            auto      (per rank, cores/workers)

MPI awareness
-------------
Under mpiexec / srun only Rank 0 runs serial_true, serial_mp, parallel,
and the non-MPI benchmark modes. All ranks participate in mpi,
mpi_no_pool, and mpi_hybrid. Worker ranks return None from meshdata();
only Rank 0 writes output.

Auto core detection for mpi_hybrid
-----------------------------------
Uses os.sched_getaffinity (respects SLURM/taskset CPU pinning) divided
by the number of worker ranks sharing the same node. Falls back to
SLURM_CPUS_PER_TASK if set, then os.cpu_count().

Usage
-----
    # Non-MPI modes only:
    python run_benchmark.py --manifest dem_manifest.json \\
        --shapefile /path/to/stofs_domain.shp \\
        --out-dir ./results --nprocs 79 \\
        --modes serial_mp parallel

    # All modes including MPI:
    srun --mpi=pmi2 -n 80 python run_benchmark.py \\
        --manifest dem_manifest.json \\
        --shapefile /path/to/stofs_domain.shp \\
        --out-dir ./results --nprocs 79 \\
        --modes serial_mp parallel mpi mpi_no_pool mpi_hybrid

    # mpi_no_pool + mpi_hybrid only (fat ranks, 8 workers x 8 cores):
    srun --mpi=pmi2 --ntasks=9 --cpus-per-task=8 python run_benchmark.py \\
        --manifest dem_manifest_smoke.json \\
        --shapefile /path/to/stofs_domain.shp \\
        --out-dir ./results \\
        --modes serial_mp mpi_no_pool mpi_hybrid \\
        --config-f
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
import ocsmesh  # noqa: F401 — triggers _configure_mpi_environment() in __init__
from ocsmesh.mpi import (
    MPIExecutor,
    _get_mpi,
    _is_mpi_active,
    _is_mpi_env_detected,
)

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
# Logging
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

import build_geom_and_hfun as recipe
from build_geom_and_hfun import GLOBAL_HMIN, GLOBAL_HMAX

# ---------------------------------------------------------------------------
# CPU utilization helpers
# ---------------------------------------------------------------------------

def _cpu_seconds() -> float:
    """CPU seconds for this process + reaped children (Pool workers).

    Pool workers are joined when the Pool closes, so their CPU time lands
    in children_user + children_system.
    """
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def _affinity_cores() -> int:
    """Cores this process is allowed to run on (respects SLURM/taskset)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _node_cores_and_local_size(comm) -> tuple:
    """Return (total node cores, ranks on this node).

    Unions each local rank's affinity mask so we get the full node budget
    even when the launcher has pinned each rank to a disjoint subset.
    """
    try:
        mask = set(os.sched_getaffinity(0))
    except AttributeError:
        mask = set(range(os.cpu_count() or 1))

    if comm is None or comm.Get_size() == 1:
        return len(mask), 1

    from mpi4py import MPI
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    try:
        masks = node_comm.allgather(mask)
        local_size = node_comm.Get_size()
    finally:
        node_comm.Free()

    return len(set().union(*masks)), local_size


def _plan_hybrid_cores(comm) -> int:
    """Decide how many Pool workers each rank may start for mpi_hybrid.

    Resolution order:
      1. SLURM_CPUS_PER_TASK  — launcher already sized the rank
      2. Affinity mask < node cores — launcher already pinned the rank
      3. node_cores / worker_ranks_on_node — auto divide
    """
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        return max(1, int(os.environ["SLURM_CPUS_PER_TASK"]))

    mine = _affinity_cores()
    size = comm.Get_size() if comm is not None else 1
    on_node, local = _node_cores_and_local_size(comm)

    if mine < on_node:
        # Launcher already pinned each rank to a disjoint subset
        return mine

    # Auto divide: exclude rank 0 (coordinator) from the divisor
    workers_here = max(1, local - 1) if local > 1 else 1
    return max(1, on_node // workers_here)


# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------
_ALL_MODES = [
    "serial_true",
    "serial_mp",
    "parallel",
    "mpi",
    "mpi_no_pool",
    "mpi_hybrid",
]

# Maps benchmark mode -> (ocsmesh_execution_mode, nprocs_override)
# nprocs_override=None  -> use args.nprocs as-is
# nprocs_override=1     -> force nprocs=1 (serial_true, mpi_no_pool)
# nprocs_override='auto'-> compute per-rank budget at runtime (mpi_hybrid)
_MODE_CONFIG = {
    "serial_true": ("serial",   1),
    "serial_mp":   ("serial",   None),
    "parallel":    ("parallel", None),
    "mpi":         ("mpi",      None),
    "mpi_no_pool": ("mpi",      1),
    "mpi_hybrid":  ("mpi",      "auto"),
}

# Modes that require MPI to be active
_MPI_MODES = {"mpi", "mpi_no_pool", "mpi_hybrid"}

# Modes that only run on rank 0
_RANK0_ONLY_MODES = {"serial_true", "serial_mp", "parallel"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest(manifest_path: Path) -> Dict:
    with open(manifest_path) as fh:
        return json.load(fh)


def _load_domain_shape(shapefile: Optional[Path]):
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
            "No shapefile provided or file not found. "
            "Using default STOFS-3D-Atlantic bounding box."
        )
        return box(-100.0, 7.0, -50.0, 47.0)


def _build_geom(manifest, domain_shape, nprocs):
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
    skip_box_refinements: bool = False,
    all_fast_refinements: bool = False,
    config_f: bool = False,
    config_g: bool = False,
) -> Hfun:
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
        skip_box_refinements=skip_box_refinements,
        all_fast_refinements=all_fast_refinements,
        config_f=config_f,
        config_g=config_g,
    )


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def _check_correctness(
    baseline_values: np.ndarray,
    baseline_mode: str,
    results: List[Dict],
    stored_values: Dict[str, np.ndarray],
) -> List[Dict]:
    """Pixel-exact comparison of meshdata values vs serial_mp baseline.

    Returns a list of correctness records appended to each result dict.
    """
    checks = []
    for r in results:
        mode = r["mode"]
        if mode == baseline_mode or r["status"] != "success":
            continue
        if mode not in stored_values:
            continue
        other = stored_values[mode]
        if baseline_values.shape != other.shape:
            checks.append({
                "mode": mode,
                "vs_baseline": baseline_mode,
                "match": False,
                "reason": (
                    f"shape mismatch: baseline={baseline_values.shape} "
                    f"vs {mode}={other.shape}"
                ),
            })
        elif np.array_equal(baseline_values, other, equal_nan=True):
            checks.append({
                "mode": mode,
                "vs_baseline": baseline_mode,
                "match": True,
                "reason": "values identical",
            })
        else:
            diff = np.abs(
                np.nan_to_num(baseline_values) - np.nan_to_num(other)
            )
            n_differ = int((diff > 0).sum())
            max_diff = float(diff.max())
            checks.append({
                "mode": mode,
                "vs_baseline": baseline_mode,
                "match": False,
                "reason": (
                    f"{n_differ} pixels differ, max diff={max_diff:.6g}"
                ),
            })
    return checks


# ---------------------------------------------------------------------------
# Profiled runner
# ---------------------------------------------------------------------------

def _run_mode(
    manifest: Dict,
    domain_shape,
    nprocs: int,
    mode: str,
    out_dir: Path,
    comm,
    light_features: bool = False,
    skip_topofunc: bool = False,
    skip_constraints: bool = False,
    skip_box_refinements: bool = False,
    all_fast_refinements: bool = False,
    config_f: bool = False,
    config_g: bool = False,
    full_pipeline: bool = False,
) -> tuple:
    """Run meshdata() for one benchmark mode.

    Returns
    -------
    (result_dict, meshdata_values)
        result_dict  : timing + stats dict (meaningful on rank 0 only)
        meshdata_values : np.ndarray of hfun values (rank 0) or None (workers)
    """
    log = _logger
    result: Dict = {"mode": mode, "status": "pending"}
    meshdata_values = None

    ocsmesh_mode, nprocs_override = _MODE_CONFIG[mode]

    # Resolve effective nprocs for this mode
    if nprocs_override == 1:
        effective_nprocs = 1
    elif nprocs_override == "auto":
        effective_nprocs = _plan_hybrid_cores(comm)
        if _IS_MANAGER:
            log.info(
                f"[{mode}] mpi_hybrid: auto core budget = "
                f"{effective_nprocs} cores/rank"
            )
    else:
        effective_nprocs = nprocs

    prof = cProfile.Profile()
    t0_wall = time.perf_counter()
    t0_cpu  = _cpu_seconds()
    stage_times: Dict = {}

    try:
        log.info(f"{'='*60}")
        log.info(f"Starting mode: {mode.upper()}")
        log.info(
            f"  ocsmesh execution_mode={ocsmesh_mode!r}  "
            f"nprocs={effective_nprocs}  full_pipeline={full_pipeline}"
        )
        log.info(f"{'='*60}")

        # ── Stage 1: (optional) Geom ──────────────────────────────────
        geom = None
        if full_pipeline:
            t_geom = time.perf_counter()
            geom = _build_geom(manifest, domain_shape, effective_nprocs)
            stage_times["geom_build_s"] = round(
                time.perf_counter() - t_geom, 3)

        # ── Stage 2: build Hfun recipe ────────────────────────────────
        t_hfun_build = time.perf_counter()
        hfun = _build_hfun(
            manifest, domain_shape, effective_nprocs, ocsmesh_mode,
            light_features=light_features,
            skip_topofunc=skip_topofunc,
            skip_constraints=skip_constraints,
            skip_box_refinements=skip_box_refinements,
            all_fast_refinements=all_fast_refinements,
            config_f=config_f,
            config_g=config_g,
        )
        stage_times["hfun_build_s"] = round(
            time.perf_counter() - t_hfun_build, 3)

        # ── Stage 3: hfun.meshdata() — MPI-parallelized stage ─────────
        t_meshdata = time.perf_counter()
        prof.enable()
        meshdata = hfun.meshdata()
        prof.disable()
        stage_times["hfun_meshdata_s"] = round(
            time.perf_counter() - t_meshdata, 3)

        # Worker ranks return None; only rank 0 continues
        if meshdata is None:
            return {}, None

        # ── Stats ─────────────────────────────────────────────────────
        n_nodes = len(meshdata.coords)
        n_tria  = len(meshdata.tria) if meshdata.tria is not None else 0
        vals    = meshdata.values
        meshdata_values = np.array(vals, copy=True)

        # ── Stage 4: (optional) MeshDriver ───────────────────────────
        final_mesh = None
        if full_pipeline and geom is not None:
            t_driver = time.perf_counter()
            log.info(f"[{mode}] Running MeshDriver (final mesh)...")
            driver = MeshDriver(
                geom, hfun, engine_name="gmsh",
                bnd_representation="exact",
            )
            final_mesh = driver.run()
            stage_times["meshdriver_run_s"] = round(
                time.perf_counter() - t_driver, 3)

        # ── CPU utilization ───────────────────────────────────────────
        wall_time = time.perf_counter() - t0_wall
        cpu_s     = _cpu_seconds() - t0_cpu

        # For MPI modes, sum CPU seconds across all ranks
        if _MPI_ACTIVE and mode in _MPI_MODES and comm is not None:
            cpu_s = comm.allreduce(cpu_s)

        busy_cores  = cpu_s / wall_time if wall_time > 0 else 0.0
        utilization = busy_cores / max(1, effective_nprocs)

        result = {
            "mode":                   mode,
            "ocsmesh_execution_mode": ocsmesh_mode,
            "effective_nprocs":       effective_nprocs,
            "full_pipeline":          full_pipeline,
            "status":                 "success",
            "wall_time_s":            round(wall_time, 3),
            "cpu_s":                  round(cpu_s, 3),
            "busy_cores":             round(busy_cores, 2),
            "utilization_pct":        round(100.0 * utilization, 1),
            "n_nodes":                n_nodes,
            "n_triangles":            n_tria,
            "hfun_min":               float(np.min(vals)),
            "hfun_max":               float(np.max(vals)),
            "hfun_mean":              float(np.mean(vals)),
            "hfun_std":               float(np.std(vals)),
            "stage_times_s":          stage_times,
        }
        log.info(
            f"[{mode}] DONE in {wall_time:.1f}s  "
            f"cpu={cpu_s:.1f}s  busy={busy_cores:.1f} cores  "
            f"util={100*utilization:.0f}%  "
            f"nodes={n_nodes:,}  tria={n_tria:,}  "
            f"hfun=[{result['hfun_min']:.0f}, {result['hfun_max']:.0f}]"
        )
        log.info(f"[{mode}] stage times (s): {stage_times}")

        # ── Save cProfile ─────────────────────────────────────────────
        prof_path = out_dir / f"profile_{mode}.prof"
        prof.dump_stats(str(prof_path))
        log.info(f"cProfile saved to {prof_path}")

        # ── Save hfun .2dm ────────────────────────────────────────────
        hfun_path = out_dir / f"hfun_{mode}.2dm"
        try:
            Mesh(meshdata).write(str(hfun_path), overwrite=True, format="2dm")
            log.info(f"Hfun size-field saved to {hfun_path}")
        except Exception as mesh_exc:  # pylint: disable=broad-exception-caught
            log.warning(f"Could not save hfun .2dm: {mesh_exc}")

        # ── Save final mesh .2dm ──────────────────────────────────────
        if final_mesh is not None:
            final_path = out_dir / f"mesh_{mode}.2dm"
            try:
                final_mesh.write(
                    str(final_path), format="2dm", overwrite=True)
                log.info(f"Final mesh saved to {final_path}")
            except Exception as mesh_exc:  # pylint: disable=broad-exception-caught
                log.warning(f"Could not save final mesh .2dm: {mesh_exc}")

        # ── cProfile top-20 to log ────────────────────────────────────
        sio = io.StringIO()
        ps  = pstats.Stats(prof, stream=sio)
        ps.sort_stats("cumulative")
        ps.print_stats(20)
        log.info(f"\ncProfile top-20 ({mode}):\n{sio.getvalue()}")

    except Exception as exc:  # pylint: disable=broad-exception-caught
        wall_time = time.perf_counter() - t0_wall
        tb = traceback.format_exc()
        log.error(f"[{mode}] FAILED after {wall_time:.1f}s: {exc}\n{tb}")
        result = {
            "mode":                   mode,
            "ocsmesh_execution_mode": ocsmesh_mode,
            "effective_nprocs":       effective_nprocs,
            "full_pipeline":          full_pipeline,
            "status":                 "failed",
            "wall_time_s":            round(wall_time, 3),
            "cpu_s":                  0.0,
            "busy_cores":             0.0,
            "utilization_pct":        0.0,
            "stage_times_s":          stage_times,
            "error":                  repr(exc),
            "traceback":              tb,
        }

    return result, meshdata_values


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
  serial_true   True single-core (nprocs=1 forced).
  serial_mp     Serial mode, Pool steps use --nprocs workers.
  parallel      Full multiprocessing (execution_mode=parallel).
  mpi           MPI via MPIExecutor, nprocs=--nprocs per rank.
  mpi_no_pool   MPI, nprocs=1 per rank (pure MPI, no internal Pool).
  mpi_hybrid    MPI, nprocs=auto per rank (MPI + internal Pool).

Recipe configs (mutually exclusive with individual flags)
---------------------------------------------------------
  --config-f    Config F: flow+const+constraints+contour/channel,
                all tiles, no boxes. All ops MPI-dispatched.
                Maps to Anas benchmark Config E.
  --config-g    Config G: Config F + patch + feature (BOX2+BOX3).
                Full pipeline, all ops MPI-dispatched.
                Maps to Anas benchmark Config F.
        """,
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(__file__).parent / "dem_manifest.json",
    )
    parser.add_argument("--shapefile", type=Path, default=None)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("./benchmark_results"),
    )
    parser.add_argument(
        "--nprocs", type=int,
        default=max(os.cpu_count() or 1, 1),
        help=(
            "Workers for serial_mp, parallel, mpi modes. "
            "Ignored for serial_true (always 1), mpi_no_pool (always 1), "
            "and mpi_hybrid (auto). Default: all CPUs."
        ),
    )
    parser.add_argument(
        "--modes", nargs="+", choices=_ALL_MODES, default=_ALL_MODES,
    )

    # ── Recipe flags ──────────────────────────────────────────────────────
    parser.add_argument("--light-features",      action="store_true")
    parser.add_argument("--skip-topofunc",        action="store_true")
    parser.add_argument("--skip-constraints",     action="store_true")
    parser.add_argument("--skip-box-refinements", action="store_true")
    parser.add_argument("--all-fast-refinements", action="store_true")
    parser.add_argument(
        "--config-f", action="store_true",
        help=(
            "Config F: flow+const+constraints+contour/channel on all tiles, "
            "no boxes. All ops are MPI-dispatched. No shape bottleneck. "
            "Maps to Anas Config E."
        ),
    )
    parser.add_argument(
        "--config-g", action="store_true",
        help=(
            "Config G: Config F + patch + feature (BOX2+BOX3). "
            "Full pipeline, all ops MPI-dispatched. "
            "Maps to Anas Config F."
        ),
    )
    parser.add_argument("--full-pipeline", action="store_true")
    parser.add_argument("--hmin", type=float, default=GLOBAL_HMIN)
    parser.add_argument("--hmax", type=float, default=GLOBAL_HMAX)
    args = parser.parse_args()

    # config_f and config_g are mutually exclusive with individual flags
    if args.config_f and args.config_g:
        if _IS_MANAGER:
            _logger.error("--config-f and --config-g are mutually exclusive.")
        sys.exit(1)

    recipe.GLOBAL_HMIN = args.hmin
    recipe.GLOBAL_HMAX = args.hmax

    # Resolve MPI comm for hybrid core planning
    comm = None
    if _MPI_ACTIVE:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    if _IS_MANAGER:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Output directory  : {args.out_dir.resolve()}")
        _logger.info(f"Manifest          : {args.manifest.resolve()}")
        _logger.info(f"Shapefile         : {args.shapefile}")
        _logger.info(f"nprocs            : {args.nprocs}")
        _logger.info(f"Modes             : {args.modes}")
        _logger.info(f"light_features    : {args.light_features}")
        _logger.info(f"skip_topofunc     : {args.skip_topofunc}")
        _logger.info(f"skip_constraints  : {args.skip_constraints}")
        _logger.info(f"skip_box_refs     : {args.skip_box_refinements}")
        _logger.info(f"all_fast_refs     : {args.all_fast_refinements}")
        _logger.info(f"config_f          : {args.config_f}")
        _logger.info(f"config_g          : {args.config_g}")
        _logger.info(f"full_pipeline     : {args.full_pipeline}")
        _logger.info(f"MPI active        : {_MPI_ACTIVE}  (size={_SIZE})")
        _logger.info(f"hmin={recipe.GLOBAL_HMIN} m  hmax={recipe.GLOBAL_HMAX} m")
        if "mpi_hybrid" in args.modes and _MPI_ACTIVE:
            hybrid_cores = _plan_hybrid_cores(comm)
            _logger.info(
                f"mpi_hybrid auto cores/rank : {hybrid_cores}"
            )

    # ── Load inputs ───────────────────────────────────────────────────────
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
        manifest     = None
        domain_shape = None

    if _MPI_ACTIVE:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        manifest     = comm.bcast(manifest,     root=0)
        domain_shape = comm.bcast(domain_shape, root=0)

    # ── Run each mode ─────────────────────────────────────────────────────
    all_results:     List[Dict]              = []
    stored_values:   Dict[str, np.ndarray]   = {}

    for mode in args.modes:

        if mode in _MPI_MODES and not _MPI_ACTIVE:
            if _IS_MANAGER:
                _logger.warning(
                    f"Skipping {mode} — not running under mpiexec/srun. "
                    f"Re-run with: srun --mpi=pmi2 -n <N+1> python "
                    f"run_benchmark.py --modes {mode}"
                )
            continue

        if mode in _RANK0_ONLY_MODES and not _IS_MANAGER:
            continue

        result, vals = _run_mode(
            manifest, domain_shape, args.nprocs, mode, args.out_dir,
            comm=comm,
            light_features=args.light_features,
            skip_topofunc=args.skip_topofunc,
            skip_constraints=args.skip_constraints,
            skip_box_refinements=args.skip_box_refinements,
            all_fast_refinements=args.all_fast_refinements,
            config_f=args.config_f,
            config_g=args.config_g,
            full_pipeline=args.full_pipeline,
        )

        if result:
            all_results.append(result)
        if vals is not None:
            stored_values[mode] = vals

    # ── Correctness check (rank 0 only) ───────────────────────────────────
    correctness_checks = []
    if _IS_MANAGER and len(stored_values) > 1:
        # Use serial_mp as baseline if present, else first successful mode
        baseline_mode = next(
            (m for m in ("serial_mp", "serial_true", "parallel")
             if m in stored_values),
            next(iter(stored_values), None),
        )
        if baseline_mode:
            correctness_checks = _check_correctness(
                stored_values[baseline_mode],
                baseline_mode,
                all_results,
                stored_values,
            )
            _logger.info("\n=== Correctness Check ===")
            for c in correctness_checks:
                status = "OK  " if c["match"] else "FAIL"
                _logger.info(
                    f"  {status}  {baseline_mode} vs {c['mode']}: "
                    f"{c['reason']}"
                )

    # ── Write summary JSON (rank 0 only) ──────────────────────────────────
    if _IS_MANAGER and all_results:
        baseline_time = next(
            (r["wall_time_s"] for r in all_results
             if r["mode"] == "serial_mp" and r["status"] == "success"),
            None,
        )
        if baseline_time is None:
            baseline_time = next(
                (r["wall_time_s"] for r in all_results
                 if r["mode"] == "serial_true" and r["status"] == "success"),
                None,
            )
        for r in all_results:
            if baseline_time and r.get("status") == "success":
                r["speedup_vs_baseline"] = round(
                    baseline_time / r["wall_time_s"], 3)

        summary = {
            "hostname":           os.uname().nodename,
            "mpi_size":           _SIZE,
            "nprocs_requested":   args.nprocs,
            "hmin":               recipe.GLOBAL_HMIN,
            "hmax":               recipe.GLOBAL_HMAX,
            "n_dems":             sum(
                1 for v in manifest.values() if v.get("available")
            ),
            "correctness_checks": correctness_checks,
            "results":            all_results,
        }
        out_json = args.out_dir / "benchmark_results.json"
        out_json.write_text(json.dumps(summary, indent=2))
        _logger.info(f"\nResults written to {out_json}")

        # ── Summary table ─────────────────────────────────────────────
        _logger.info("\n" + "=" * 90)
        _logger.info("  BENCHMARK SUMMARY")
        _logger.info("=" * 90)
        _logger.info(
            f"  {'Mode':<14} {'nprocs':>6}  {'Status':<10} "
            f"{'Time (s)':>10}  {'Speedup':>9}  {'CPU (s)':>9}  "
            f"{'BusyCores':>10}  {'Util%':>6}  {'Nodes':>10}"
        )
        _logger.info(
            f"  {'-'*14} {'-'*6}  {'-'*10} "
            f"{'-'*10}  {'-'*9}  {'-'*9}  "
            f"{'-'*10}  {'-'*6}  {'-'*10}"
        )
        for r in all_results:
            sp  = f"{r.get('speedup_vs_baseline', 1.0):.2f}x"
            nd  = f"{r.get('n_nodes', 0):,}"
            np_ = r.get("effective_nprocs", "?")
            cpu = f"{r.get('cpu_s', 0.0):.1f}"
            bc  = f"{r.get('busy_cores', 0.0):.1f}"
            ut  = f"{r.get('utilization_pct', 0.0):.0f}%"
            _logger.info(
                f"  {r['mode']:<14} {np_:>6}  {r['status']:<10} "
                f"{r.get('wall_time_s', 0):>10.2f}  {sp:>9}  {cpu:>9}  "
                f"{bc:>10}  {ut:>6}  {nd:>10}"
            )
        _logger.info("=" * 90)

        # ── Correctness summary ────────────────────────────────────────
        if correctness_checks:
            _logger.info("\n  CORRECTNESS vs serial_mp baseline:")
            for c in correctness_checks:
                icon = "OK  " if c["match"] else "FAIL"
                _logger.info(f"    {icon}  {c['mode']}: {c['reason']}")
        _logger.info("=" * 90)


if __name__ == "__main__":
    main()
