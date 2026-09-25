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

4.  Times each mode with wall-clock, CPU seconds, and cProfile.
5.  Records CPU utilization (busy_cores, utilization %) per mode.
6.  After all modes complete, runs a pixel-exact correctness check.
7.  Saves per-mode timing, stats, .2dm mesh files, cProfile .prof files,
    and a JSON summary (benchmark_results.json).

Recipe configs (mutually exclusive)
-------------------------------------
  --config-f      Config F-anas: Anas PR benchmark (flow+const+constraints+
                  contour/channel, CUDEM tiles only, no boxes).
  --config-g      Config G: Config F-anas + patch + feature (BOX2+BOX3).
  --config-r      Config R: full production recipe (5 ops, 451 tiles).
  --config-r0     Isolation: no refinements.
  --config-r1     Isolation: add_constant_value only.
  --config-r2     Isolation: add_topo_bound_constraint only.
  --config-r3     Isolation: add_contour only.
  --config-r4     Isolation: add_subtidal_flow_limiter only.
  --config-r5     Isolation: add_channel only.
  --config-ffat   Config F-fat: 2x finer production recipe
                  (MA/NH/ME, 44 tiles, all 5 ops).
  --config-f0     Isolation: no refinements (F-fat domain).
  --config-f1     Isolation: add_constant_value only (F-fat).
  --config-f2     Isolation: add_topo_bound_constraint only (F-fat).
  --config-f3     Isolation: add_contour only (F-fat).
  --config-f4     Isolation: add_subtidal_flow_limiter only (F-fat).
  --config-f5     Isolation: add_channel only (F-fat).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# TMPDIR override — must happen before ANY import of ocsmesh or rasterio.
# SLURM's node prolog sets TMPDIR=/local/scratch/$USER/$JOBID (node-local,
# not visible to other nodes) AFTER our bash export runs. By re-reading
# OCSMESH_SHARED_TMPDIR (set in the SLURM script) here in Python, we ensure
# all temp files go to the shared Lustre filesystem regardless of what the
# prolog did to TMPDIR.
# ---------------------------------------------------------------------------
import os as _os
import tempfile as _tempfile
_shared_tmpdir = _os.environ.get('OCSMESH_SHARED_TMPDIR', '')
if _shared_tmpdir:
    _os.makedirs(_shared_tmpdir, exist_ok=True)
    _os.environ['TMPDIR'] = _shared_tmpdir
    _tempfile.tempdir = _shared_tmpdir

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
import ocsmesh  # noqa: F401
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
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def _affinity_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _node_cores_and_local_size(comm) -> tuple:
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
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        return max(1, int(os.environ["SLURM_CPUS_PER_TASK"]))
    mine = _affinity_cores()
    on_node, local = _node_cores_and_local_size(comm)
    if mine < on_node:
        return mine
    workers_here = max(1, local - 1) if local > 1 else 1
    return max(1, on_node // workers_here)


# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------
_ALL_MODES = [
    "serial_true", "serial_mp", "parallel",
    "mpi", "mpi_no_pool", "mpi_hybrid",
]
_MODE_CONFIG = {
    "serial_true": ("serial",   1),
    "serial_mp":   ("serial",   None),
    "parallel":    ("parallel", None),
    "mpi":         ("mpi",      None),
    "mpi_no_pool": ("mpi",      1),
    "mpi_hybrid":  ("mpi",      "auto"),
}
_MPI_MODES        = {"mpi", "mpi_no_pool", "mpi_hybrid"}
_RANK0_ONLY_MODES = {"serial_true", "serial_mp", "parallel"}

# All mutually exclusive config flags
_CONFIG_FLAGS = [
    "config_f", "config_g",
    "config_r", "config_r0", "config_r1", "config_r2",
    "config_r3", "config_r4", "config_r5",
    "config_ffat", "config_f0", "config_f1", "config_f2",
    "config_f3", "config_f4", "config_f5",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest(manifest_path: Path) -> Dict:
    with open(manifest_path) as fh:
        return json.load(fh)


def _load_domain_shape(shapefile):
    if shapefile and Path(shapefile).exists():
        _logger.info(f"Loading domain shapefile: {shapefile}")
        gdf = gpd.read_file(shapefile)
        if gdf.crs and not gdf.crs.equals("EPSG:4326"):
            gdf = gdf.to_crs("EPSG:4326")
        geom = gdf.union_all()
        _logger.info(f"Domain geometry type: {geom.geom_type}")
        return geom
    else:
        _logger.warning("No shapefile — using default STOFS bbox.")
        return box(-100.0, 7.0, -50.0, 47.0)


def _build_geom(manifest, domain_shape, nprocs):
    raster_paths, _ = recipe.load_ordered_rasters(manifest)
    if not raster_paths:
        raise RuntimeError("No DEM files found for geom build.")
    return recipe.build_geom(raster_paths, domain_shape, nprocs)


def _build_hfun(manifest, domain_shape, nprocs, execution_mode,
                light_features=False, skip_topofunc=False,
                skip_constraints=False, skip_box_refinements=False,
                all_fast_refinements=False,
                config_f=False, config_g=False,
                config_r=False, config_r0=False,
                config_r1=False, config_r2=False,
                config_r3=False, config_r4=False,
                config_r5=False,
                config_ffat=False,
                config_f0=False, config_f1=False,
                config_f2=False, config_f3=False,
                config_f4=False, config_f5=False) -> Hfun:
    raster_paths, raster_metas = recipe.load_ordered_rasters(manifest)
    if not raster_paths:
        raise RuntimeError("No DEM files found.")
    return recipe.build_hfun(
        raster_paths, raster_metas, domain_shape,
        nprocs, execution_mode,
        light_features=light_features,
        skip_topofunc=skip_topofunc,
        skip_constraints=skip_constraints,
        skip_box_refinements=skip_box_refinements,
        all_fast_refinements=all_fast_refinements,
        config_f=config_f, config_g=config_g,
        config_r=config_r, config_r0=config_r0,
        config_r1=config_r1, config_r2=config_r2,
        config_r3=config_r3, config_r4=config_r4,
        config_r5=config_r5,
        config_ffat=config_ffat,
        config_f0=config_f0, config_f1=config_f1,
        config_f2=config_f2, config_f3=config_f3,
        config_f4=config_f4, config_f5=config_f5,
    )


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def _check_correctness(baseline_values, baseline_mode,
                       results, stored_values):
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
                "mode": mode, "vs_baseline": baseline_mode,
                "match": False,
                "reason": (f"shape mismatch: "
                           f"baseline={baseline_values.shape} "
                           f"vs {mode}={other.shape}"),
            })
        elif np.array_equal(baseline_values, other, equal_nan=True):
            checks.append({
                "mode": mode, "vs_baseline": baseline_mode,
                "match": True, "reason": "values identical",
            })
        else:
            diff = np.abs(
                np.nan_to_num(baseline_values) - np.nan_to_num(other)
            )
            checks.append({
                "mode": mode, "vs_baseline": baseline_mode,
                "match": False,
                "reason": (f"{int((diff>0).sum())} pixels differ, "
                           f"max diff={float(diff.max()):.6g}"),
            })
    return checks


# ---------------------------------------------------------------------------
# Profiled runner
# ---------------------------------------------------------------------------

def _run_mode(manifest, domain_shape, nprocs, mode, out_dir, comm,
              light_features=False, skip_topofunc=False,
              skip_constraints=False, skip_box_refinements=False,
              all_fast_refinements=False,
              config_f=False, config_g=False,
              config_r=False, config_r0=False,
              config_r1=False, config_r2=False,
              config_r3=False, config_r4=False,
              config_r5=False,
              config_ffat=False,
              config_f0=False, config_f1=False,
              config_f2=False, config_f3=False,
              config_f4=False, config_f5=False,
              full_pipeline=False) -> tuple:
    """Run meshdata() for one benchmark mode."""
    log = _logger
    result: Dict = {"mode": mode, "status": "pending"}
    meshdata_values = None

    ocsmesh_mode, nprocs_override = _MODE_CONFIG[mode]

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

    prof      = cProfile.Profile()
    t0_wall   = time.perf_counter()
    t0_cpu    = _cpu_seconds()
    stage_times: Dict = {}

    try:
        log.info(f"{'='*60}")
        log.info(f"Starting mode: {mode.upper()}")
        log.info(
            f"  ocsmesh execution_mode={ocsmesh_mode!r}  "
            f"nprocs={effective_nprocs}  full_pipeline={full_pipeline}"
        )
        log.info(f"{'='*60}")

        geom = None
        if full_pipeline:
            t_geom = time.perf_counter()
            geom = _build_geom(manifest, domain_shape, effective_nprocs)
            stage_times["geom_build_s"] = round(
                time.perf_counter() - t_geom, 3)

        t_hfun_build = time.perf_counter()
        hfun = _build_hfun(
            manifest, domain_shape, effective_nprocs, ocsmesh_mode,
            light_features=light_features,
            skip_topofunc=skip_topofunc,
            skip_constraints=skip_constraints,
            skip_box_refinements=skip_box_refinements,
            all_fast_refinements=all_fast_refinements,
            config_f=config_f, config_g=config_g,
            config_r=config_r, config_r0=config_r0,
            config_r1=config_r1, config_r2=config_r2,
            config_r3=config_r3, config_r4=config_r4,
            config_r5=config_r5,
            config_ffat=config_ffat,
            config_f0=config_f0, config_f1=config_f1,
            config_f2=config_f2, config_f3=config_f3,
            config_f4=config_f4, config_f5=config_f5,
        )
        stage_times["hfun_build_s"] = round(
            time.perf_counter() - t_hfun_build, 3)

        t_meshdata = time.perf_counter()
        prof.enable()
        meshdata = hfun.meshdata()
        prof.disable()
        stage_times["hfun_meshdata_s"] = round(
            time.perf_counter() - t_meshdata, 3)

        # FIX: workers participate in allreduce before exiting
        if meshdata is None:
            if _MPI_ACTIVE and mode in _MPI_MODES and comm is not None:
                cpu_s = _cpu_seconds() - t0_cpu
                comm.allreduce(cpu_s)
            return {}, None

        n_nodes = len(meshdata.coords)
        n_tria  = len(meshdata.tria) if meshdata.tria is not None else 0
        vals    = meshdata.values
        meshdata_values = np.array(vals, copy=True)

        final_mesh = None
        if full_pipeline and geom is not None:
            t_driver = time.perf_counter()
            log.info(f"[{mode}] Running MeshDriver...")
            driver = MeshDriver(
                geom, hfun, engine_name="gmsh",
                bnd_representation="exact",
            )
            final_mesh = driver.run()
            stage_times["meshdriver_run_s"] = round(
                time.perf_counter() - t_driver, 3)

        wall_time = time.perf_counter() - t0_wall
        cpu_s     = _cpu_seconds() - t0_cpu
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

        prof_path = out_dir / f"profile_{mode}.prof"
        prof.dump_stats(str(prof_path))

        hfun_path = out_dir / f"hfun_{mode}.2dm"
        try:
            Mesh(meshdata).write(
                str(hfun_path), overwrite=True, format="2dm")
            log.info(f"Hfun saved to {hfun_path}")
        except Exception as e:
            log.warning(f"Could not save hfun .2dm: {e}")

        if final_mesh is not None:
            final_path = out_dir / f"mesh_{mode}.2dm"
            try:
                final_mesh.write(
                    str(final_path), format="2dm", overwrite=True)
                log.info(f"Final mesh saved to {final_path}")
            except Exception as e:
                log.warning(f"Could not save mesh .2dm: {e}")

        sio = io.StringIO()
        ps  = pstats.Stats(prof, stream=sio)
        ps.sort_stats("cumulative")
        ps.print_stats(20)
        log.info(f"\ncProfile top-20 ({mode}):\n{sio.getvalue()}")

    except Exception as exc:
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
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(__file__).parent / "dem_manifest.json",
    )
    parser.add_argument("--shapefile", type=Path, default=None)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("./benchmark_results"),
    )
    parser.add_argument(
        "--nprocs", type=int,
        default=max(os.cpu_count() or 1, 1),
    )
    parser.add_argument(
        "--modes", nargs="+", choices=_ALL_MODES,
        default=_ALL_MODES,
    )

    # ── Standard recipe flags ─────────────────────────────────────────────
    parser.add_argument("--light-features",      action="store_true")
    parser.add_argument("--skip-topofunc",        action="store_true")
    parser.add_argument("--skip-constraints",     action="store_true")
    parser.add_argument("--skip-box-refinements", action="store_true")
    parser.add_argument("--all-fast-refinements", action="store_true")

    # ── Config F-anas / G (Anas PR benchmarks) ────────────────────────────
    parser.add_argument(
        "--config-f", action="store_true",
        help="Config F-anas: Anas PR benchmark ops, CUDEM tiles only.",
    )
    parser.add_argument(
        "--config-g", action="store_true",
        help="Config G: Config F-anas + patch + feature.",
    )

    # ── Config R and isolation runs (full STOFS domain, 451 tiles) ────────
    parser.add_argument("--config-r",  action="store_true",
        help="Config R: full production recipe, 451 tiles.")
    parser.add_argument("--config-r0", action="store_true",
        help="Config R0: no refinements.")
    parser.add_argument("--config-r1", action="store_true",
        help="Config R1: add_constant_value only.")
    parser.add_argument("--config-r2", action="store_true",
        help="Config R2: add_topo_bound_constraint only.")
    parser.add_argument("--config-r3", action="store_true",
        help="Config R3: add_contour only.")
    parser.add_argument("--config-r4", action="store_true",
        help="Config R4: add_subtidal_flow_limiter only.")
    parser.add_argument("--config-r5", action="store_true",
        help="Config R5: add_channel only.")

    # ── Config F-fat and isolation runs (MA/NH/ME, 44 tiles, 2x finer) ───
    parser.add_argument(
        "--config-ffat", action="store_true",
        help=(
            "Config F-fat: 2x finer production recipe on MA/NH/ME region. "
            "shelf=2km, contour=1km, flow_hmin=500m, channel=500m. "
            "Use with dem_manifest_config_f.json (44 tiles, 45 ranks)."
        ),
    )
    parser.add_argument("--config-f0", action="store_true",
        help="Config F0: no refinements (F-fat domain).")
    parser.add_argument("--config-f1", action="store_true",
        help="Config F1: add_constant_value only (F-fat).")
    parser.add_argument("--config-f2", action="store_true",
        help="Config F2: add_topo_bound_constraint only (F-fat, value=2km).")
    parser.add_argument("--config-f3", action="store_true",
        help="Config F3: add_contour only (F-fat, size=1km).")
    parser.add_argument("--config-f4", action="store_true",
        help="Config F4: add_subtidal_flow_limiter only (F-fat, hmin=500m).")
    parser.add_argument("--config-f5", action="store_true",
        help="Config F5: add_channel only (F-fat, size=500m).")

    parser.add_argument("--full-pipeline", action="store_true")
    parser.add_argument("--hmin", type=float, default=GLOBAL_HMIN)
    parser.add_argument("--hmax", type=float, default=GLOBAL_HMAX)
    args = parser.parse_args()

    # Mutual exclusion check
    active_configs = [f for f in _CONFIG_FLAGS if getattr(args, f)]
    if len(active_configs) > 1:
        if _IS_MANAGER:
            _logger.error(
                f"Mutually exclusive config flags: {active_configs}. "
                "Pass only one."
            )
        sys.exit(1)

    recipe.GLOBAL_HMIN = args.hmin
    recipe.GLOBAL_HMAX = args.hmax

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
        _logger.info(f"Active config     : "
                     f"{active_configs[0] if active_configs else 'standard'}")
        _logger.info(f"full_pipeline     : {args.full_pipeline}")
        _logger.info(f"MPI active        : {_MPI_ACTIVE}  (size={_SIZE})")
        _logger.info(
            f"hmin={recipe.GLOBAL_HMIN} m  hmax={recipe.GLOBAL_HMAX} m"
        )
        if "mpi_hybrid" in args.modes and _MPI_ACTIVE:
            _logger.info(
                f"mpi_hybrid auto cores/rank : {_plan_hybrid_cores(comm)}"
            )

    if _IS_MANAGER:
        try:
            manifest = _load_manifest(args.manifest)
        except FileNotFoundError:
            _logger.error(f"Manifest not found: {args.manifest}")
            sys.exit(1)
        domain_shape = _load_domain_shape(args.shapefile)
    else:
        manifest     = None
        domain_shape = None

    if _MPI_ACTIVE:
        from mpi4py import MPI
        comm         = MPI.COMM_WORLD
        manifest     = comm.bcast(manifest,     root=0)
        domain_shape = comm.bcast(domain_shape, root=0)

    all_results:   List[Dict]            = []
    stored_values: Dict[str, np.ndarray] = {}

    for mode in args.modes:
        if mode in _MPI_MODES and not _MPI_ACTIVE:
            if _IS_MANAGER:
                _logger.warning(
                    f"Skipping {mode} — not running under mpiexec/srun."
                )
            continue
        if mode in _RANK0_ONLY_MODES and not _IS_MANAGER:
            continue

        result, vals = _run_mode(
            manifest, domain_shape, args.nprocs, mode,
            args.out_dir, comm=comm,
            light_features=args.light_features,
            skip_topofunc=args.skip_topofunc,
            skip_constraints=args.skip_constraints,
            skip_box_refinements=args.skip_box_refinements,
            all_fast_refinements=args.all_fast_refinements,
            config_f=args.config_f,
            config_g=args.config_g,
            config_r=args.config_r,
            config_r0=args.config_r0,
            config_r1=args.config_r1,
            config_r2=args.config_r2,
            config_r3=args.config_r3,
            config_r4=args.config_r4,
            config_r5=args.config_r5,
            config_ffat=args.config_ffat,
            config_f0=args.config_f0,
            config_f1=args.config_f1,
            config_f2=args.config_f2,
            config_f3=args.config_f3,
            config_f4=args.config_f4,
            config_f5=args.config_f5,
            full_pipeline=args.full_pipeline,
        )
        if result:
            all_results.append(result)
        if vals is not None:
            stored_values[mode] = vals

    correctness_checks = []
    if _IS_MANAGER and len(stored_values) > 1:
        baseline_mode = next(
            (m for m in ("serial_mp", "serial_true", "parallel")
             if m in stored_values),
            next(iter(stored_values), None),
        )
        if baseline_mode:
            correctness_checks = _check_correctness(
                stored_values[baseline_mode], baseline_mode,
                all_results, stored_values,
            )
            _logger.info("\n=== Correctness Check ===")
            for c in correctness_checks:
                status = "OK  " if c["match"] else "FAIL"
                _logger.info(
                    f"  {status}  {baseline_mode} vs {c['mode']}: "
                    f"{c['reason']}"
                )

    if _IS_MANAGER and all_results:
        baseline_time = next(
            (r["wall_time_s"] for r in all_results
             if r["mode"] == "serial_mp" and r["status"] == "success"),
            None,
        )
        if baseline_time is None:
            baseline_time = next(
                (r["wall_time_s"] for r in all_results
                 if r["mode"] == "serial_true"
                 and r["status"] == "success"),
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
            "active_config":      (active_configs[0]
                                   if active_configs else "standard"),
            "n_dems":             sum(
                1 for v in manifest.values() if v.get("available")
            ),
            "correctness_checks": correctness_checks,
            "results":            all_results,
        }
        out_json = args.out_dir / "benchmark_results.json"
        out_json.write_text(json.dumps(summary, indent=2))
        _logger.info(f"\nResults written to {out_json}")

        _logger.info("\n" + "=" * 90)
        _logger.info("  BENCHMARK SUMMARY")
        _logger.info("=" * 90)
        _logger.info(
            f"  {'Mode':<14} {'nprocs':>6}  {'Status':<10} "
            f"{'Time (s)':>10}  {'Speedup':>9}  {'CPU (s)':>9}  "
            f"{'BusyCores':>10}  {'Util%':>6}  {'Nodes':>10}"
        )
        _logger.info("  " + "-" * 88)
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

        if correctness_checks:
            _logger.info("\n  CORRECTNESS vs serial_mp:")
            for c in correctness_checks:
                icon = "OK  " if c["match"] else "FAIL"
                _logger.info(f"    {icon}  {c['mode']}: {c['reason']}")
        _logger.info("=" * 90)


if __name__ == "__main__":
    main()
