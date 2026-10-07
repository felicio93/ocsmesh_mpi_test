"""NJ_test/build_nj_mesh.py

Build a New Jersey coastal mesh — one test, one mode, one method.

Usage
-----
    python build_nj_mesh.py --test 0 --mode serial   --method fast
    python build_nj_mesh.py --test 0 --mode serial   --method exact
    python build_nj_mesh.py --test 4 --mode parallel --method exact

    srun --mpi=pmi2 -n 43 python build_nj_mesh.py \\
        --test 4 --mode mpi_no_pool --method exact

    python build_nj_mesh.py --list-tests
"""

from __future__ import annotations

import os as _os
import tempfile as _tempfile
_shared_tmpdir = _os.environ.get("OCSMESH_SHARED_TMPDIR", "")
if _shared_tmpdir:
    _os.makedirs(_shared_tmpdir, exist_ok=True)
    _os.environ["TMPDIR"] = _shared_tmpdir
    _tempfile.tempdir = _shared_tmpdir

import argparse
import json
import logging
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import geopandas as gpd

import ocsmesh  # noqa: F401
from ocsmesh import Geom, Hfun, Mesh, MeshDriver
from ocsmesh.mpi import _is_mpi_env_detected, _is_mpi_active

from nj_mesh_config import (
    DOMAIN_SHP, MANIFEST_PATH, RESULTS_ROOT,
    HMIN, HMAX,
    TESTS, METHODS,
    DEPTH_BANDS,
    CONTOUR_LEVEL, CONTOUR_EXPANSION_RATE, CONTOUR_TARGET_SIZE,
    FLOW_LIMITERS,
    CHANNELS,
    MESH_ENGINE, MESH_BND_REPR,
    PARALLEL_NPROCS, SERIAL_NPROCS,
    MPI_HYBRID_CORES,
)

# ---------------------------------------------------------------------------
# MPI bootstrap
# ---------------------------------------------------------------------------
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

logging.basicConfig(
    level=logging.INFO if _IS_MANAGER else logging.WARNING,
    format=f"[rank {_RANK}] %(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
_logger = logging.getLogger("nj_mesh")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> Dict:
    with open(path) as fh:
        return json.load(fh)


def get_raster_paths(manifest: Dict) -> List[str]:
    entries = [v for v in manifest.values() if v.get("available")]
    entries.sort(key=lambda e: e["priority"])
    paths = []
    for e in entries:
        p = Path(e["path"])
        if p.exists():
            paths.append(str(p))
        else:
            _logger.warning(f"Missing: {p}")
    _logger.info(
        f"Rasters: {len(paths)} "
        f"(gebco={sum(1 for e in entries if e['source']=='gebco')}, "
        f"third={sum(1 for e in entries if e['source']=='ncei_third')}, "
        f"ninth={sum(1 for e in entries if e['source']=='ncei_ninth')})"
    )
    return paths


def load_domain(shp_path: Path):
    gdf = gpd.read_file(shp_path)
    if gdf.crs and not gdf.crs.equals("EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")
    domain_shape = gdf.union_all()
    _logger.info(
        f"Domain: {shp_path.name}  "
        f"bounds={gdf.total_bounds.round(4).tolist()}"
    )
    return domain_shape, gdf.crs


def _resolve_nprocs(mode: str) -> int:
    if mode == "serial":        return SERIAL_NPROCS
    if mode == "parallel":      return PARALLEL_NPROCS
    if mode == "mpi_no_pool":   return 1
    if mode == "mpi_hybrid":    return MPI_HYBRID_CORES
    return 1


def _ocsmesh_mode(mode: str) -> str:
    if mode in ("mpi_no_pool", "mpi_hybrid"):
        return "mpi"
    return mode


# ---------------------------------------------------------------------------
# Geom
# ---------------------------------------------------------------------------

def build_geom(raster_paths, domain_shape, domain_crs, nprocs) -> Geom:
    _logger.info(f"Building Geom (nprocs={nprocs}) ...")
    t0 = time.perf_counter()
    geom = Geom(
        raster_paths,
        base_shape=domain_shape,
        base_shape_crs=domain_crs,
        zmin=-10000.0,
        zmax=10.0,
        nprocs=nprocs,
    )
    _logger.info(f"Geom done in {time.perf_counter()-t0:.1f}s")
    return geom


# ---------------------------------------------------------------------------
# Hfun
# ---------------------------------------------------------------------------

def build_hfun(
    raster_paths, domain_shape, domain_crs,
    nprocs, execution_mode, method, test_cfg,
) -> Hfun:
    _logger.info(
        f"Building Hfun  mode={execution_mode}  "
        f"method={method}  nprocs={nprocs}"
    )

    hfun = Hfun(
        raster_paths,
        hmin=HMIN,
        hmax=HMAX,
        nprocs=nprocs,
        base_shape=domain_shape,
        base_shape_crs=domain_crs,
        method=method,
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        hfun.execution_mode = execution_mode

    # Depth bands
    if test_cfg["depth_bands"]:
        _logger.info("  Depth bands ...")
        for value, lower, upper, desc in DEPTH_BANDS:
            _logger.info(f"    {desc}")
            kwargs = {"value": value}
            if lower is not None: kwargs["lower_bound"] = lower
            if upper is not None: kwargs["upper_bound"] = upper
            hfun.add_constant_value(**kwargs)

    # Contour
    if test_cfg["contour"]:
        _logger.info(
            f"  Contour: level={CONTOUR_LEVEL}m  "
            f"rate={CONTOUR_EXPANSION_RATE}  target={CONTOUR_TARGET_SIZE}m"
        )
        hfun.add_contour(
            level=CONTOUR_LEVEL,
            expansion_rate=CONTOUR_EXPANSION_RATE,
            target_size=CONTOUR_TARGET_SIZE,
        )

    # Flow limiters
    if test_cfg["flow_limiter"]:
        _logger.info("  Flow limiters ...")
        for hmin_fl, hmax_fl, lower, upper, desc in FLOW_LIMITERS:
            _logger.info(f"    {desc}")
            hfun.add_subtidal_flow_limiter(
                hmin=hmin_fl,
                hmax=hmax_fl,
                lower_bound=lower,
                upper_bound=upper,
            )

    # Channels
    if test_cfg["channels"]:
        _logger.info("  Channels ...")
        for width, target, rate, desc in CHANNELS:
            _logger.info(f"    {desc}")
            hfun.add_channel(
                level=0.0,
                width=width,
                target_size=target,
                expansion_rate=rate,
            )

    _logger.info("Hfun refinements done.")
    return hfun


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_test(
    test_id: int,
    mode: str,
    method: str,
    manifest_path: Path,
    out_dir: Path,
) -> Dict:

    test_cfg = next((t for t in TESTS if t["id"] == test_id), None)
    if test_cfg is None:
        raise ValueError(f"Unknown test id: {test_id}")

    out_dir.mkdir(parents=True, exist_ok=True)
    nprocs       = _resolve_nprocs(mode)
    ocsmesh_mode = _ocsmesh_mode(mode)

    _logger.info("=" * 65)
    _logger.info(
        f"  TEST {test_id}: {test_cfg['name']}  |  "
        f"mode={mode}  |  method={method}"
    )
    _logger.info(f"  {test_cfg['description']}")
    _logger.info(f"  output → {out_dir}")
    _logger.info("=" * 65)

    summary = {
        "test_id":       test_id,
        "test_name":     test_cfg["name"],
        "mode":          mode,
        "method":        method,
        "ocsmesh_mode":  ocsmesh_mode,
        "nprocs":        nprocs,
        "mpi_size":      _SIZE,
        "hmin":          HMIN,
        "hmax":          HMAX,
        "out_dir":       str(out_dir),
    }

    t_total = time.perf_counter()

    try:
        manifest     = load_manifest(manifest_path)
        raster_paths = get_raster_paths(manifest)
        domain_shape, domain_crs = load_domain(DOMAIN_SHP)
        summary["n_rasters"] = len(raster_paths)

        # Geom
        t0 = time.perf_counter()
        geom = build_geom(
            raster_paths, domain_shape, domain_crs, nprocs)
        summary["geom_time_s"] = round(time.perf_counter() - t0, 2)

        # Hfun
        t0 = time.perf_counter()
        hfun = build_hfun(
            raster_paths, domain_shape, domain_crs,
            nprocs, ocsmesh_mode, method, test_cfg,
        )
        summary["hfun_build_time_s"] = round(time.perf_counter() - t0, 2)

        # meshdata
        _logger.info("Computing meshdata ...")
        t0 = time.perf_counter()
        meshdata = hfun.meshdata()
        summary["meshdata_time_s"] = round(time.perf_counter() - t0, 2)
        _logger.info(f"meshdata done in {summary['meshdata_time_s']:.1f}s")

        # Worker ranks exit here
        if meshdata is None:
            summary["status"] = "worker_rank"
            summary["total_time_s"] = round(
                time.perf_counter() - t_total, 2)
            return summary

        # Save hfun
        hfun_path = (
            out_dir /
            f"hfun_t{test_id}_{test_cfg['name']}_{mode}_{method}.2dm"
        )
        try:
            Mesh(meshdata).write(
                str(hfun_path), overwrite=True, format="2dm")
            _logger.info(f"Hfun saved: {hfun_path.name}")
            summary["hfun_path"] = str(hfun_path)
        except Exception as e:
            _logger.warning(f"Could not save hfun: {e}")

        # Hfun stats
        try:
            vals = np.array(meshdata.value)
            summary["hfun_min"]   = float(np.nanmin(vals))
            summary["hfun_max"]   = float(np.nanmax(vals))
            summary["hfun_mean"]  = float(np.nanmean(vals))
            summary["hfun_nodes"] = int(meshdata.num_nodes)
        except Exception:
            pass

        # MeshDriver
        # Apply buffer(0) to fix any invalid geometries before meshing.
        # GEOSException TopologyException can occur when the size function
        # produces nearly-degenerate geometry near tile boundaries.
        _logger.info(
            f"Running MeshDriver "
            f"(engine={MESH_ENGINE}, bnd={MESH_BND_REPR}) ..."
        )
        try:
            from shapely.validation import make_valid
            geom_gs = geom.geoseries()
            fixed = geom_gs.copy()
            for i, g in enumerate(geom_gs):
                if not g.is_valid:
                    _logger.warning(f"Fixing invalid geometry at index {i}")
                    fixed.iloc[i] = make_valid(g)
            geom._geoseries = fixed
        except Exception as _gfix_e:
            _logger.warning(f"Geometry fix skipped: {_gfix_e}")
        t0 = time.perf_counter()
        driver = MeshDriver(
            geom, hfun,
            engine_name=MESH_ENGINE,
            bnd_representation=MESH_BND_REPR,
        )
        final_mesh = driver.run()
        summary["meshdriver_time_s"] = round(
            time.perf_counter() - t0, 2)
        _logger.info(
            f"MeshDriver done in {summary['meshdriver_time_s']:.1f}s"
        )

        # Save mesh
        mesh_path = (
            out_dir /
            f"mesh_t{test_id}_{test_cfg['name']}_{mode}_{method}.2dm"
        )
        final_mesh.write(
            str(mesh_path), format="2dm", overwrite=True)
        _logger.info(f"Mesh saved: {mesh_path.name}")
        summary["mesh_path"] = str(mesh_path)

        # Mesh stats
        try:
            n_nodes = final_mesh.msh_t.num_nodes
            n_tria  = len(final_mesh.msh_t.tria)
            summary["mesh_nodes"]     = int(n_nodes)
            summary["mesh_triangles"] = int(n_tria)
            _logger.info(
                f"Mesh: {n_nodes:,} nodes, {n_tria:,} triangles"
            )
        except Exception as e:
            _logger.warning(f"Could not get mesh stats: {e}")

        summary["status"] = "success"

    except Exception as exc:
        tb = traceback.format_exc()
        _logger.error(f"FAILED: {exc}\n{tb}")
        summary["status"]    = "failed"
        summary["error"]     = repr(exc)
        summary["traceback"] = tb

    summary["total_time_s"] = round(time.perf_counter() - t_total, 2)

    if _IS_MANAGER:
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        _logger.info(
            f"Total: {summary['total_time_s']:.1f}s  "
            f"status={summary['status']}"
        )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build NJ coastal mesh — one test, one mode, one method.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test",   type=int,   required=False, default=None)
    parser.add_argument(
        "--mode",
        choices=["serial", "parallel", "mpi_no_pool", "mpi_hybrid"],
        default="serial",
    )
    parser.add_argument(
        "--method",
        choices=["fast", "exact"],
        default="exact",
        help="Hfun sampling method (default: exact)",
    )
    parser.add_argument("--manifest",      type=Path, default=MANIFEST_PATH)
    parser.add_argument("--results-root",  type=Path, default=RESULTS_ROOT)
    parser.add_argument("--list-tests",    action="store_true")
    args = parser.parse_args()

    if args.list_tests:
        print(f"\n{'ID':<4} {'Name':<15} {'Description'}")
        print("-" * 65)
        for t in TESTS:
            flags = []
            if t["depth_bands"]:  flags.append("depth")
            if t["contour"]:      flags.append("contour")
            if t["flow_limiter"]: flags.append("flow")
            if t["channels"]:     flags.append("channels")
            print(
                f"{t['id']:<4} {t['name']:<15} {t['description']}"
            )
        print(f"\nMethods: {METHODS}")
        print()
        return

    if args.test is None:
        parser.error("--test required (use --list-tests to see options)")

    test_name = next(
        t["name"] for t in TESTS if t["id"] == args.test)
    out_dir = (
        args.results_root
        / f"test{args.test}_{test_name}"
        / args.mode
        / args.method
    )

    run_test(
        test_id=args.test,
        mode=args.mode,
        method=args.method,
        manifest_path=args.manifest.resolve(),
        out_dir=out_dir.resolve(),
    )


if __name__ == "__main__":
    main()
