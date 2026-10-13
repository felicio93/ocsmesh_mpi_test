"""Unalakleet_test/build_unalakleet_mesh.py

Build Unalakleet dual-resolution coastal mesh using OCSMesh.

Key design:
  - Geom uses domain.shp as authoritative boundary (no elevation clipping)
  - Outer refinements: entire domain (source_index=[0,1])
  - Inner refinements: USGS DEM only (source_index=[1]) → confined to highres area
  - Channel mask: add_patch with polygon shapefile

Usage
-----
    python build_unalakleet_mesh.py --test 0 --mode mpi_no_pool --method exact
    python build_unalakleet_mesh.py --list-tests
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
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import geopandas as gpd

import ocsmesh
from ocsmesh import Geom, Hfun, Mesh, MeshDriver
from ocsmesh.mpi import _is_mpi_env_detected, _is_mpi_active

from unalakleet_config import (
    DOMAIN_SHP, HIGHRES_SHP, CHANNEL_MASK_SHP,
    DEM_PATHS, RESULTS_ROOT,
    HMIN, HMAX,
    GEOM_ZMIN, GEOM_ZMAX,
    TESTS, METHODS,
    OUTER_DEPTH_BANDS,
    OUTER_CONTOUR_LEVEL, OUTER_CONTOUR_EXPANSION_RATE,
    OUTER_CONTOUR_TARGET_SIZE,
    INNER_DEPTH_BANDS,
    INNER_CONTOUR_LEVEL, INNER_CONTOUR_EXPANSION_RATE,
    INNER_CONTOUR_TARGET_SIZE,
    INNER_FLOW_LIMITER,
    CHANNEL_PATCH,
    MESH_ENGINE, MESH_BND_REPR,
    MPI_NOPOOL_NTASKS, MPI_NOPOOL_CPUS,
    PARALLEL_NPROCS, SERIAL_NPROCS,
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
_logger = logging.getLogger("unalakleet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_shape(shp_path: Path):
    """Load shapefile, reproject to EPSG:4326, return (geometry, crs)."""
    gdf = gpd.read_file(shp_path)
    if gdf.crs and not gdf.crs.equals("EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")
    return gdf.union_all(), gdf.crs


def get_all_indices() -> List[int]:
    """All DEM indices — used for outer (whole domain) refinements."""
    return list(range(len(DEM_PATHS)))


def get_usgs_index() -> List[int]:
    """USGS DEM index only — used for inner (highres area) refinements.
    
    Using source_index=[1] (USGS only) effectively restricts the refinement
    to the area covered by the USGS DEM, which corresponds to the inner
    highres_domain area. The GEBCO DEM covers the full outer domain.
    """
    return [len(DEM_PATHS) - 1]   # index 1 — USGS is last/highest priority


def _resolve_nprocs(mode: str) -> int:
    if mode == "serial":      return SERIAL_NPROCS
    if mode == "parallel":    return PARALLEL_NPROCS
    if mode == "mpi_no_pool": return 1
    return 1


def _ocsmesh_mode(mode: str) -> str:
    if mode == "mpi_no_pool": return "mpi"
    return mode


# ---------------------------------------------------------------------------
# Geom — uses domain.shp as authoritative boundary
# ---------------------------------------------------------------------------

def build_geom(nprocs: int) -> Geom:
    """Build domain geometry.
    
    CRITICAL: zmax=99999 ensures domain.shp is the authoritative boundary.
    Without this, zmax clips land areas above the threshold, causing the
    mesh to fall short of the domain shapefile extent.
    """
    _logger.info(f"Building Geom (nprocs={nprocs}) ...")
    _logger.info(f"  Using domain.shp as authoritative boundary")
    _logger.info(f"  zmin={GEOM_ZMIN}  zmax={GEOM_ZMAX} (no elevation clipping)")
    t0 = time.perf_counter()
    domain_shape, domain_crs = load_shape(DOMAIN_SHP)
    geom = Geom(
        DEM_PATHS,
        base_shape=domain_shape,
        base_shape_crs=domain_crs,
        zmin=GEOM_ZMIN,
        zmax=GEOM_ZMAX,
        nprocs=nprocs,
    )
    _logger.info(f"Geom done in {time.perf_counter()-t0:.1f}s")
    return geom


# ---------------------------------------------------------------------------
# Hfun
# ---------------------------------------------------------------------------

def build_hfun(
    nprocs: int,
    ocsmesh_mode: str,
    method: str,
    test_cfg: Dict,
) -> Hfun:
    """Build size function with dual-resolution refinements.
    
    Outer refinements use source_index=[0,1] (all DEMs — full domain).
    Inner refinements use source_index=[1] (USGS only — highres area).
    This naturally restricts inner refinements to the USGS DEM footprint
    which corresponds to the highres_domain area.
    """
    _logger.info(
        f"Building Hfun  mode={ocsmesh_mode}  "
        f"method={method}  nprocs={nprocs}"
    )

    domain_shape, domain_crs = load_shape(DOMAIN_SHP)

    _logger.info("── VERIFYING DEM PATHS ──")
    for dem_path in DEM_PATHS:
        p = Path(dem_path)
        if p.exists():
            _logger.info(f"  Path: {p.resolve()}")
            _logger.info(f"  Size: {p.stat().st_size / 1e6:.2f} MB")
        else:
            _logger.error(f"  Path not found: {dem_path}")
    _logger.info("───────────────────────────")

    hfun = Hfun(
        DEM_PATHS,
        hmin=HMIN,
        hmax=HMAX,
        nprocs=nprocs,
        base_shape=domain_shape,
        base_shape_crs=domain_crs,
        method=method,
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        hfun.execution_mode = ocsmesh_mode

    all_idx  = get_all_indices()   # [0, 1] — full domain
    usgs_idx = get_usgs_index()    # [1]    — highres area only

    # ── t1: Outer depth bands (entire domain, both DEMs) ──────────────────
    if test_cfg["outer_depth"]:
        _logger.info("  t1: Outer depth bands (full domain) ...")
        for value, lower, upper, desc in OUTER_DEPTH_BANDS:
            _logger.info(f"    {desc}")
            kwargs = {"value": value, "source_index": all_idx}
            if lower is not None: kwargs["lower_bound"] = lower
            if upper is not None: kwargs["upper_bound"] = upper
            hfun.add_constant_value(**kwargs)

    # ── t2: Outer contour (entire domain) ─────────────────────────────────
    if test_cfg["outer_contour"]:
        _logger.info(
            f"  t2: Outer contour  level={OUTER_CONTOUR_LEVEL}m  "
            f"target={OUTER_CONTOUR_TARGET_SIZE}m  "
            f"rate={OUTER_CONTOUR_EXPANSION_RATE}"
        )
        hfun.add_contour(
            level=OUTER_CONTOUR_LEVEL,
            expansion_rate=OUTER_CONTOUR_EXPANSION_RATE,
            target_size=OUTER_CONTOUR_TARGET_SIZE,
        )

    # ── t3: Inner depth bands (USGS DEM only → restricted to highres area) ─
    if test_cfg["inner_depth"]:
        _logger.info(
            "  t3: Inner depth bands "
            "(source_index=[1] = USGS DEM → highres area only) ..."
        )
        for value, lower, upper, desc in INNER_DEPTH_BANDS:
            _logger.info(f"    {desc}")
            kwargs = {"value": value, "source_index": usgs_idx}
            if lower is not None: kwargs["lower_bound"] = lower
            if upper is not None: kwargs["upper_bound"] = upper
            hfun.add_constant_value(**kwargs)

    # ── t4: Inner contour (USGS DEM only → restricted to highres area) ────
    # Using source_index restricts this contour to the USGS DEM footprint.
    # Outside the highres area the outer contour (t2) already applies.
    if test_cfg["inner_contour"]:
        _logger.info(
            f"  t4: Inner contour  level={INNER_CONTOUR_LEVEL}m  "
            f"target={INNER_CONTOUR_TARGET_SIZE}m  "
            f"rate={INNER_CONTOUR_EXPANSION_RATE}  "
            f"(source_index=[1] = USGS DEM → highres area only)"
        )
        hfun.add_contour(
            level=INNER_CONTOUR_LEVEL,
            expansion_rate=INNER_CONTOUR_EXPANSION_RATE,
            target_size=INNER_CONTOUR_TARGET_SIZE,
            source_index=usgs_idx,
        )

    # ── t5: Inner flow limiter (USGS DEM only → restricted to highres area) ─
    if test_cfg["flow_limiter"]:
        fl = INNER_FLOW_LIMITER
        _logger.info(
            f"  t5: {fl['description']}  "
            f"(source_index=[1] = USGS DEM → highres area only)"
        )
        hfun.add_subtidal_flow_limiter(
            hmin=fl["hmin"],
            hmax=fl["hmax"],
            lower_bound=fl["lower_bound"],
            upper_bound=fl["upper_bound"],
            source_index=usgs_idx,
        )

    # ── t6: Channel mask patch (polygon shapefile) ────────────────────────
    if test_cfg["channels"]:
        cp = CHANNEL_PATCH
        _logger.info(f"  t6: {cp['description']}")
        channel_shape, _ = load_shape(CHANNEL_MASK_SHP)
        hfun.add_patch(
            shape=channel_shape,
            expansion_rate=cp["expansion_rate"],
            target_size=cp["target_size"],
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
    out_dir: Path,
) -> Dict:

    test_cfg = next((t for t in TESTS if t["id"] == test_id), None)
    if test_cfg is None:
        raise ValueError(f"Unknown test id: {test_id}")

    out_dir.mkdir(parents=True, exist_ok=True)
    nprocs       = _resolve_nprocs(mode)
    ocsmesh_mode = _ocsmesh_mode(mode)

    if _IS_MANAGER:
        _logger.info("=" * 65)
        _logger.info(
            f"  TEST {test_id}: {test_cfg['name']}  |  "
            f"mode={mode}  |  method={method}"
        )
        _logger.info(f"  {test_cfg['description']}")
        _logger.info(f"  output → {out_dir}")
        _logger.info("=" * 65)

    summary = {
        "test_id":      test_id,
        "test_name":    test_cfg["name"],
        "mode":         mode,
        "method":       method,
        "ocsmesh_mode": ocsmesh_mode,
        "nprocs":       nprocs,
        "mpi_size":     _SIZE,
        "hmin":         HMIN,
        "hmax":         HMAX,
        "out_dir":      str(out_dir),
    }

    t_total = time.perf_counter()

    try:
        # Geom
        t0 = time.perf_counter()
        geom = build_geom(nprocs)
        summary["geom_time_s"] = round(time.perf_counter() - t0, 2)

        # Hfun
        t0 = time.perf_counter()
        hfun = build_hfun(nprocs, ocsmesh_mode, method, test_cfg)
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

        try:
            summary["hfun_nodes"] = int(meshdata.num_nodes)
        except Exception:
            pass

        # MeshDriver — uses domain.shp via geom object
        _logger.info(
            f"Running MeshDriver "
            f"(engine={MESH_ENGINE}, bnd={MESH_BND_REPR}) ..."
        )
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
        description="Build Unalakleet dual-resolution coastal mesh.",
    )
    parser.add_argument("--test",   type=int,  required=False, default=None)
    parser.add_argument(
        "--mode",
        choices=["serial", "parallel", "mpi_no_pool"],
        default="mpi_no_pool",
    )
    parser.add_argument(
        "--method",
        choices=["fast", "exact"],
        default="exact",
    )
    parser.add_argument(
        "--results-root", type=Path, default=RESULTS_ROOT,
    )
    parser.add_argument("--list-tests", action="store_true")
    args = parser.parse_args()

    if args.list_tests:
        print(f"\n{'ID':<4} {'Name':<18} {'Description'}")
        print("-" * 72)
        for t in TESTS:
            print(f"{t['id']:<4} {t['name']:<18} {t['description']}")
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
        out_dir=out_dir.resolve(),
    )


if __name__ == "__main__":
    main()
