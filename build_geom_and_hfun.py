"""Build the STOFS-3D-Atlantic Geom and Hfun for the OCSMesh MPI benchmark.

This is the core "recipe" module. It is imported by run_benchmark.py, but
can also be run standalone to inspect how the refinements resolve to the
rasters before launching a full meshdata() computation.

Refinement strategies
=====================

Config F (maps to Anas PR Config E)
-------------------------------------
All ops MPI-dispatched, no shape bottleneck:
    add_subtidal_flow_limiter  — all CUDEM tiles
    add_constant_value         — all CUDEM tiles
    add_topo_bound_constraint  — all CUDEM tiles
    add_topo_func_constraint   — all CUDEM tiles
    add_contour (0m + -200m)   — all tiles
    add_channel                — all tiles

Config G (maps to Anas PR Config F)
-------------------------------------
Config F + shape-based refinements:
    add_patch  (BOX2: SC/GA coast + BOX3: Gulf Coast)
    add_feature (line features at BOX2 + BOX3)

Config R (real-world production recipe)
-----------------------------------------
Realistic storm surge mesh size function recipe:
    add_constant_value         — all tiles, value=7km, upper_bound=-2000m
    add_topo_bound_constraint  — all tiles, value=4km, -2000..-200m, min
    add_contour                — all tiles, level=0m, rate=0.01, size=2km
    add_subtidal_flow_limiter  — all tiles, hmin=1km, hmax=7km
    add_channel                — all tiles, width=2km, size=1km, rate=0.01

Config R0-R5 (isolation runs)
------------------------------
Each applies only one of the Config R operations to isolate its effect:
    R0 — no refinements (flat background)
    R1 — add_constant_value only
    R2 — add_topo_bound_constraint only
    R3 — add_contour only
    R4 — add_subtidal_flow_limiter only
    R5 — add_channel only

Standard recipe (Configs A-E)
------------------------------
Index-modulo scheme assigning different refinements to different
CUDEM tiles by position. Used for the original cost-ladder benchmarks.

Execution mode translation
--------------------------
OCSMesh accepts: 'serial', 'parallel', 'mpi'.
mpi_no_pool -> execution_mode='mpi', nprocs=1 per rank
mpi_hybrid  -> execution_mode='mpi', nprocs=auto per rank
Translation is done by run_benchmark.py before calling build_hfun().
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from shapely.geometry import box, LineString

from ocsmesh import Geom, Hfun

_logger = logging.getLogger("stofs_benchmark.build")

# ---------------------------------------------------------------------------
# Global mesh size bounds (metres)
# ---------------------------------------------------------------------------
GLOBAL_HMIN = 1000.0
GLOBAL_HMAX = 7000.0
EXPANSION_RATE = 0.15     # standard A-E configs transition zone (~8 km)

# Config R specific parameters
R_HMIN            = 1000.0   # 1 km finest
R_HMAX            = 7000.0   # 7 km coarsest
R_OCEAN_VALUE     = 7000.0   # open ocean constant resolution (m)
R_OCEAN_DEPTH     = -2000.0  # depth threshold for open ocean (m)
R_SHELF_VALUE     = 4000.0   # shelf minimum resolution (m)
R_SHELF_UPPER     = -200.0   # shelf upper bound (m)
R_SHELF_LOWER     = -2000.0  # shelf lower bound (m)
R_CONTOUR_SIZE    = 2000.0   # shoreline target resolution (m)
R_CONTOUR_RATE    = 0.01     # shoreline expansion rate (~250 km transition)
R_CHANNEL_WIDTH   = 2000.0   # channel detection width (m)
R_CHANNEL_SIZE    = 1000.0   # channel target resolution (m)
R_CHANNEL_RATE    = 0.01     # channel expansion rate

# ---------------------------------------------------------------------------
# Index-modulo scheme (standard A-E configs)
# ---------------------------------------------------------------------------
MODULO_STRIDE = 6

# ---------------------------------------------------------------------------
# Fixed lat/lon boxes for shape-based refinements (Configs F, G)
# (lon_min, lat_min, lon_max, lat_max) in EPSG:4326
# ---------------------------------------------------------------------------
BOX1 = (-85.0, 25.0, -82.0, 31.0)   # West Florida shelf
BOX2 = (-80.0, 31.0, -77.0, 35.0)   # SC/GA coast
BOX3 = (-90.0, 28.0, -86.0, 31.0)   # Gulf Coast


# ---------------------------------------------------------------------------
# Module-level helpers for topo_func_constraint
# ---------------------------------------------------------------------------
def _half_depth(depth: np.ndarray) -> np.ndarray:
    """Mesh size = |depth| / 2, used by topo_func_constraint."""
    return np.abs(depth) / 2.0


# ---------------------------------------------------------------------------
# Manifest / raster helpers
# ---------------------------------------------------------------------------

def load_ordered_rasters(manifest: Dict) -> Tuple[List[str], List[Dict]]:
    """Return raster paths + metadata sorted by priority (GEBCO first)."""
    ordered = sorted(
        manifest.items(),
        key=lambda kv: kv[1].get("priority", 99),
    )
    paths: List[str] = []
    metas: List[Dict] = []
    for _name, meta in ordered:
        if not meta.get("available") or not meta.get("path"):
            continue
        if not Path(meta["path"]).exists():
            continue
        paths.append(str(meta["path"]))
        metas.append(meta)
    return paths, metas


def _cudem_indices_by_class(metas: List[Dict]) -> Dict[int, List[int]]:
    """Group CUDEM raster-list indices by modulo class."""
    classes: Dict[int, List[int]] = {c: [] for c in range(MODULO_STRIDE)}
    cudem_pos = 0
    for i, meta in enumerate(metas):
        if meta.get("source") == "gebco":
            continue
        cls = cudem_pos % MODULO_STRIDE
        classes[cls].append(i)
        cudem_pos += 1
    return classes


def _all_indices(metas: List[Dict]) -> List[int]:
    """Return indices of all tiles (GEBCO + CUDEM)."""
    return list(range(len(metas)))


def _cudem_only_indices(metas: List[Dict]) -> List[int]:
    """Return indices of CUDEM tiles only."""
    return [i for i, m in enumerate(metas) if m.get("source") != "gebco"]


# ---------------------------------------------------------------------------
# Geom builder
# ---------------------------------------------------------------------------

def build_geom(raster_paths: List[str], domain_shape, nprocs: int):
    """Build a GeomCollector clipped to the STOFS-3D-Atlantic domain."""
    _logger.info(
        f"Building Geom from {len(raster_paths)} DEMs (nprocs={nprocs})"
    )
    geom = Geom(
        raster_paths,
        base_shape=domain_shape,
        base_shape_crs="EPSG:4326",
        zmin=-11000.0,
        zmax=10.0,
        nprocs=nprocs,
    )
    return geom


# ---------------------------------------------------------------------------
# Hfun builder
# ---------------------------------------------------------------------------

def build_hfun(
    raster_paths: List[str],
    raster_metas: List[Dict],
    domain_shape,
    nprocs: int,
    execution_mode: str,
    # Standard A-E flags
    light_features: bool = False,
    skip_topofunc: bool = False,
    skip_constraints: bool = False,
    skip_box_refinements: bool = False,
    all_fast_refinements: bool = False,
    # Config F / G
    config_f: bool = False,
    config_g: bool = False,
    # Config R and isolation runs
    config_r: bool = False,
    config_r0: bool = False,
    config_r1: bool = False,
    config_r2: bool = False,
    config_r3: bool = False,
    config_r4: bool = False,
    config_r5: bool = False,
):
    """Build an HfunCollector and apply all refinements.

    Parameters
    ----------
    raster_paths : list of str
        DEM paths in ascending-priority order (GEBCO first).
    raster_metas : list of dict
        Parallel metadata for each raster.
    domain_shape : Polygon or MultiPolygon
        Domain boundary.
    nprocs : int
        Worker count per rank.
    execution_mode : str
        One of 'serial', 'parallel', 'mpi'.
    config_r : bool
        Full production recipe (all 5 Config R operations).
    config_r0 : bool
        Isolation: no refinements (flat background).
    config_r1 : bool
        Isolation: add_constant_value only (open ocean 7km).
    config_r2 : bool
        Isolation: add_topo_bound_constraint only (shelf 4km).
    config_r3 : bool
        Isolation: add_contour only (shoreline 2km).
    config_r4 : bool
        Isolation: add_subtidal_flow_limiter only (1km).
    config_r5 : bool
        Isolation: add_channel only (2km wide, 1km).

    Returns
    -------
    Hfun (HfunCollector)
    """
    _logger.info(
        f"Building Hfun from {len(raster_paths)} DEMs "
        f"(mode={execution_mode}, nprocs={nprocs}, "
        f"hmin={GLOBAL_HMIN}, hmax={GLOBAL_HMAX})"
    )

    # Determine hmin/hmax — Config R uses its own bounds
    is_r_config = any([
        config_r, config_r0, config_r1, config_r2,
        config_r3, config_r4, config_r5,
    ])
    hmin = R_HMIN if is_r_config else GLOBAL_HMIN
    hmax = R_HMAX if is_r_config else GLOBAL_HMAX

    hfun = Hfun(
        raster_paths,
        hmin=hmin,
        hmax=hmax,
        nprocs=nprocs,
        base_shape=domain_shape,
        base_shape_crs="EPSG:4326",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        hfun.execution_mode = execution_mode
    for w in caught:
        _logger.warning(
            "execution_mode setter [%s -> %s]: %s",
            execution_mode, hfun.execution_mode, str(w.message)
        )

    all_idx   = _all_indices(raster_metas)
    cudem_idx = _cudem_only_indices(raster_metas)

    # =========================================================================
    # Config R0 — flat background (no refinements)
    # =========================================================================
    if config_r0:
        _logger.info(
            "Config R0: flat background — no refinements applied. "
            f"hmin={hmin}m hmax={hmax}m"
        )
        return hfun

    # =========================================================================
    # Config R1 — open ocean constant 7km only
    # =========================================================================
    if config_r1:
        _logger.info(
            f"Config R1: add_constant_value only "
            f"(value={R_OCEAN_VALUE}m, upper_bound={R_OCEAN_DEPTH}m) "
            f"on all {len(all_idx)} tiles"
        )
        hfun.add_constant_value(
            value=R_OCEAN_VALUE,
            upper_bound=R_OCEAN_DEPTH,
            source_index=all_idx,
        )
        return hfun

    # =========================================================================
    # Config R2 — shelf constraint only
    # =========================================================================
    if config_r2:
        _logger.info(
            f"Config R2: add_topo_bound_constraint only "
            f"(value={R_SHELF_VALUE}m, "
            f"lower={R_SHELF_LOWER}m, upper={R_SHELF_UPPER}m, min) "
            f"on all {len(all_idx)} tiles"
        )
        hfun.add_topo_bound_constraint(
            value=R_SHELF_VALUE,
            lower_bound=R_SHELF_LOWER,
            upper_bound=R_SHELF_UPPER,
            value_type="min",
            source_index=all_idx,
        )
        return hfun

    # =========================================================================
    # Config R3 — shoreline contour only
    # =========================================================================
    if config_r3:
        _logger.info(
            f"Config R3: add_contour only "
            f"(level=0m, rate={R_CONTOUR_RATE}, size={R_CONTOUR_SIZE}m)"
        )
        hfun.add_contour(
            level=0.0,
            expansion_rate=R_CONTOUR_RATE,
            target_size=R_CONTOUR_SIZE,
        )
        return hfun

    # =========================================================================
    # Config R4 — subtidal flow limiter only
    # =========================================================================
    if config_r4:
        _logger.info(
            f"Config R4: add_subtidal_flow_limiter only "
            f"(hmin={R_HMIN}m, hmax={R_HMAX}m) "
            f"on all {len(all_idx)} tiles"
        )
        hfun.add_subtidal_flow_limiter(
            hmin=R_HMIN,
            hmax=R_HMAX,
            source_index=all_idx,
        )
        return hfun

    # =========================================================================
    # Config R5 — channel detection only
    # =========================================================================
    if config_r5:
        _logger.info(
            f"Config R5: add_channel only "
            f"(width={R_CHANNEL_WIDTH}m, size={R_CHANNEL_SIZE}m, "
            f"rate={R_CHANNEL_RATE})"
        )
        hfun.add_channel(
            level=0.0,
            width=R_CHANNEL_WIDTH,
            target_size=R_CHANNEL_SIZE,
            expansion_rate=R_CHANNEL_RATE,
        )
        return hfun

    # =========================================================================
    # Config R — full production recipe
    # =========================================================================
    if config_r:
        _logger.info(
            f"Config R: full production recipe on {len(all_idx)} tiles"
        )
        _logger.info(
            f"  1. add_constant_value     : value={R_OCEAN_VALUE}m, "
            f"upper_bound={R_OCEAN_DEPTH}m"
        )
        hfun.add_constant_value(
            value=R_OCEAN_VALUE,
            upper_bound=R_OCEAN_DEPTH,
            source_index=all_idx,
        )

        _logger.info(
            f"  2. add_topo_bound_constraint: value={R_SHELF_VALUE}m, "
            f"lower={R_SHELF_LOWER}m, upper={R_SHELF_UPPER}m, min"
        )
        hfun.add_topo_bound_constraint(
            value=R_SHELF_VALUE,
            lower_bound=R_SHELF_LOWER,
            upper_bound=R_SHELF_UPPER,
            value_type="min",
            source_index=all_idx,
        )

        _logger.info(
            f"  3. add_contour            : level=0m, "
            f"rate={R_CONTOUR_RATE}, size={R_CONTOUR_SIZE}m"
        )
        hfun.add_contour(
            level=0.0,
            expansion_rate=R_CONTOUR_RATE,
            target_size=R_CONTOUR_SIZE,
        )

        _logger.info(
            f"  4. add_subtidal_flow_limiter: "
            f"hmin={R_HMIN}m, hmax={R_HMAX}m"
        )
        hfun.add_subtidal_flow_limiter(
            hmin=R_HMIN,
            hmax=R_HMAX,
            source_index=all_idx,
        )

        _logger.info(
            f"  5. add_channel            : width={R_CHANNEL_WIDTH}m, "
            f"size={R_CHANNEL_SIZE}m, rate={R_CHANNEL_RATE}"
        )
        hfun.add_channel(
            level=0.0,
            width=R_CHANNEL_WIDTH,
            target_size=R_CHANNEL_SIZE,
            expansion_rate=R_CHANNEL_RATE,
        )

        _logger.info("Config R refinements applied.")
        return hfun

    # =========================================================================
    # Config F — all MPI-dispatched, no shapes
    # =========================================================================
    if config_f:
        _logger.info(
            "Config F: flow_limiter + const_value + constraints + "
            "contour/channel (all tiles, no boxes)"
        )
        hfun.add_subtidal_flow_limiter(
            hmin=GLOBAL_HMIN, hmax=GLOBAL_HMAX,
            lower_bound=-200.0, upper_bound=0.0,
            source_index=cudem_idx,
        )
        hfun.add_constant_value(
            value=1000.0, lower_bound=-5.0, upper_bound=0.0,
            source_index=cudem_idx,
        )
        hfun.add_topo_bound_constraint(
            value=1500.0, upper_bound=1.0, lower_bound=-2.0,
            value_type="min", rate=EXPANSION_RATE,
            source_index=cudem_idx,
        )
        hfun.add_topo_func_constraint(
            func=_half_depth,
            upper_bound=0.0, lower_bound=-3000.0,
            value_type="min", rate=EXPANSION_RATE,
            source_index=cudem_idx,
        )
        hfun.add_contour(
            level=[0.0, -200.0],
            expansion_rate=EXPANSION_RATE,
            target_size=3500.0,
        )
        hfun.add_channel(
            level=0.0, width=1000.0,
            target_size=3500.0,
            expansion_rate=EXPANSION_RATE,
        )
        _logger.info("Config F refinements applied.")
        return hfun

    # =========================================================================
    # Config G — Config F + shapes
    # =========================================================================
    if config_g:
        _logger.info(
            "Config G: Config F + patch + feature (BOX2 + BOX3)"
        )
        hfun.add_subtidal_flow_limiter(
            hmin=GLOBAL_HMIN, hmax=GLOBAL_HMAX,
            lower_bound=-200.0, upper_bound=0.0,
            source_index=cudem_idx,
        )
        hfun.add_constant_value(
            value=1000.0, lower_bound=-5.0, upper_bound=0.0,
            source_index=cudem_idx,
        )
        hfun.add_topo_bound_constraint(
            value=1500.0, upper_bound=1.0, lower_bound=-2.0,
            value_type="min", rate=EXPANSION_RATE,
            source_index=cudem_idx,
        )
        hfun.add_topo_func_constraint(
            func=_half_depth,
            upper_bound=0.0, lower_bound=-3000.0,
            value_type="min", rate=EXPANSION_RATE,
            source_index=cudem_idx,
        )
        hfun.add_contour(
            level=[0.0, -200.0],
            expansion_rate=EXPANSION_RATE,
            target_size=3500.0,
        )
        hfun.add_channel(
            level=0.0, width=1000.0,
            target_size=3500.0,
            expansion_rate=EXPANSION_RATE,
        )
        mid_lat2 = (BOX2[1] + BOX2[3]) / 2.0
        hfun.add_patch(
            shape=box(*BOX2),
            expansion_rate=EXPANSION_RATE,
            target_size=1000.0,
        )
        hfun.add_feature(
            shape=LineString([(BOX2[0], mid_lat2), (BOX2[2], mid_lat2)]),
            expansion_rate=EXPANSION_RATE,
            target_size=1000.0,
            crs=4326,
        )
        mid_lat3 = (BOX3[1] + BOX3[3]) / 2.0
        hfun.add_patch(
            shape=box(*BOX3),
            expansion_rate=EXPANSION_RATE,
            target_size=800.0,
        )
        hfun.add_feature(
            shape=LineString([(BOX3[0], mid_lat3), (BOX3[2], mid_lat3)]),
            expansion_rate=EXPANSION_RATE,
            target_size=800.0,
            crs=4326,
        )
        _logger.info("Config G refinements applied.")
        return hfun

    # =========================================================================
    # Standard A-E recipe: index-modulo + flags
    # =========================================================================
    classes     = _cudem_indices_by_class(raster_metas)
    flow_idx    = classes[0]
    const_idx   = classes[1]
    bound_idx   = classes[2]
    func_idx    = classes[3]
    courant_idx = classes[4]
    skip_idx    = classes[5]

    if all_fast_refinements:
        flow_idx  = list(cudem_idx)
        const_idx = list(cudem_idx)
        bound_idx = func_idx = courant_idx = []
        skip_idx  = []
        skip_constraints     = True
        skip_box_refinements = True
        light_features       = True
        _logger.info(
            f"  all_fast_refinements=True -> flow_limiter + const_value "
            f"on ALL {len(cudem_idx)} CUDEM tiles"
        )

    _logger.info("  Index-modulo assignment (CUDEM tiles):")
    _logger.info(f"    flow_limiter : {len(flow_idx)} tiles")
    _logger.info(f"    const_value  : {len(const_idx)} tiles")
    _logger.info(f"    topo_bound   : {len(bound_idx)} tiles")
    _logger.info(f"    topo_func    : {len(func_idx)} tiles")
    _logger.info(f"    courant      : {len(courant_idx)} tiles")
    _logger.info(f"    skipped      : {len(skip_idx)} tiles")

    if flow_idx:
        hfun.add_subtidal_flow_limiter(
            hmin=GLOBAL_HMIN, hmax=GLOBAL_HMAX,
            lower_bound=-200.0, upper_bound=0.0,
            source_index=flow_idx,
        )
    if const_idx:
        hfun.add_constant_value(
            value=1000.0, lower_bound=-5.0, upper_bound=0.0,
            source_index=const_idx,
        )
    if bound_idx:
        if skip_constraints:
            _logger.info("  topo_bound SKIPPED")
        else:
            hfun.add_topo_bound_constraint(
                value=1500.0, upper_bound=1.0, lower_bound=-2.0,
                value_type="min", rate=EXPANSION_RATE,
                source_index=bound_idx,
            )
    if func_idx and not (skip_topofunc or skip_constraints):
        hfun.add_topo_func_constraint(
            func=_half_depth,
            upper_bound=0.0, lower_bound=-3000.0,
            value_type="min", rate=EXPANSION_RATE,
            source_index=func_idx,
        )
    elif func_idx and (skip_topofunc or skip_constraints):
        _logger.info("  topo_func SKIPPED")
    if courant_idx:
        if skip_constraints:
            _logger.info("  courant SKIPPED")
        else:
            hfun.add_courant_num_constraint(
                upper_bound=0.9,
                timestep=150.0,
                wave_amplitude=2.0,
                source_index=courant_idx,
            )
    if light_features:
        _logger.info("  add_contour / add_channel SKIPPED")
    else:
        hfun.add_contour(
            level=[0.0, -200.0],
            expansion_rate=EXPANSION_RATE,
            target_size=3500.0,
        )
        hfun.add_channel(
            level=0.0, width=1000.0,
            target_size=3500.0,
            expansion_rate=EXPANSION_RATE,
        )
    if skip_box_refinements:
        _logger.info("  Box refinements SKIPPED")
    else:
        hfun.add_region_constraint(
            value=3500.0,
            shape=box(*BOX1),
            crs="EPSG:4326",
            value_type="max",
            rate=EXPANSION_RATE,
        )
        hfun.add_patch(
            shape=box(*BOX2),
            expansion_rate=EXPANSION_RATE,
            target_size=1000.0,
        )
        mid_lat2 = (BOX2[1] + BOX2[3]) / 2.0
        hfun.add_feature(
            shape=LineString([(BOX2[0], mid_lat2), (BOX2[2], mid_lat2)]),
            expansion_rate=EXPANSION_RATE,
            target_size=1000.0,
            crs=4326,
        )
        hfun.add_patch(
            shape=box(*BOX3),
            expansion_rate=EXPANSION_RATE,
            target_size=800.0,
        )
        mid_lat3 = (BOX3[1] + BOX3[3]) / 2.0
        hfun.add_feature(
            shape=LineString([(BOX3[0], mid_lat3), (BOX3[2], mid_lat3)]),
            expansion_rate=EXPANSION_RATE,
            target_size=800.0,
            crs=4326,
        )

    _logger.info("Hfun refinements applied.")
    return hfun


# ---------------------------------------------------------------------------
# Standalone inspection
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    raster_paths, raster_metas = load_ordered_rasters(manifest)
    n_gebco = sum(1 for m in raster_metas if m.get("source") == "gebco")
    n_cudem = sum(1 for m in raster_metas if m.get("source") == "cudem")
    _logger.info(
        f"Loaded {len(raster_paths)} rasters "
        f"({n_gebco} GEBCO, {n_cudem} CUDEM)"
    )
    classes = _cudem_indices_by_class(raster_metas)
    names = {
        0: "subtidal_flow_limiter",
        1: "constant_value",
        2: "topo_bound_constraint",
        3: "topo_func_constraint",
        4: "courant_constraint",
        5: "(skipped)",
    }
    _logger.info("Index-modulo classes:")
    for cls, idxs in classes.items():
        _logger.info(
            f"  class {cls} {names[cls]:<24} -> {len(idxs)} tiles "
            f"{idxs[:8]}{'...' if len(idxs) > 8 else ''}"
        )


if __name__ == "__main__":
    _main()
