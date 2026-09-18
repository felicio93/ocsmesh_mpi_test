"""Build the STOFS-3D-Atlantic Geom and Hfun for the OCSMesh MPI benchmark.

This is the core "recipe" module. It is imported by run_benchmark.py, but
can also be run standalone to inspect how the refinements resolve to the
rasters before launching a full meshdata() computation.

Refinement strategy
====================

Global mesh size bounds
-----------------------
    hmin = 1000 m   (1 km — finest)
    hmax = 7000 m   (7 km — coarsest / background)

Per-source refinements — index modulo (CUDEM tiles only)
--------------------------------------------------------
Instead of testing whether each raster falls inside a lat/lon box
(expensive geometry ops), we assign refinements by the raster's *position*
in the input list. GEBCO is index 0 (skipped here); CUDEM tiles occupy
indices 1..N. For CUDEM tile at list index ``i`` we use ``(i - 1) % 6``:

    (i-1) % 6 == 0  ->  add_subtidal_flow_limiter
    (i-1) % 6 == 1  ->  add_constant_value
    (i-1) % 6 == 2  ->  add_topo_bound_constraint
    (i-1) % 6 == 3  ->  add_topo_func_constraint
    (i-1) % 6 == 4  ->  add_courant_num_constraint
    (i-1) % 6 == 5  ->  (skipped — leaves some tiles refinement-free)

``source_index`` accepts a list, so each refinement is added with ONE call
passing all the tiles in its class.

Global refinements — all rasters
--------------------------------
    add_contour   (shoreline 0 m + shelf break -200 m)
    add_channel   (narrow-region detection)

Shape-based refinements — fixed lat/lon boxes
---------------------------------------------
    box1 (-85..-82, 25..31)  ->  add_region_constraint (max 3500 m)
    box2 (-80..-77, 31..35)  ->  add_patch (target 1000 m)
    feature line across box2 mid-latitude -> add_feature

    box3 (-90..-86, 28..31)  ->  add_patch (target 800 m)   [NEW — Gulf Coast]
    feature line across box3 mid-latitude -> add_feature     [NEW — Gulf Coast]

Priority note
-------------
Rasters are passed to HfunCollector in ascending-priority order
(GEBCO first = lowest, CUDEM tiles last = highest). HfunCollector reverses
the list internally, so the last CUDEM tile wins on overlap.

Execution mode translation
--------------------------
OCSMesh execution_mode accepts: 'serial', 'parallel', 'mpi'.
This module additionally accepts 'mpi_no_pool' and 'mpi_hybrid' as aliases:

    mpi_no_pool  ->  execution_mode='mpi', nprocs forced to 1 per rank
                     (pure MPI, no internal Pool per rank)
    mpi_hybrid   ->  execution_mode='mpi', nprocs = auto (cores/rank)
                     (MPI ranks each spawn an internal Pool)

The nprocs translation is handled by the caller (run_benchmark.py).
build_hfun() receives the already-translated ocsmesh_mode string.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import box, LineString

from ocsmesh import Geom, Hfun

_logger = logging.getLogger("stofs_benchmark.build")

# ---------------------------------------------------------------------------
# Global mesh size bounds (metres)
# ---------------------------------------------------------------------------
GLOBAL_HMIN = 1000.0      # 1 km — finest
GLOBAL_HMAX = 7000.0      # 7 km — coarsest / background
EXPANSION_RATE = 0.15     # transition zone width (~8 km); was 0.05 (gave 50 km)

# ---------------------------------------------------------------------------
# Index-modulo scheme for per-source refinements (CUDEM tiles only)
# ---------------------------------------------------------------------------
MODULO_STRIDE = 6

# ---------------------------------------------------------------------------
# Fixed lat/lon boxes for shape-based refinements
# (lon_min, lat_min, lon_max, lat_max) in EPSG:4326
# All confirmed inside STOFS domain (-98..−53, 7.8..52.1)
# ---------------------------------------------------------------------------
BOX1 = (-85.0, 25.0, -82.0, 31.0)   # West Florida shelf  -> region_constraint
BOX2 = (-80.0, 31.0, -77.0, 35.0)   # SC/GA coast         -> patch + feature
BOX3 = (-90.0, 28.0, -86.0, 31.0)   # Gulf Coast (new)    -> patch + feature


# ---------------------------------------------------------------------------
# Module-level helper for topo_func_constraint.
# MUST be module-level (picklable). Anas's branch uses keyword 'function='.
# A lambda here raises ValueError due to the lambda guard in TopoFuncConstraint.
# ---------------------------------------------------------------------------
def _half_depth(depth: np.ndarray) -> np.ndarray:
    """Mesh size = |depth| / 2, used by topo_func_constraint."""
    return np.abs(depth) / 2.0


# ---------------------------------------------------------------------------
# Manifest / raster helpers
# ---------------------------------------------------------------------------

def load_ordered_rasters(manifest: Dict) -> Tuple[List[str], List[Dict]]:
    """Return raster paths + metadata sorted by priority (GEBCO first).

    Only DEMs marked ``available`` and present on disk are included.

    Returns
    -------
    (paths, metas) : (list of str, list of dict)
        Parallel lists; index i in each corresponds to the raster at
        position i in the list passed to Hfun() — i.e. a valid
        ``source_index`` value.
    """
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
    """Group CUDEM raster-list indices by their modulo class.

    GEBCO (source == 'gebco') is excluded. For each CUDEM tile at
    raster-list index ``i``, its class is
    ``(cudem_position) % MODULO_STRIDE`` where cudem_position is the
    0-based counter over CUDEM tiles only.

    Returns
    -------
    dict : {class_id: [raster_list_index, ...]}
    """
    classes: Dict[int, List[int]] = {c: [] for c in range(MODULO_STRIDE)}
    cudem_pos = 0
    for i, meta in enumerate(metas):
        if meta.get("source") == "gebco":
            continue
        cls = cudem_pos % MODULO_STRIDE
        classes[cls].append(i)
        cudem_pos += 1
    return classes


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
    light_features: bool = False,
    skip_topofunc: bool = False,
    skip_constraints: bool = False,
    skip_box_refinements: bool = False,
    all_fast_refinements: bool = False,
    config_f: bool = False,
    config_g: bool = False,
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
        Worker count. For mpi_no_pool this is 1; for mpi_hybrid this is
        the per-rank core budget computed by run_benchmark.py.
    execution_mode : str
        One of 'serial', 'parallel', 'mpi'. The mpi_no_pool / mpi_hybrid
        translation is done by run_benchmark.py before calling here.
    light_features : bool
        Skip global add_contour / add_channel.
    skip_topofunc : bool
        Skip add_topo_func_constraint only.
    skip_constraints : bool
        Skip ALL topo/courant constraints.
    skip_box_refinements : bool
        Skip add_region_constraint / add_patch / add_feature.
    all_fast_refinements : bool
        Apply flow_limiter + const_value to every CUDEM tile; force all
        slow stages off.
    config_f : bool
        Config F recipe: flow_limiter + const_value + constraints +
        contour/channel on all tiles. No box refinements. Maps to
        Anas's Config E (all MPI-dispatched, no shape bottleneck).
    config_g : bool
        Config G recipe: Config F + patch + feature (BOX2 + BOX3).
        Maps to Anas's Config F (full pipeline, all ops MPI-dispatched).

    Returns
    -------
    Hfun (HfunCollector)
    """
    _logger.info(
        f"Building Hfun from {len(raster_paths)} DEMs "
        f"(mode={execution_mode}, nprocs={nprocs}, "
        f"hmin={GLOBAL_HMIN}, hmax={GLOBAL_HMAX})"
    )

    hfun = Hfun(
        raster_paths,
        hmin=GLOBAL_HMIN,
        hmax=GLOBAL_HMAX,
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

    # ── Config F: all MPI-dispatched ops, no shape bottleneck ─────────────
    # Mirrors Anas's Config E. Every operation below is MPI-dispatched.
    # add_patch / add_feature intentionally excluded to avoid coordinator-
    # only serialization bottleneck — lets mpi_hybrid show true advantage.
    if config_f:
        _logger.info("Config F recipe: flow_limiter + const_value + "
                     "constraints + contour/channel (all tiles, no boxes)")
        all_cudem_idx = [
            i for i, meta in enumerate(raster_metas)
            if meta.get("source") != "gebco"
        ]

        hfun.add_subtidal_flow_limiter(
            hmin=GLOBAL_HMIN, hmax=GLOBAL_HMAX,
            lower_bound=-200.0, upper_bound=0.0,
            source_index=all_cudem_idx,
        )
        hfun.add_constant_value(
            value=1000.0, lower_bound=-5.0, upper_bound=0.0,
            source_index=all_cudem_idx,
        )
        hfun.add_topo_bound_constraint(
            value=1500.0, upper_bound=1.0, lower_bound=-2.0,
            value_type="min", rate=EXPANSION_RATE,
            source_index=all_cudem_idx,
        )
        hfun.add_topo_func_constraint(
            func=_half_depth,
            upper_bound=0.0, lower_bound=-3000.0,
            value_type="min", rate=EXPANSION_RATE,
            source_index=all_cudem_idx,
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

    # ── Config G: full pipeline, all ops MPI-dispatched ───────────────────
    # Mirrors Anas's Config F. Config F ops + add_patch + add_feature
    # on BOX2 and BOX3. Tests that shape-based ops work alongside all
    # other MPI-dispatched ops without introducing bottlenecks.
    if config_g:
        _logger.info("Config G recipe: Config F + patch + feature "
                     "(BOX2 SC/GA + BOX3 Gulf Coast)")
        all_cudem_idx = [
            i for i, meta in enumerate(raster_metas)
            if meta.get("source") != "gebco"
        ]

        hfun.add_subtidal_flow_limiter(
            hmin=GLOBAL_HMIN, hmax=GLOBAL_HMAX,
            lower_bound=-200.0, upper_bound=0.0,
            source_index=all_cudem_idx,
        )
        hfun.add_constant_value(
            value=1000.0, lower_bound=-5.0, upper_bound=0.0,
            source_index=all_cudem_idx,
        )
        hfun.add_topo_bound_constraint(
            value=1500.0, upper_bound=1.0, lower_bound=-2.0,
            value_type="min", rate=EXPANSION_RATE,
            source_index=all_cudem_idx,
        )
        hfun.add_topo_func_constraint(
            func=_half_depth,
            upper_bound=0.0, lower_bound=-3000.0,
            value_type="min", rate=EXPANSION_RATE,
            source_index=all_cudem_idx,
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
        # BOX2: SC/GA coast patch + line feature
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
        # BOX3: Gulf Coast patch + line feature
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

    # ── Standard recipe (Configs A–E): index-modulo + flags ──────────────
    classes = _cudem_indices_by_class(raster_metas)
    flow_idx    = classes[0]
    const_idx   = classes[1]
    bound_idx   = classes[2]
    func_idx    = classes[3]
    courant_idx = classes[4]
    skip_idx    = classes[5]

    # all_fast_refinements: flow+const on every tile, everything else off
    if all_fast_refinements:
        all_cudem_idx = [
            i for i, meta in enumerate(raster_metas)
            if meta.get("source") != "gebco"
        ]
        flow_idx  = list(all_cudem_idx)
        const_idx = list(all_cudem_idx)
        bound_idx = func_idx = courant_idx = []
        skip_idx  = []
        skip_constraints     = True
        skip_box_refinements = True
        light_features       = True
        _logger.info(
            "  all_fast_refinements=True -> flow_limiter + const_value on "
            f"ALL {len(all_cudem_idx)} CUDEM tiles; constraints, "
            "contour/channel, and box refinements SKIPPED."
        )

    _logger.info("  Index-modulo assignment (CUDEM tiles):")
    _logger.info(f"    flow_limiter    : {len(flow_idx)} tiles")
    _logger.info(f"    constant_value  : {len(const_idx)} tiles")
    _logger.info(f"    topo_bound      : {len(bound_idx)} tiles")
    _logger.info(f"    topo_func       : {len(func_idx)} tiles")
    _logger.info(f"    courant         : {len(courant_idx)} tiles")
    _logger.info(f"    skipped         : {len(skip_idx)} tiles")

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
            _logger.info("  topo_bound_constraint SKIPPED (skip_constraints=True)")
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
        _logger.info(
            "  topo_func_constraint SKIPPED "
            "(skip_topofunc/skip_constraints=True)"
        )

    if courant_idx:
        if skip_constraints:
            _logger.info(
                "  courant_num_constraint SKIPPED (skip_constraints=True)"
            )
        else:
            hfun.add_courant_num_constraint(
                upper_bound=0.9,
                timestep=150.0,
                wave_amplitude=2.0,
                source_index=courant_idx,
            )

    # ── Global refinements: contour + channel ─────────────────────────────
    if light_features:
        _logger.info(
            "  Global: add_contour / add_channel SKIPPED (light_features=True)"
        )
    else:
        _logger.info("  Global: add_contour (0 m + -200 m)")
        hfun.add_contour(
            level=[0.0, -200.0],
            expansion_rate=EXPANSION_RATE,
            target_size=3500.0,
        )
        _logger.info("  Global: add_channel")
        hfun.add_channel(
            level=0.0, width=1000.0,
            target_size=3500.0,
            expansion_rate=EXPANSION_RATE,
        )

    # ── Shape-based refinements: fixed boxes ──────────────────────────────
    if skip_box_refinements:
        _logger.info(
            "  Box refinements SKIPPED (skip_box_refinements=True): "
            "region_constraint, patch, feature"
        )
    else:
        _logger.info(f"  Box1 {BOX1}: add_region_constraint (max 3500 m)")
        hfun.add_region_constraint(
            value=3500.0,
            shape=box(*BOX1),
            crs="EPSG:4326",
            value_type="max",
            rate=EXPANSION_RATE,
        )

        _logger.info(f"  Box2 {BOX2}: add_patch (target 1000 m)")
        hfun.add_patch(
            shape=box(*BOX2),
            expansion_rate=EXPANSION_RATE,
            target_size=1000.0,
        )
        mid_lat2 = (BOX2[1] + BOX2[3]) / 2.0
        _logger.info(f"  Box2: add_feature (line at lat={mid_lat2})")
        hfun.add_feature(
            shape=LineString([(BOX2[0], mid_lat2), (BOX2[2], mid_lat2)]),
            expansion_rate=EXPANSION_RATE,
            target_size=1000.0,
            crs=4326,
        )

        _logger.info(f"  Box3 {BOX3}: add_patch (target 800 m)")
        hfun.add_patch(
            shape=box(*BOX3),
            expansion_rate=EXPANSION_RATE,
            target_size=800.0,
        )
        mid_lat3 = (BOX3[1] + BOX3[3]) / 2.0
        _logger.info(f"  Box3: add_feature (line at lat={mid_lat3})")
        hfun.add_feature(
            shape=LineString([(BOX3[0], mid_lat3), (BOX3[2], mid_lat3)]),
            expansion_rate=EXPANSION_RATE,
            target_size=800.0,
            crs=4326,
        )

    _logger.info("Hfun refinements applied.")
    return hfun


# ---------------------------------------------------------------------------
# Standalone inspection entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    """Standalone: load manifest and report the index-modulo assignment."""
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser(
        description="Inspect Hfun refinement assignment for the STOFS benchmark."
    )
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    raster_paths, raster_metas = load_ordered_rasters(manifest)
    _logger.info(
        f"Loaded {len(raster_paths)} available rasters "
        f"({sum(1 for m in raster_metas if m.get('source')=='gebco')} GEBCO, "
        f"{sum(1 for m in raster_metas if m.get('source')=='cudem')} CUDEM)."
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
