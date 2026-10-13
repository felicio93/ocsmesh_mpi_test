"""Unalakleet_test/unalakleet_config.py

Central configuration for the Unalakleet dual-resolution mesh benchmark.

Domain
------
    Outer domain : domain.shp          (coarse, GEBCO background)
    Inner domain : highres_domain.shp  (fine, USGS 3m DEM priority)
    Channel mask : USGS_AK5M_Alaska_Mid_Ac_Clip_mask.shp (patch)

DEM stack (low → high priority)
---------------------------------
    1. GEBCO 2026 (~450m)                          — lowest priority
    2. USGS_AK5M_Alaska_Mid_Ac_Clip_river_bathymetry (~3m) — highest priority

Tests (cumulative refinements)
-------------------------------
    t0  baseline        no refinements
    t1  outer_depth     outer depth bands (whole domain)
    t2  outer_contour   + outer contour (0m, 500m, rate=0.001)
    t3  inner_depth     + inner depth bands (USGS DEM only, within highres)
    t4  inner_contour   + inner contour (0m, 40m, rate=0.0001, within highres)
    t5  flow_limiter    + inner tidal flow limiter (hmin=200m, within highres)
    t6  channels        + channel mask patch (20m inside polygon)
    t7  full            all combined

Key design decisions:
    - Geom uses domain.shp as authoritative boundary (zmax=99999 — no elev clipping)
    - Inner refinements restricted to highres_domain.shp via source_index
    - Outer refinements apply to full domain via all source indices
    - fast method disabled — exact is fast enough with 2 DEMs (<60s for t0-t3)
"""

from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR    = Path(__file__).resolve().parent

# Shapefiles
DOMAIN_SHP       = _THIS_DIR / "shapes" / "domain.shp"
HIGHRES_SHP      = _THIS_DIR / "shapes" / "highres_domain.shp"
CHANNEL_MASK_SHP = _THIS_DIR / "shapes" / "USGS_AK5M_Alaska_Mid_Ac_Clip_mask.shp"

# DEMs — order matters: GEBCO first (lowest priority), USGS last (highest)
DEM_GEBCO = _THIS_DIR / "dems" / "gebco_2026_n65.5_s63.0_w-163.0_e-160.0_geotiff.tif"
DEM_USGS  = _THIS_DIR / "dems" / "USGS_AK5M_Alaska_Mid_Ac_Clip_river_bathymetry.tif"

DEM_PATHS = [
    str(DEM_GEBCO),   # index 0 — lowest priority
    str(DEM_USGS),    # index 1 — highest priority
]

# Results
RESULTS_ROOT = _THIS_DIR / "results"

# ---------------------------------------------------------------------------
# Global mesh size bounds
# ---------------------------------------------------------------------------
HMIN =   10.0    # m — finest element (channel patch)
HMAX = 3000.0    # m — coarsest element

# ---------------------------------------------------------------------------
# Geom settings
# ---------------------------------------------------------------------------
# zmax set very high so domain.shp is the authoritative boundary —
# no elevation-based clipping of land areas.
GEOM_ZMIN = -10000.0
GEOM_ZMAX  =  99999.0

# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------
TESTS = [
    {
        "id":           0,
        "name":         "baseline",
        "description":  "No refinements — mesh defaults to hmax (3000m) everywhere",
        "outer_depth":  False,
        "outer_contour":False,
        "inner_depth":  False,
        "inner_contour":False,
        "flow_limiter": False,
        "channels":     False,
    },
    {
        "id":           1,
        "name":         "outer_depth",
        "description":  "Outer depth bands across full domain",
        "outer_depth":  True,
        "outer_contour":False,
        "inner_depth":  False,
        "inner_contour":False,
        "flow_limiter": False,
        "channels":     False,
    },
    {
        "id":           2,
        "name":         "outer_contour",
        "description":  "Outer depth bands + outer shoreline contour (500m, rate=0.001)",
        "outer_depth":  True,
        "outer_contour":True,
        "inner_depth":  False,
        "inner_contour":False,
        "flow_limiter": False,
        "channels":     False,
    },
    {
        "id":           3,
        "name":         "inner_depth",
        "description":  "All above + inner depth bands (USGS DEM only, within highres_domain)",
        "outer_depth":  True,
        "outer_contour":True,
        "inner_depth":  True,
        "inner_contour":False,
        "flow_limiter": False,
        "channels":     False,
    },
    {
        "id":           4,
        "name":         "inner_contour",
        "description":  "All above + inner contour (40m, rate=0.0001, within highres_domain)",
        "outer_depth":  True,
        "outer_contour":True,
        "inner_depth":  True,
        "inner_contour":True,
        "flow_limiter": False,
        "channels":     False,
    },
    {
        "id":           5,
        "name":         "flow_limiter",
        "description":  "All above + inner tidal flow limiter (hmin=200m, within highres_domain)",
        "outer_depth":  True,
        "outer_contour":True,
        "inner_depth":  True,
        "inner_contour":True,
        "flow_limiter": True,
        "channels":     False,
    },
    {
        "id":           6,
        "name":         "channels",
        "description":  "All above + channel mask patch (20m inside polygon)",
        "outer_depth":  True,
        "outer_contour":True,
        "inner_depth":  True,
        "inner_contour":True,
        "flow_limiter": True,
        "channels":     True,
    },
    {
        "id":           7,
        "name":         "full",
        "description":  "All refinements combined — canonical Unalakleet mesh",
        "outer_depth":  True,
        "outer_contour":True,
        "inner_depth":  True,
        "inner_contour":True,
        "flow_limiter": True,
        "channels":     True,
    },
]

# fast method disabled — exact is fast enough with 2 DEMs (<60s for t0-t3)
# fast method fails on mixed-resolution DEM stack (GEBCO + USGS 3m)
METHODS = ["exact"]

# ---------------------------------------------------------------------------
# Outer domain depth bands (add_constant_value, entire domain, all DEMs)
# Updated: 1500m for deep, 1000m for shallow water, 500m for land
# ---------------------------------------------------------------------------
OUTER_DEPTH_BANDS = [
    # (value_m, lower_bound, upper_bound, description)
    (1500.0, None,   -15.0, "depth > 15m          → 1500m"),
    (1000.0, -15.0,   0.0,  "depth 0–15m          → 1000m"),
    ( 500.0,   0.0,  None,  "above 0m (land/inter)→  500m"),
]

# ---------------------------------------------------------------------------
# Outer contour (add_contour, entire domain)
# Updated: 500m target, smoother rate=0.001
# ---------------------------------------------------------------------------
OUTER_CONTOUR_LEVEL          =    0.0
OUTER_CONTOUR_EXPANSION_RATE =  0.001   # smooth transition
OUTER_CONTOUR_TARGET_SIZE    =  500.0   # m

# ---------------------------------------------------------------------------
# Inner domain depth bands (add_constant_value, USGS DEM only — source_index=[1])
# Applied within highres_domain area via USGS DEM index
# Outer area follows outer depth bands (t1)
# Updated: doubled resolutions (was 100m→200m, was 200m→400m)
# ---------------------------------------------------------------------------
INNER_DEPTH_BANDS = [
    # (value_m, lower_bound, upper_bound, description)
    (200.0,  0.0,  5.0, "0–5m elevation (intertidal) → 200m"),
    (400.0,  5.0, None, "above 5m (land)             → 400m"),
]

# ---------------------------------------------------------------------------
# Inner contour (add_contour, restricted to highres_domain via source_index)
# Updated: doubled resolution (was 20m→40m), smoother rate (0.001→0.0001)
# ---------------------------------------------------------------------------
INNER_CONTOUR_LEVEL          =    0.0
INNER_CONTOUR_EXPANSION_RATE =  0.0001   # very smooth — stays confined to highres area
INNER_CONTOUR_TARGET_SIZE    =   40.0    # m (was 20m)

# ---------------------------------------------------------------------------
# Inner tidal flow limiter (add_subtidal_flow_limiter, USGS DEM only)
# Updated: doubled hmin (was 100m→200m)
# Applied only to USGS DEM (source_index=[1]) — confined to highres area
# ---------------------------------------------------------------------------
INNER_FLOW_LIMITER = {
    "hmin":         200.0,   # m (was 100m)
    "hmax":        3000.0,   # matches global HMAX
    "lower_bound":  -20.0,   # m depth
    "upper_bound":   -5.0,   # m depth
    "description":  "inner tidal flow limiter hmin=200m, -5 to -20m (USGS only)",
}

# ---------------------------------------------------------------------------
# Channel patch (add_patch, inside USGS_AK5M_Alaska_Mid_Ac_Clip_mask.shp)
# Updated: doubled target (was 10m→20m)
# ---------------------------------------------------------------------------
CHANNEL_PATCH = {
    "target_size":      20.0,   # m (was 10m)
    "expansion_rate":  0.001,
    "description":     "channel mask patch 20m inside USGS polygon",
}

# ---------------------------------------------------------------------------
# MeshDriver
# ---------------------------------------------------------------------------
MESH_ENGINE   = "gmsh"
MESH_BND_REPR = "adapt"

# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------
N_TILES           = 2    # GEBCO + USGS
MPI_NOPOOL_NTASKS = 3    # 1 manager + 2 workers
MPI_NOPOOL_CPUS   = 1

PARALLEL_NPROCS   = 2
SERIAL_NPROCS     = 1

# Hercules SLURM settings
SLURM_ACCOUNT   = "nos-surge"
SLURM_PARTITION = "hercules"
SLURM_EMAIL     = "felicio.cassalho@noaa.gov"
SLURM_WALLTIME  = "08:00:00"
