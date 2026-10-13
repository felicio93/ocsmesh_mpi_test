"""NJ_test/nj_mesh_config.py

All mesh generation parameters for the NJ benchmark tests.
Central place to change any parameter — all other scripts import from here.
"""

from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR       = Path(__file__).resolve().parent
DOMAIN_SHP      = _THIS_DIR / "domain" / "AOI_domain_FC.shp"
MANIFEST_PATH   = _THIS_DIR / "NJ_dem_manifest.json"
RESULTS_ROOT    = _THIS_DIR / "results"

# ---------------------------------------------------------------------------
# Global mesh size bounds
# ---------------------------------------------------------------------------
HMIN =    50.0   # m — finest element
HMAX =  3000.0   # m — coarsest element (open ocean background)

# ---------------------------------------------------------------------------
# Test definitions
# Each test is CUMULATIVE — it includes all refinements from previous tests.
#
# Test 0  baseline      no refinements → mesh defaults to hmax everywhere
# Test 1  depth_bands   + add_constant_value (5 depth bands)
# Test 2  contour       + add_contour (shoreline 0m, 200m target)
# Test 3  flow          + add_subtidal_flow_limiter (2 calls)
# Test 4  channels      + add_channel (tidal inlets only)
# Test 5  full          all above combined
# ---------------------------------------------------------------------------
TESTS = [
    {
        "id":          0,
        "name":        "baseline",
        "description": "No refinements — mesh defaults to hmax everywhere",
        "depth_bands":     False,
        "contour":         False,
        "flow_limiter":    False,
        "channels":        False,
    },
    {
        "id":          1,
        "name":        "depth_bands",
        "description": "5 depth-band constant values (3000→1000m by depth)",
        "depth_bands":     True,
        "contour":         False,
        "flow_limiter":    False,
        "channels":        False,
    },
    {
        "id":          2,
        "name":        "contour",
        "description": "Depth bands + shoreline contour (0m, 200m target, rate=0.001)",
        "depth_bands":     True,
        "contour":         True,
        "flow_limiter":    False,
        "channels":        False,
    },
    {
        "id":          3,
        "name":        "flow_limiter",
        "description": "Depth bands + contour + 2 flow limiters",
        "depth_bands":     True,
        "contour":         True,
        "flow_limiter":    True,
        "channels":        False,
    },
    {
        "id":          4,
        "name":        "channels",
        "description": "Depth bands + contour + flow limiters + channel refinement",
        "depth_bands":     True,
        "contour":         True,
        "flow_limiter":    True,
        "channels":        True,
    },
    {
        "id":          5,
        "name":        "full",
        "description": "All refinements combined — canonical final NJ mesh",
        "depth_bands":     True,
        "contour":         True,
        "flow_limiter":    True,
        "channels":        True,
    },
]

# Methods — both run for every test
METHODS = ["fast", "exact"]

# ---------------------------------------------------------------------------
# Depth bands (add_constant_value)
# Applied deepest first so shallower bands override deeper ones.
# ---------------------------------------------------------------------------
DEPTH_BANDS = [
    # (value_m, lower_bound, upper_bound, description)
    (3000.0, None,   -30.0, "ocean >30m deep  → 3000m"),
    (2500.0, -30.0,  -25.0, "25–30m deep      → 2500m"),
    (2000.0, -25.0,  -20.0, "20–25m deep      → 2000m"),
    (1500.0, -20.0,  -15.0, "15–20m deep      → 1500m"),
    (1000.0, -15.0,   None, "shallower -15m   → 1000m"),
]

# ---------------------------------------------------------------------------
# Contour (add_contour)
# ---------------------------------------------------------------------------
CONTOUR_LEVEL          =    0.0   # m — shoreline
CONTOUR_EXPANSION_RATE =  0.001   # smooth gradation away from shoreline
CONTOUR_TARGET_SIZE    =  200.0   # m — resolution at the shoreline

# ---------------------------------------------------------------------------
# Subtidal flow limiters (add_subtidal_flow_limiter)
# hmax matches global HMAX=3000m to avoid overriding background resolution
# ---------------------------------------------------------------------------
FLOW_LIMITERS = [
    # (hmin_m, hmax_m, lower_bound, upper_bound, description)
    (100.0,  3000.0, -10.0,   0.0,  "nearshore  0→-10m,  hmin=100m"),
    (500.0,  3000.0, -100.0, -10.0, "subtidal -10→-100m, hmin=500m"),
]

# ---------------------------------------------------------------------------
# Channel refinements (add_channel)
#
# Goal: resolve tidal inlets (Barnegat Inlet, Beach Haven Inlet, Absecon
# Inlet) and main back-barrier channels without flooding the entire domain
# with fine resolution.
#
# Key parameters:
#   width        — minimum channel width to detect. 1000m captures tidal
#                  inlets without flagging entire bays (Barnegat Bay ~2-3km
#                  wide — wider than this threshold so not flagged as channel)
#   target_size  — resolution inside detected channels. 100m is fine enough
#                  for tidal inlet hydrodynamics.
#   expansion_rate=0.1 — coarsens 10x faster than the previous 0.01 setting,
#                  so fine resolution stays confined to the inlet throat
#                  rather than spreading across the surrounding bay.
#
# Previous config (width=2000m, target=500m, rate=0.01) and
# (width=500m, target=100m, rate=0.01) produced 50m resolution everywhere
# because the slow expansion rate flooded the entire domain.
# ---------------------------------------------------------------------------
CHANNELS = [
    # (width_m, target_size_m, expansion_rate, description)
    (1000.0, 100.0, 0.1, "tidal inlets and back-barrier channels  width=1000m  target=100m"),
]

# ---------------------------------------------------------------------------
# MeshDriver settings
# ---------------------------------------------------------------------------
MESH_ENGINE   = "gmsh"
MESH_BND_REPR = "adapt"

# ---------------------------------------------------------------------------
# Execution modes and resource allocation
# ---------------------------------------------------------------------------
# Total tiles in manifest: 42 (2 GEBCO + 20 third + 20 ninth)
N_TILES           = 42

MPI_NOPOOL_NTASKS = N_TILES + 1   # 43 ranks (1 manager + 42 workers)
MPI_NOPOOL_CPUS   = 1

MPI_HYBRID_NTASKS = 6             # 1 manager + 5 workers
MPI_HYBRID_CPUS   = 7             # 7 cores per rank
MPI_HYBRID_CORES  = 7             # alias used in build_nj_mesh.py

PARALLEL_NPROCS   = 40
SERIAL_NPROCS     = 1

# Hercules SLURM settings
SLURM_ACCOUNT   = "nos-surge"
SLURM_PARTITION = "hercules"
SLURM_EMAIL     = "felicio.cassalho@noaa.gov"
