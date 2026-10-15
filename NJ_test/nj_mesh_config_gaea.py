"""NJ_test/nj_mesh_config_gaea.py

Gaea C6 mesh parameters for the NJ benchmark tests. Same recipe as
nj_mesh_config.py (Hercules) except CHANNELS, the test 4/5 definitions,
the results root and the SLURM settings.
"""

from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR       = Path(__file__).resolve().parent
DOMAIN_SHP      = _THIS_DIR / "domain" / "AOI_domain_FC.shp"
MANIFEST_PATH   = _THIS_DIR / "NJ_dem_manifest.json"
RESULTS_ROOT    = _THIS_DIR / "results_gaea"

# ---------------------------------------------------------------------------
# Global mesh size bounds
# ---------------------------------------------------------------------------
HMIN =    50.0   # m
HMAX =  3000.0   # m

# ---------------------------------------------------------------------------
# Tests
# Tests 0-3 are cumulative; test 4 isolates the channel refinement;
# test 5 applies everything (tests 1-4 combined).
# Each test ID is run with BOTH method='fast' and method='exact'.
# NOTE: OCSMesh only distributes contours/channels/flow limiters for
# method='exact'; with 'fast' parallel/mpi give no speedup there.
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
        "description": "Depth bands + shoreline contour (level 0m, target 200m, rate=0.001)",
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
        "description": "Channel refinement only (1 channel, 2000m width)",
        "depth_bands":     False,
        "contour":         False,
        "flow_limiter":    False,
        "channels":        True,
    },
    {
        "id":          5,
        "name":        "full",
        "description": "All refinements (tests 1-4 combined) — canonical final NJ mesh",
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
# Contour
# ---------------------------------------------------------------------------
CONTOUR_LEVEL          =    0.0
CONTOUR_EXPANSION_RATE =  0.001
CONTOUR_TARGET_SIZE    =  200.0   # m

# ---------------------------------------------------------------------------
# Flow limiters
# ---------------------------------------------------------------------------
FLOW_LIMITERS = [
    # (hmin_m, hmax_m, lower_bound, upper_bound, description)
    # hmax matches global HMAX=3000m to avoid overriding the background resolution
    (100.0,  3000.0, -10.0,   0.0,  "nearshore  0→-10m,  hmin=100m"),
    (500.0,  3000.0, -100.0, -10.0, "subtidal -10→-100m, hmin=500m"),
]

# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------
CHANNELS = [
    # (width_m, target_size_m, expansion_rate, description)
    # Single channel only: width=2000m, rate=0.01 (target_size 500m is
    # the element size inside the channel, not a second channel).
    (2000.0, 500.0, 0.01, "large coastal inlets   width=2000m target=500m"),
]

# ---------------------------------------------------------------------------
# MeshDriver
# ---------------------------------------------------------------------------
MESH_ENGINE   = "gmsh"
MESH_BND_REPR = "adapt"

# ---------------------------------------------------------------------------
# Execution modes and resource allocation
# ---------------------------------------------------------------------------
# 42 tiles total (2 GEBCO + 20 third + 20 ninth)
N_TILES           = 42

MPI_NOPOOL_NTASKS = N_TILES + 1   # 43 ranks (1 manager + 42 workers)
MPI_NOPOOL_CPUS   = 1

# Multi-node MPI: same 1-core-per-rank layout as mpi_no_pool, but ranks
# are spread over MPI_MULTINODE_NODES nodes (ntasks-per-node enforced,
# otherwise Slurm's block distribution packs all ranks on node 1).
MPI_MULTINODE_NODES = 2
MPI_MULTINODE_TASKS_PER_NODE = -(-MPI_NOPOOL_NTASKS // MPI_MULTINODE_NODES)

MPI_HYBRID_NTASKS = 6             # 1 manager + 5 workers
MPI_HYBRID_CPUS   = 7             # 7 cores per rank
MPI_HYBRID_CORES  =  7            # alias used in build_nj_mesh.py

PARALLEL_NPROCS   = 40
SERIAL_NPROCS     = 1

# Gaea C6
SLURM_ACCOUNT   = "nos-surge"
SLURM_PARTITION = "batch"
SLURM_EMAIL     = "felicio.cassalho@noaa.gov"
