"""NJ_test/dem_config.py

Central configuration for the New Jersey coastal mesh benchmark.

Domain
------
    Latitude  : 39.0 N to 40.5 N
    Longitude : -75.0 to -73.0

DEM priority stack (lowest to highest)
---------------------------------------
    1. GEBCO split tiles  (~450m, 15 arc-sec)   — lowest priority
    2. NCEI third arc-sec (~10m,  1/3 arc-sec)  — middle priority
    3. CUDEM ninth arc-sec (~3m,  1/9 arc-sec)  — highest priority

Resolution note
---------------
    ncei13_  = 1/3 arc-second ≈ 10 m  (THIRD  — lower res)
    ncei19_  = 1/9 arc-second ≈  3 m  (NINTH  — higher res, highest priority)

    The finer the arc-second fraction, the higher the resolution.
    1/9 > 1/3, so CUDEM (ninth) overrides NCEI third where they overlap.

NCEI ninth arc-second subfolders covering NJ (dataset 8483):
    northeast_sandy  — 19 NJ tiles
    chesapeake_bay   —  1 NJ tile

NCEI third arc-second subfolders covering NJ (dataset 8580):
    northeast_sandy  — 20 NJ tiles
    (chesapeake_bay third tiles are all south of 39N — excluded)
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Domain bounds
# ---------------------------------------------------------------------------
DOMAIN_LAT_MIN =  39.0
DOMAIN_LAT_MAX =  40.5
DOMAIN_LON_MIN = -75.0
DOMAIN_LON_MAX = -73.0

DOMAIN_BBOX = (DOMAIN_LON_MIN, DOMAIN_LAT_MIN, DOMAIN_LON_MAX, DOMAIN_LAT_MAX)

TILE_INTERSECT_BUFFER_DEG = 0.05

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # NJ_test/
_REPO_DIR = _THIS_DIR.parent                         # ocsmesh_mpi_test/
_PROJ_DIR = _REPO_DIR.parent                         # OCSMesh_MPI/

# Existing GEBCO split tiles (already on disk from prior STOFS work)
GEBCO_SPLIT_DIR = Path(
    os.environ.get(
        "GEBCO_SPLIT_DIR",
        str(_PROJ_DIR / "stofs_dems" / "gebco_split"),
    )
)

DEFAULT_DEM_DIR     = str(_THIS_DIR / "NJ_dems")
DEFAULT_MANIFEST    = str(_THIS_DIR / "NJ_dem_manifest.json")
DEFAULT_RESULTS_DIR = str(_THIS_DIR / "results")

# ---------------------------------------------------------------------------
# GEBCO
# ---------------------------------------------------------------------------
GEBCO_TILE_PATTERN = "gebco_tile_*.tif"

# ---------------------------------------------------------------------------
# NCEI third arc-second  (dataset 8580, ~10m, MIDDLE priority)
# ---------------------------------------------------------------------------
THIRD_S3_BASE = (
    "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com"
    "/dem/NCEI_third_Topobathy_2014_8580"
)
THIRD_INDEX_BASE = (
    "https://coast.noaa.gov/htdata/raster2/elevation"
    "/NCEI_third_Topobathy_2014_8580"
)
# Only northeast_sandy has tiles in the NJ domain.
# chesapeake_bay third tiles are all south of 39N — skip.
THIRD_SUBFOLDERS = [
    "northeast_sandy",
]
# ncei13 tiles are 0.25 degrees per side (confirmed by domain filter test)
THIRD_TILE_SIZE_DEG = 0.25

# ---------------------------------------------------------------------------
# CUDEM ninth arc-second  (dataset 8483, ~3m, HIGHEST priority)
# ---------------------------------------------------------------------------
NINTH_S3_BASE = (
    "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com"
    "/dem/NCEI_ninth_Topobathy_2014_8483"
)
NINTH_INDEX_BASE = (
    "https://coast.noaa.gov/htdata/raster2/elevation"
    "/NCEI_ninth_Topobathy_2014_8483"
)
NINTH_SUBFOLDERS = [
    "northeast_sandy",
    "chesapeake_bay",
]
NINTH_TILE_SIZE_DEG = 0.25

# ---------------------------------------------------------------------------
# Priority constants (used in manifest JSON and HfunCollector ordering)
# ---------------------------------------------------------------------------
PRIORITY_GEBCO = 0    # lowest  — ocean background (~450m)
PRIORITY_THIRD = 1    # middle  — 1/3 arc-sec (~10m)
PRIORITY_NINTH = 2    # highest — 1/9 arc-sec (~3m)

SOURCE_GEBCO = "gebco"
SOURCE_THIRD = "ncei_third"
SOURCE_NINTH = "ncei_ninth"
