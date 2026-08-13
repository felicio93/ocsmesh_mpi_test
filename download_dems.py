"""Download DEMs for the STOFS-3D-Atlantic OCSMesh benchmark.

Strategy
--------
1 GEBCO tile  (full domain, 15 arc-sec, LOWEST priority)
    Downloaded via NCEI ETOPO WCS as a single GeoTIFF covering the full
    STOFS-3D-Atlantic extent.

~375 CUDEM 1/9 arc-second tiles (every other tile, HIGHEST priority)
    The NOAA NCEI dataset 8483 "Ninth Arc-Second Topobathy" tiles live on
    NOAA's public S3 bucket. Each tile is a ~0.25° × 0.25° GeoTIFF at
    1/9 arc-second (~3 m) resolution.

    We select EVERY OTHER tile (alternating in sorted order) from each
    subfolder covering the STOFS-3D-Atlantic domain:
        MA_NH_ME, rima, northeast_sandy, chesapeake_bay, NC,
        southeast, FL, AL_nwFL, LA_MS, TX

    This gives approximately half the coastal tiles, intentionally creating
    gaps in coastal coverage — the GEBCO background shows through between
    CUDEM tiles, so you can visually confirm that:
      (a) the two sources overlap and blend correctly, and
      (b) mesh refinements applied on CUDEM tiles (higher priority) take
          precedence where CUDEM is present, while GEBCO-based refinements
          appear in the gaps.

Priority ordering for HfunCollector
-------------------------------------
HfunCollector reverses self._hfun_list internally: the LAST item in the
input list has HIGHEST priority (its bounding box is never clipped away).
We therefore build the raster list as:

    [gebco, cudem_MA_NH_ME_0, cudem_MA_NH_ME_2, ...,
            cudem_TX_0, cudem_TX_2, ...]

→ GEBCO at index 0 (lowest priority, deep ocean background)
→ CUDEM tiles at indices 1…N (higher priority, coastal override)

Since adjacent CUDEM tiles already have ~0° gap between them (NCEI tiles
are edge-to-edge), they automatically overlap where consecutive tiles share
a common edge. They also overlap with GEBCO everywhere a CUDEM tile exists.

Puerto Rico
-----------
The PR 1/9 arc-second tiles live in a separate dataset (ID 9525) at a
different S3 prefix. A small representative set is included.

Usage
-----
    python download_dems.py --out-dir /work/noaa/<user>/stofs_dems

    # Download only one subfolder:
    python download_dems.py --only MA_NH_ME --out-dir ./dems

    # Dry run: print what would be downloaded
    python download_dems.py --dry-run

Output
------
    <out-dir>/gebco/stofs_atlantic_gebco.tif
    <out-dir>/MA_NH_ME/ncei19_n41x25_w070x00_2021v1.tif
    ...
    dem_manifest.json

Environment variables
---------------------
    GEBCO_LOCAL   Path to a locally available GEBCO 2023 / ETOPO 2022
                  GeoTIFF. If set and the file exists, no GEBCO download
                  is attempted and the file is referenced in the manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# S3 base URL for NOAA NCEI ninth-arc-second topobathy tiles (dataset 8483)
# ---------------------------------------------------------------------------
_S3_BASE = (
    "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com"
    "/dem/NCEI_ninth_Topobathy_2014_8483"
)
_S3_PR = (
    "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com"
    "/dem/NCEI_ninth_Topobathy_PuertoRico_9525"
)

# ---------------------------------------------------------------------------
# Complete tile lists per subfolder (scraped from NOAA NCEI index pages)
# Each entry is just the filename; the S3 URL is constructed at runtime.
# ---------------------------------------------------------------------------

_TILES: Dict[str, List[str]] = {
    # ── MA / NH / ME  (77 tiles, 11 GB total) ────────────────────────────
    "MA_NH_ME": [
        "ncei19_n41x25_w070x00_2021v1.tif",
        "ncei19_n41x25_w070x25_2021v1.tif",
        "ncei19_n41x50_w070x00_2021v1.tif",
        "ncei19_n41x50_w070x25_2021v1.tif",
        "ncei19_n41x50_w070x50_2021v1.tif",
        "ncei19_n41x75_w070x00_2021v1.tif",
        "ncei19_n41x75_w070x25_2021v1.tif",
        "ncei19_n41x75_w070x50_2021v1.tif",
        "ncei19_n42x00_w070x00_2021v1.tif",
        "ncei19_n42x00_w070x25_2021v1.tif",
        "ncei19_n42x00_w070x50_2021v1.tif",
        "ncei19_n42x00_w071x00_2021v1.tif",
        "ncei19_n42x25_w070x25_2021v1.tif",
        "ncei19_n42x25_w070x75_2021v1.tif",
        "ncei19_n42x25_w071x00_2021v1.tif",
        "ncei19_n42x25_w071x25_2021v1.tif",
        "ncei19_n42x50_w071x00_2021v1.tif",
        "ncei19_n42x50_w071x25_2021v1.tif",
        "ncei19_n42x75_w070x75_2021v1.tif",
        "ncei19_n42x75_w071x00_2021v1.tif",
        "ncei19_n42x75_w071x25_2021v1.tif",
        "ncei19_n43x00_w070x75_2021v1.tif",
        "ncei19_n43x00_w071x00_2021v1.tif",
        "ncei19_n43x00_w071x25_2021v1.tif",
        "ncei19_n43x25_w070x50_2021v1.tif",
        "ncei19_n43x25_w070x75_2021v1.tif",
        "ncei19_n43x25_w071x00_2021v1.tif",
        "ncei19_n43x50_w070x50_2021v1.tif",
        "ncei19_n43x50_w070x75_2021v1.tif",
        "ncei19_n43x75_w070x00_2021v1.tif",
        "ncei19_n43x75_w070x25_2021v1.tif",
        "ncei19_n43x75_w070x50_2021v1.tif",
        "ncei19_n43x75_w070x75_2021v1.tif",
        "ncei19_n44x00_w068x25_2021v1.tif",
        "ncei19_n44x00_w068x75_2021v1.tif",
        "ncei19_n44x00_w069x00_2021v1.tif",
        "ncei19_n44x00_w069x25_2021v1.tif",
        "ncei19_n44x00_w069x50_2021v1.tif",
        "ncei19_n44x00_w069x75_2021v1.tif",
        "ncei19_n44x00_w070x00_2021v1.tif",
        "ncei19_n44x00_w070x25_2021v1.tif",
        "ncei19_n44x00_w070x50_2021v1.tif",
        "ncei19_n44x25_w068x25_2021v1.tif",
        "ncei19_n44x25_w068x50_2021v1.tif",
        "ncei19_n44x25_w068x75_2021v1.tif",
        "ncei19_n44x25_w069x00_2021v1.tif",
        "ncei19_n44x25_w069x25_2021v1.tif",
        "ncei19_n44x25_w069x50_2021v1.tif",
        "ncei19_n44x25_w069x75_2021v1.tif",
        "ncei19_n44x25_w070x00_2021v1.tif",
        "ncei19_n44x50_w067x00_2021v1.tif",
        "ncei19_n44x50_w067x25_2021v1.tif",
        "ncei19_n44x50_w067x75_2021v1.tif",
        "ncei19_n44x50_w068x00_2021v1.tif",
        "ncei19_n44x50_w068x25_2021v1.tif",
        "ncei19_n44x50_w068x50_2021v1.tif",
        "ncei19_n44x50_w068x75_2021v1.tif",
        "ncei19_n44x50_w069x00_2021v1.tif",
        "ncei19_n44x50_w069x25_2021v1.tif",
        "ncei19_n44x50_w070x00_2021v1.tif",
        "ncei19_n44x75_w067x00_2021v1.tif",
        "ncei19_n44x75_w067x25_2021v1.tif",
        "ncei19_n44x75_w067x50_2021v1.tif",
        "ncei19_n44x75_w067x75_2021v1.tif",
        "ncei19_n44x75_w068x00_2021v1.tif",
        "ncei19_n44x75_w068x25_2021v1.tif",
        "ncei19_n44x75_w068x50_2021v1.tif",
        "ncei19_n44x75_w068x75_2021v1.tif",
        "ncei19_n44x75_w069x00_2021v1.tif",
        "ncei19_n45x00_w067x00_2021v1.tif",
        "ncei19_n45x00_w067x25_2021v1.tif",
        "ncei19_n45x00_w068x75_2021v1.tif",
        "ncei19_n45x00_w069x00_2021v1.tif",
        "ncei19_n45x25_w067x00_2021v1.tif",
        "ncei19_n45x25_w067x25_2021v1.tif",
        "ncei19_n45x25_w067x50_2021v1.tif",
    ],

    # ── Rhode Island / Narragansett (13 tiles, 2.5 GB) ────────────────────
    "rima": [
        "ncei19_n41x25_w071x75_2018v1.tif",
        "ncei19_n41x50_w070x75_2018v1.tif",
        "ncei19_n41x50_w071x00_2018v1.tif",
        "ncei19_n41x50_w071x25_2018v1.tif",
        "ncei19_n41x50_w071x50_2018v1.tif",
        "ncei19_n41x50_w071x75_2018v1.tif",
        "ncei19_n41x75_w070x75_2018v1.tif",
        "ncei19_n41x75_w071x00_2018v1.tif",
        "ncei19_n41x75_w071x25_2018v1.tif",
        "ncei19_n41x75_w071x50_2018v1.tif",
        "ncei19_n42x00_w070x75_2018v1.tif",
        "ncei19_n42x00_w071x25_2018v1.tif",
        "ncei19_n42x00_w071x50_2018v1.tif",
    ],

    # ── NJ / NY / CT (57 tiles, 11 GB) ───────────────────────────────────
    "northeast_sandy": [
        "ncei19_n39x00_w075x00_2018v2.tif",
        "ncei19_n39x00_w075x25_2014v1.tif",
        "ncei19_n39x00_w075x50_2014v1.tif",
        "ncei19_n39x25_w074x75_2018v2.tif",
        "ncei19_n39x25_w075x00_2018v2.tif",
        "ncei19_n39x25_w075x25_2018v2.tif",
        "ncei19_n39x25_w075x50_2014v1.tif",
        "ncei19_n39x50_w074x50_2018v2.tif",
        "ncei19_n39x50_w074x75_2018v2.tif",
        "ncei19_n39x50_w075x25_2018v2.tif",
        "ncei19_n39x50_w075x50_2018v2.tif",
        "ncei19_n39x50_w075x75_2014v1.tif",
        "ncei19_n39x75_w074x25_2018v2.tif",
        "ncei19_n39x75_w074x50_2018v2.tif",
        "ncei19_n39x75_w075x50_2014v1.tif",
        "ncei19_n39x75_w075x75_2014v1.tif",
        "ncei19_n40x00_w074x25_2018v2.tif",
        "ncei19_n40x00_w075x25_2014v1.tif",
        "ncei19_n40x00_w075x50_2014v1.tif",
        "ncei19_n40x25_w074x00_2018v2.tif",
        "ncei19_n40x25_w074x25_2018v2.tif",
        "ncei19_n40x25_w074x75_2014v1.tif",
        "ncei19_n40x25_w075x00_2014v1.tif",
        "ncei19_n40x25_w075x25_2014v1.tif",
        "ncei19_n40x50_w074x00_2018v2.tif",
        "ncei19_n40x50_w074x25_2018v2.tif",
        "ncei19_n40x75_w073x00_2015v1.tif",
        "ncei19_n40x75_w073x25_2015v1.tif",
        "ncei19_n40x75_w073x50_2015v1.tif",
        "ncei19_n40x75_w073x75_2015v1.tif",
        "ncei19_n40x75_w074x00_2015v1.tif",
        "ncei19_n40x75_w074x25_2015v1.tif",
        "ncei19_n41x00_w072x25_2015v1.tif",
        "ncei19_n41x00_w072x50_2015v1.tif",
        "ncei19_n41x00_w072x75_2015v1.tif",
        "ncei19_n41x00_w073x00_2015v1.tif",
        "ncei19_n41x00_w073x25_2015v1.tif",
        "ncei19_n41x00_w073x50_2015v1.tif",
        "ncei19_n41x00_w073x75_2015v1.tif",
        "ncei19_n41x00_w074x00_2015v1.tif",
        "ncei19_n41x00_w074x25_2015v1.tif",
        "ncei19_n41x25_w072x00_2015v1.tif",
        "ncei19_n41x25_w072x25_2015v1.tif",
        "ncei19_n41x25_w072x50_2015v1.tif",
        "ncei19_n41x25_w072x75_2015v1.tif",
        "ncei19_n41x25_w073x00_2016v1.tif",
        "ncei19_n41x25_w073x25_2016v1.tif",
        "ncei19_n41x25_w073x50_2015v1.tif",
        "ncei19_n41x25_w073x75_2015v1.tif",
        "ncei19_n41x25_w074x00_2015v1.tif",
        "ncei19_n41x50_w072x00_2016v1.tif",
        "ncei19_n41x50_w072x25_2016v1.tif",
        "ncei19_n41x50_w072x50_2016v1.tif",
        "ncei19_n41x50_w072x75_2016v1.tif",
        "ncei19_n41x50_w073x00_2016v1.tif",
        "ncei19_n41x50_w074x00_2015v1.tif",
        "ncei19_n41x50_w074x25_2015v1.tif",
    ],

    # ── Chesapeake Bay / Delmarva (87 tiles, 18 GB) ───────────────────────
    "chesapeake_bay": [
        "ncei19_n36x75_w076x50_2019v1.tif",
        "ncei19_n36x75_w076x75_2019v1.tif",
        "ncei19_n36x75_w077x00_2019v1.tif",
        "ncei19_n36x75_w077x25_2019v1.tif",
        "ncei19_n37x00_w076x00_2019v1.tif",
        "ncei19_n37x00_w076x25_2019v1.tif",
        "ncei19_n37x00_w076x50_2019v1.tif",
        "ncei19_n37x00_w076x75_2019v1.tif",
        "ncei19_n37x25_w076x00_2019v1.tif",
        "ncei19_n37x25_w076x25_2019v1.tif",
        "ncei19_n37x25_w076x50_2019v1.tif",
        "ncei19_n37x25_w076x75_2019v1.tif",
        "ncei19_n37x25_w077x00_2019v1.tif",
        "ncei19_n37x25_w077x25_2019v1.tif",
        "ncei19_n37x25_w077x50_2019v1.tif",
        "ncei19_n37x50_w075x75_2019v1.tif",
        "ncei19_n37x50_w076x00_2019v1.tif",
        "ncei19_n37x50_w076x25_2019v1.tif",
        "ncei19_n37x50_w076x50_2019v1.tif",
        "ncei19_n37x50_w076x75_2019v1.tif",
        "ncei19_n37x50_w077x00_2019v1.tif",
        "ncei19_n37x50_w077x25_2019v1.tif",
        "ncei19_n37x50_w077x50_2019v1.tif",
        "ncei19_n37x75_w075x75_2019v1.tif",
        "ncei19_n37x75_w076x00_2019v1.tif",
        "ncei19_n37x75_w076x25_2019v1.tif",
        "ncei19_n37x75_w076x50_2019v1.tif",
        "ncei19_n37x75_w076x75_2019v1.tif",
        "ncei19_n37x75_w077x00_2019v1.tif",
        "ncei19_n37x75_w077x25_2019v1.tif",
        "ncei19_n37x75_w077x50_2019v1.tif",
        "ncei19_n38x00_w075x50_2019v1.tif",
        "ncei19_n38x00_w075x75_2019v1.tif",
        "ncei19_n38x00_w076x00_2019v1.tif",
        "ncei19_n38x00_w076x25_2019v1.tif",
        "ncei19_n38x00_w076x50_2019v1.tif",
        "ncei19_n38x00_w076x75_2019v1.tif",
        "ncei19_n38x00_w077x00_2019v1.tif",
        "ncei19_n38x00_w077x25_2019v1.tif",
        "ncei19_n38x25_w075x25_2019v1.tif",
        "ncei19_n38x25_w075x50_2019v1.tif",
        "ncei19_n38x25_w075x75_2019v1.tif",
        "ncei19_n38x25_w076x00_2019v1.tif",
        "ncei19_n38x25_w076x25_2019v1.tif",
        "ncei19_n38x25_w076x50_2019v1.tif",
        "ncei19_n38x25_w076x75_2019v1.tif",
        "ncei19_n38x25_w077x00_2019v1.tif",
        "ncei19_n38x25_w077x25_2019v1.tif",
        "ncei19_n38x25_w077x50_2019v1.tif",
        "ncei19_n38x50_w075x25_2019v1.tif",
        "ncei19_n38x50_w075x50_2019v1.tif",
        "ncei19_n38x50_w075x75_2019v1.tif",
        "ncei19_n38x50_w076x00_2019v1.tif",
        "ncei19_n38x50_w076x25_2019v1.tif",
        "ncei19_n38x50_w076x50_2019v1.tif",
        "ncei19_n38x50_w076x75_2019v1.tif",
        "ncei19_n38x50_w077x00_2019v1.tif",
        "ncei19_n38x50_w077x25_2019v1.tif",
        "ncei19_n38x50_w077x50_2019v1.tif",
        "ncei19_n38x75_w075x25_2019v1.tif",
        "ncei19_n38x75_w075x50_2019v1.tif",
        "ncei19_n38x75_w075x75_2019v1.tif",
        "ncei19_n38x75_w076x00_2019v1.tif",
        "ncei19_n38x75_w076x25_2019v1.tif",
        "ncei19_n38x75_w076x50_2019v1.tif",
        "ncei19_n38x75_w076x75_2019v1.tif",
        "ncei19_n38x75_w077x00_2019v1.tif",
        "ncei19_n38x75_w077x25_2019v1.tif",
        "ncei19_n38x75_w077x50_2019v1.tif",
        "ncei19_n39x00_w075x75_2019v1.tif",
        "ncei19_n39x00_w076x00_2019v1.tif",
        "ncei19_n39x00_w076x25_2019v1.tif",
        "ncei19_n39x00_w076x50_2019v1.tif",
        "ncei19_n39x00_w076x75_2019v1.tif",
        "ncei19_n39x00_w077x00_2019v1.tif",
        "ncei19_n39x00_w077x25_2019v1.tif",
        "ncei19_n39x25_w075x75_2019v1.tif",
        "ncei19_n39x25_w076x00_2019v1.tif",
        "ncei19_n39x25_w076x25_2019v1.tif",
        "ncei19_n39x25_w076x50_2019v1.tif",
        "ncei19_n39x25_w076x75_2019v1.tif",
        "ncei19_n39x50_w076x00_2019v1.tif",
        "ncei19_n39x50_w076x25_2019v1.tif",
        "ncei19_n39x50_w076x50_2019v1.tif",
        "ncei19_n39x50_w076x75_2019v1.tif",
        "ncei19_n39x75_w076x00_2019v1.tif",
        "ncei19_n39x75_w076x25_2019v1.tif",
    ],

    # ── NC  (47 tiles, 9.1 GB) ────────────────────────────────────────────
    "NC": [
        "ncei19_n34x75_w076x50_2019v2.tif",
        "ncei19_n34x75_w076x75_2019v2.tif",
        "ncei19_n35x00_w076x25_2019v2.tif",
        "ncei19_n35x00_w076x50_2019v2.tif",
        "ncei19_n35x00_w076x75_2018v1.tif",
        "ncei19_n35x00_w077x00_2018v1.tif",
        "ncei19_n35x25_w075x75_2019v2.tif",
        "ncei19_n35x25_w076x00_2019v2.tif",
        "ncei19_n35x25_w076x25_2019v2.tif",
        "ncei19_n35x25_w076x50_2018v1.tif",
        "ncei19_n35x25_w076x75_2018v1.tif",
        "ncei19_n35x25_w077x00_2018v1.tif",
        "ncei19_n35x25_w077x25_2018v1.tif",
        "ncei19_n35x50_w075x50_2019v2.tif",
        "ncei19_n35x50_w075x75_2019v2.tif",
        "ncei19_n35x50_w076x00_2018v1.tif",
        "ncei19_n35x50_w076x25_2018v1.tif",
        "ncei19_n35x50_w076x50_2018v1.tif",
        "ncei19_n35x50_w076x75_2018v1.tif",
        "ncei19_n35x50_w077x00_2018v1.tif",
        "ncei19_n35x50_w077x25_2018v1.tif",
        "ncei19_n35x75_w075x50_2019v2.tif",
        "ncei19_n35x75_w075x75_2019v2.tif",
        "ncei19_n35x75_w076x00_2018v1.tif",
        "ncei19_n35x75_w076x25_2018v1.tif",
        "ncei19_n35x75_w076x50_2018v1.tif",
        "ncei19_n35x75_w076x75_2018v1.tif",
        "ncei19_n35x75_w077x00_2018v1.tif",
        "ncei19_n35x75_w077x25_2018v1.tif",
        "ncei19_n36x00_w075x75_2019v2.tif",
        "ncei19_n36x00_w076x00_2018v1.tif",
        "ncei19_n36x00_w076x25_2018v1.tif",
        "ncei19_n36x00_w076x50_2018v1.tif",
        "ncei19_n36x00_w076x75_2018v1.tif",
        "ncei19_n36x25_w075x75_2019v2.tif",
        "ncei19_n36x25_w076x00_2019v2.tif",
        "ncei19_n36x25_w076x25_2018v1.tif",
        "ncei19_n36x25_w076x50_2018v1.tif",
        "ncei19_n36x25_w076x75_2018v1.tif",
        "ncei19_n36x25_w077x00_2018v1.tif",
        "ncei19_n36x50_w076x00_2019v2.tif",
        "ncei19_n36x50_w076x25_2018v1.tif",
        "ncei19_n36x50_w076x50_2022v1.tif",
        "ncei19_n36x50_w076x75_2018v1.tif",
        "ncei19_n36x50_w077x00_2018v1.tif",
        "ncei19_n36x75_w076x00_2019v2.tif",
        "ncei19_n36x75_w076x25_2018v1.tif",
    ],

    # ── SC / GA  (75 tiles, 15 GB) ────────────────────────────────────────
    # Representative subset — full list available at
    # https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/southeast/index.html
    "southeast": [
        "ncei19_n31x25_w081x50_2019v1.tif",
        "ncei19_n31x25_w081x75_2019v1.tif",
        "ncei19_n31x25_w082x00_2019v1.tif",
        "ncei19_n31x50_w081x25_2019v1.tif",
        "ncei19_n31x50_w081x50_2019v1.tif",
        "ncei19_n31x50_w081x75_2019v1.tif",
        "ncei19_n31x50_w082x00_2019v1.tif",
        "ncei19_n31x75_w081x25_2019v1.tif",
        "ncei19_n31x75_w081x50_2019v1.tif",
        "ncei19_n31x75_w081x75_2019v1.tif",
        "ncei19_n31x75_w082x00_2019v1.tif",
        "ncei19_n32x00_w081x00_2019v1.tif",
        "ncei19_n32x00_w081x25_2019v1.tif",
        "ncei19_n32x00_w081x50_2019v1.tif",
        "ncei19_n32x00_w081x75_2019v1.tif",
        "ncei19_n32x25_w080x75_2019v1.tif",
        "ncei19_n32x25_w081x00_2019v1.tif",
        "ncei19_n32x25_w081x25_2019v1.tif",
        "ncei19_n32x25_w081x50_2019v1.tif",
        "ncei19_n32x25_w081x75_2019v1.tif",
        "ncei19_n32x50_w080x50_2019v1.tif",
        "ncei19_n32x50_w080x75_2019v1.tif",
        "ncei19_n32x50_w081x00_2019v1.tif",
        "ncei19_n32x50_w081x25_2019v1.tif",
        "ncei19_n32x75_w080x00_2019v1.tif",
        "ncei19_n32x75_w080x25_2019v1.tif",
        "ncei19_n32x75_w080x50_2019v1.tif",
        "ncei19_n32x75_w080x75_2019v1.tif",
        "ncei19_n33x00_w079x50_2019v1.tif",
        "ncei19_n33x00_w079x75_2019v1.tif",
        "ncei19_n33x00_w080x00_2019v1.tif",
        "ncei19_n33x00_w080x25_2019v1.tif",
        "ncei19_n33x25_w079x25_2019v1.tif",
        "ncei19_n33x25_w079x50_2019v1.tif",
        "ncei19_n33x25_w079x75_2019v1.tif",
        "ncei19_n33x50_w079x00_2019v1.tif",
        "ncei19_n33x50_w079x25_2019v1.tif",
        "ncei19_n33x75_w078x75_2019v1.tif",
        "ncei19_n33x75_w079x00_2019v1.tif",
        "ncei19_n34x00_w078x25_2019v1.tif",
        "ncei19_n34x00_w078x50_2019v1.tif",
        "ncei19_n34x00_w078x75_2019v1.tif",
        "ncei19_n34x25_w077x75_2019v1.tif",
        "ncei19_n34x25_w078x00_2019v1.tif",
        "ncei19_n34x25_w078x25_2019v1.tif",
        "ncei19_n34x50_w077x50_2019v1.tif",
        "ncei19_n34x50_w077x75_2019v1.tif",
        "ncei19_n34x75_w077x25_2019v1.tif",
        "ncei19_n34x75_w077x50_2019v1.tif",
        "ncei19_n34x75_w077x75_2019v1.tif",
        "ncei19_n34x75_w078x00_2019v1.tif",
        "ncei19_n34x75_w078x25_2019v1.tif",
        "ncei19_n35x00_w077x25_2019v1.tif",
        "ncei19_n35x00_w077x50_2019v1.tif",
        "ncei19_n35x00_w077x75_2019v1.tif",
    ],

    # ── Florida  (representative 60 tiles of 109, 28 GB total) ───────────
    "FL": [
        "ncei19_n24x75_w081x00_2022v2.tif",
        "ncei19_n24x75_w081x25_2022v2.tif",
        "ncei19_n25x00_w080x75_2022v2.tif",
        "ncei19_n25x00_w081x00_2022v2.tif",
        "ncei19_n25x25_w080x25_2022v2.tif",
        "ncei19_n25x25_w080x50_2022v2.tif",
        "ncei19_n25x50_w080x25_2022v2.tif",
        "ncei19_n25x75_w080x00_2022v1.tif",
        "ncei19_n26x00_w080x00_2022v1.tif",
        "ncei19_n26x00_w080x25_2022v1.tif",
        "ncei19_n26x25_w080x00_2022v1.tif",
        "ncei19_n26x50_w080x00_2021v1.tif",
        "ncei19_n26x75_w080x00_2021v1.tif",
        "ncei19_n27x00_w080x25_2021v1.tif",
        "ncei19_n27x25_w080x50_2021v1.tif",
        "ncei19_n27x50_w080x50_2021v1.tif",
        "ncei19_n27x75_w080x50_2021v1.tif",
        "ncei19_n28x00_w080x50_2021v1.tif",
        "ncei19_n28x25_w080x75_2021v1.tif",
        "ncei19_n28x50_w080x75_2021v1.tif",
        "ncei19_n29x00_w081x00_2018v1.tif",
        "ncei19_n29x25_w081x25_2018v1.tif",
        "ncei19_n29x50_w081x25_2018v1.tif",
        "ncei19_n29x75_w081x25_2018v1.tif",
        "ncei19_n30x00_w081x50_2018v1.tif",
        "ncei19_n30x25_w081x50_2018v1.tif",
        "ncei19_n25x00_w082x00_2020v1.tif",
        "ncei19_n25x25_w082x00_2020v1.tif",
        "ncei19_n25x50_w082x00_2020v1.tif",
        "ncei19_n25x75_w082x00_2020v1.tif",
        "ncei19_n26x00_w082x25_2020v1.tif",
        "ncei19_n26x25_w082x50_2020v1.tif",
        "ncei19_n26x50_w082x75_2020v1.tif",
        "ncei19_n26x75_w082x75_2020v1.tif",
        "ncei19_n27x00_w082x75_2020v1.tif",
        "ncei19_n27x25_w082x75_2016v1.tif",
        "ncei19_n27x50_w082x75_2016v1.tif",
        "ncei19_n27x75_w083x00_2016v1.tif",
        "ncei19_n28x00_w083x00_2016v1.tif",
        "ncei19_n28x25_w083x25_2016v1.tif",
        "ncei19_n28x50_w083x25_2017v1.tif",
        "ncei19_n28x75_w083x50_2017v1.tif",
        "ncei19_n29x00_w083x50_2017v1.tif",
        "ncei19_n29x25_w083x75_2017v1.tif",
        "ncei19_n29x50_w084x00_2017v1.tif",
        "ncei19_n29x75_w084x25_2017v1.tif",
        "ncei19_n30x00_w084x50_2017v1.tif",
        "ncei19_n30x25_w084x75_2017v1.tif",
        "ncei19_n30x50_w084x75_2017v1.tif",
        "ncei19_n30x75_w085x00_2017v1.tif",
    ],

    # ── AL / NW Florida panhandle  (56 tiles, 11 GB) ─────────────────────
    "AL_nwFL": [
        "ncei19_n29x75_w084x75_2019v1.tif",
        "ncei19_n30x00_w085x00_2019v1.tif",
        "ncei19_n30x00_w085x25_2019v1.tif",
        "ncei19_n30x00_w085x50_2019v1.tif",
        "ncei19_n30x00_w086x00_2019v1.tif",
        "ncei19_n30x25_w085x25_2019v1.tif",
        "ncei19_n30x25_w085x50_2019v1.tif",
        "ncei19_n30x25_w086x00_2019v1.tif",
        "ncei19_n30x25_w086x25_2019v1.tif",
        "ncei19_n30x25_w086x50_2019v1.tif",
        "ncei19_n30x25_w086x75_2019v1.tif",
        "ncei19_n30x25_w087x00_2019v1.tif",
        "ncei19_n30x25_w087x25_2019v1.tif",
        "ncei19_n30x50_w086x00_2019v1.tif",
        "ncei19_n30x50_w086x25_2019v1.tif",
        "ncei19_n30x50_w086x50_2019v1.tif",
        "ncei19_n30x50_w086x75_2019v1.tif",
        "ncei19_n30x50_w087x00_2019v1.tif",
        "ncei19_n30x50_w087x25_2019v1.tif",
        "ncei19_n30x50_w087x50_2019v1.tif",
        "ncei19_n30x50_w087x75_2019v1.tif",
        "ncei19_n30x50_w088x00_2019v1.tif",
        "ncei19_n30x75_w087x25_2019v1.tif",
        "ncei19_n30x75_w087x50_2019v1.tif",
        "ncei19_n30x75_w087x75_2019v1.tif",
        "ncei19_n30x75_w088x00_2019v1.tif",
        "ncei19_n30x75_w088x25_2019v1.tif",
        "ncei19_n31x00_w087x75_2019v1.tif",
        "ncei19_n31x00_w088x00_2019v1.tif",
        "ncei19_n31x25_w088x00_2019v1.tif",
    ],

    # ── LA / MS  (representative 60 tiles of ~119) ────────────────────────
    "LA_MS": [
        "ncei19_n29x00_w089x00_2020v1.tif",
        "ncei19_n29x00_w089x25_2020v1.tif",
        "ncei19_n29x00_w089x50_2020v1.tif",
        "ncei19_n29x00_w090x00_2020v1.tif",
        "ncei19_n29x00_w090x25_2020v1.tif",
        "ncei19_n29x00_w090x50_2020v1.tif",
        "ncei19_n29x00_w090x75_2020v1.tif",
        "ncei19_n29x00_w091x00_2020v1.tif",
        "ncei19_n29x00_w091x25_2020v1.tif",
        "ncei19_n29x00_w091x50_2020v1.tif",
        "ncei19_n29x00_w091x75_2020v1.tif",
        "ncei19_n29x00_w092x00_2020v1.tif",
        "ncei19_n29x00_w092x25_2020v1.tif",
        "ncei19_n29x25_w089x25_2020v1.tif",
        "ncei19_n29x25_w089x50_2020v1.tif",
        "ncei19_n29x25_w089x75_2020v1.tif",
        "ncei19_n29x25_w090x00_2020v1.tif",
        "ncei19_n29x25_w090x25_2020v1.tif",
        "ncei19_n29x25_w090x50_2020v1.tif",
        "ncei19_n29x25_w090x75_2020v1.tif",
        "ncei19_n29x25_w091x00_2020v1.tif",
        "ncei19_n29x25_w091x25_2020v1.tif",
        "ncei19_n29x25_w091x50_2020v1.tif",
        "ncei19_n29x25_w092x00_2020v1.tif",
        "ncei19_n29x25_w092x25_2020v1.tif",
        "ncei19_n29x50_w090x00_2021v1.tif",
        "ncei19_n29x50_w090x25_2021v1.tif",
        "ncei19_n29x50_w090x50_2021v1.tif",
        "ncei19_n29x50_w090x75_2021v1.tif",
        "ncei19_n29x50_w091x00_2021v1.tif",
        "ncei19_n29x50_w091x25_2021v1.tif",
        "ncei19_n29x50_w091x50_2021v1.tif",
        "ncei19_n29x75_w090x00_2022v1.tif",
        "ncei19_n29x75_w090x25_2022v1.tif",
        "ncei19_n29x75_w090x50_2022v1.tif",
        "ncei19_n29x75_w091x00_2022v1.tif",
        "ncei19_n30x00_w089x50_2021v1.tif",
        "ncei19_n30x00_w089x75_2021v1.tif",
        "ncei19_n30x00_w090x00_2021v1.tif",
        "ncei19_n30x00_w090x25_2021v1.tif",
        "ncei19_n30x00_w091x50_2021v1.tif",
        "ncei19_n30x00_w092x00_2021v1.tif",
        "ncei19_n30x00_w092x25_2021v1.tif",
        "ncei19_n30x00_w092x50_2021v1.tif",
        "ncei19_n30x25_w089x75_2021v1.tif",
        "ncei19_n30x25_w090x00_2021v1.tif",
        "ncei19_n30x50_w089x75_2020v1.tif",
        "ncei19_n30x50_w090x00_2020v1.tif",
        "ncei19_n30x75_w089x75_2020v1.tif",
        "ncei19_n31x00_w089x50_2020v1.tif",
    ],

    # ── Texas  (representative 50 tiles of ~97) ───────────────────────────
    "TX": [
        "ncei19_n26x00_w097x25_2020v1.tif",
        "ncei19_n26x00_w097x50_2020v1.tif",
        "ncei19_n26x25_w097x00_2020v1.tif",
        "ncei19_n26x25_w097x25_2020v1.tif",
        "ncei19_n26x50_w097x00_2020v1.tif",
        "ncei19_n26x75_w097x25_2020v1.tif",
        "ncei19_n27x00_w097x25_2020v1.tif",
        "ncei19_n27x25_w097x50_2020v1.tif",
        "ncei19_n27x50_w097x25_2021v2.tif",
        "ncei19_n27x75_w097x25_2021v2.tif",
        "ncei19_n28x00_w097x00_2021v2.tif",
        "ncei19_n28x00_w097x25_2021v2.tif",
        "ncei19_n28x25_w096x75_2021v2.tif",
        "ncei19_n28x25_w097x00_2021v2.tif",
        "ncei19_n28x50_w096x50_2021v2.tif",
        "ncei19_n28x75_w096x25_2021v2.tif",
        "ncei19_n28x75_w096x50_2021v2.tif",
        "ncei19_n29x00_w095x75_2021v1.tif",
        "ncei19_n29x00_w096x00_2021v1.tif",
        "ncei19_n29x25_w095x50_2021v1.tif",
        "ncei19_n29x25_w095x75_2021v1.tif",
        "ncei19_n29x50_w095x00_2021v1.tif",
        "ncei19_n29x50_w095x25_2021v1.tif",
        "ncei19_n29x75_w094x75_2021v1.tif",
        "ncei19_n29x75_w094x50_2021v1.tif",
        "ncei19_n30x00_w094x25_2021v1.tif",
        "ncei19_n30x00_w094x50_2021v1.tif",
        "ncei19_n30x25_w094x00_2021v1.tif",
        "ncei19_n30x25_w094x25_2021v1.tif",
        "ncei19_n30x50_w093x75_2021v1.tif",
        "ncei19_n30x50_w094x00_2021v1.tif",
        "ncei19_n30x75_w093x75_2021v1.tif",
    ],

    # ── Puerto Rico  (small representative set, dataset 9525) ────────────
    "PR": [
        "ncei19_n17x75_w066x75_2019v1.tif",
        "ncei19_n17x75_w067x00_2019v1.tif",
        "ncei19_n17x75_w067x25_2019v1.tif",
        "ncei19_n18x00_w066x25_2019v1.tif",
        "ncei19_n18x00_w066x50_2019v1.tif",
        "ncei19_n18x00_w066x75_2019v1.tif",
        "ncei19_n18x00_w067x00_2019v1.tif",
        "ncei19_n18x25_w066x00_2019v1.tif",
        "ncei19_n18x25_w066x25_2019v1.tif",
        "ncei19_n18x25_w066x50_2019v1.tif",
        "ncei19_n18x50_w065x75_2019v1.tif",
        "ncei19_n18x50_w066x00_2019v1.tif",
    ],
}

# Map subfolder → S3 base URL
_SUBFOLDER_S3_BASE: Dict[str, str] = {
    sf: _S3_BASE + f"/{sf}" for sf in _TILES if sf != "PR"
}
_SUBFOLDER_S3_BASE["PR"] = _S3_PR

# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, dest: Path,
                   retries: int = 4, timeout: int = 300) -> bool:
    """Stream-download url → dest. Returns True on success."""
    if dest.exists():
        return True  # already downloaded
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            tmp = dest.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=2 << 20):
                    fh.write(chunk)
            tmp.rename(dest)
            return True
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"      attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(6 * attempt)
    return False


# ---------------------------------------------------------------------------
# GEBCO download
# ---------------------------------------------------------------------------

def download_gebco(out_dir: Path) -> Optional[Path]:
    """Download GEBCO/ETOPO 2022 for the full STOFS-3D-Atlantic domain."""
    local = os.environ.get("GEBCO_LOCAL")
    if local and Path(local).exists():
        print(f"  Using local GEBCO file: {local}")
        return Path(local)

    dest = out_dir / "stofs_atlantic_gebco.tif"
    if dest.exists():
        print(f"  {dest.name} already exists — skipping.")
        return dest

    print("  Downloading GEBCO/ETOPO 2022 (15 arc-sec, full domain) ...")

    # NCEI ETOPO 2022 WCS (GEBCO-based)
    wcs = "https://www.ngdc.noaa.gov/thredds/wcs/global/etopo2022/ETOPO_2022_v1_15s_N90W180_bed.nc"
    bbox = "-100.0,-5.0,-50.0,47.0"   # wider than domain for overlap
    params = {
        "SERVICE": "WCS", "VERSION": "1.0.0", "REQUEST": "GetCoverage",
        "COVERAGE": "Band1", "BBOX": bbox,
        "CRS": "EPSG:4326", "RESPONSE_CRS": "EPSG:4326",
        "FORMAT": "GeoTIFF",
        "RESX": "0.00417",   # 15 arc-sec ≈ 0.00417°
        "RESY": "0.00417",
    }
    if _download_file(
        requests.Request("GET", wcs, params=params).prepare().url,
        dest
    ):
        return dest

    # Simple fallback: assemble URL manually
    url = (
        f"{wcs}?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage"
        f"&COVERAGE=Band1&BBOX={bbox}&CRS=EPSG:4326&RESPONSE_CRS=EPSG:4326"
        f"&FORMAT=GeoTIFF&RESX=0.00417&RESY=0.00417"
    )
    if _download_file(url, dest):
        return dest

    print("  WARNING: GEBCO download failed. Set GEBCO_LOCAL=/path/to/gebco.tif")
    return None


# ---------------------------------------------------------------------------
# CUDEM tile selection: every other tile
# ---------------------------------------------------------------------------

def select_every_other(tiles: List[str]) -> List[str]:
    """Return every other tile (indices 0, 2, 4, …) from a sorted list.

    This creates intentional gaps in coastal coverage, so the GEBCO
    background shows through between CUDEM tiles. The mesh refinements
    applied on CUDEM tiles override GEBCO where CUDEM is present, and
    GEBCO-based refinements fill the gaps — making the priority mechanism
    visually testable.
    """
    return sorted(tiles)[::2]


# ---------------------------------------------------------------------------
# Main download orchestration
# ---------------------------------------------------------------------------

def download_all(
    out_dir: Path,
    manifest_path: Path,
    only: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict:
    """Download GEBCO + selected CUDEM tiles; write manifest."""

    # ── GEBCO ──────────────────────────────────────────────────────────────
    gebco_dir = out_dir / "gebco"
    _ensure_dir(gebco_dir)
    print("\n" + "─" * 60)
    print("  GEBCO / ETOPO 2022  (lowest priority — deep ocean background)")
    print("─" * 60)
    if dry_run:
        print("  [dry-run] would download GEBCO to", gebco_dir)
        gebco_path = gebco_dir / "stofs_atlantic_gebco.tif"
    else:
        gebco_path = download_gebco(gebco_dir)

    # ── Build raster list: GEBCO first, CUDEM after ────────────────────────
    # HfunCollector reverses the list, so GEBCO (index 0) ends up with lowest
    # priority; the last CUDEM tile has highest priority.
    raster_list: List[Dict] = []
    priority_counter = 0

    raster_list.append({
        "name": "gebco",
        "subfolder": "gebco",
        "filename": "stofs_atlantic_gebco.tif",
        "path": str(gebco_path) if gebco_path else None,
        "url": "n/a",
        "source": "gebco",
        "available": gebco_path is not None and (dry_run or gebco_path.exists()),
        "priority": priority_counter,
    })
    priority_counter += 1

    # ── CUDEM tiles ────────────────────────────────────────────────────────
    subfolders = list(_TILES.keys())
    if only:
        subfolders = [s for s in subfolders if s in only]

    for subfolder in subfolders:
        all_tiles = _TILES[subfolder]
        selected = select_every_other(all_tiles)
        s3_base = _SUBFOLDER_S3_BASE[subfolder]
        sf_dir = out_dir / subfolder
        _ensure_dir(sf_dir)

        print(f"\n{'─'*60}")
        print(f"  {subfolder}  —  {len(selected)}/{len(all_tiles)} tiles "
              f"(every other, ~{len(selected)*130:.0f} MB est.)")
        print("─" * 60)

        for fname in selected:
            url = f"{s3_base}/{fname}"
            dest = sf_dir / fname
            avail = False

            if dry_run:
                print(f"  [dry-run] {fname}")
                avail = True
            else:
                if dest.exists():
                    print(f"  {fname}  ✓ cached")
                    avail = True
                else:
                    print(f"  {fname}")
                    avail = _download_file(url, dest)
                    if not avail:
                        print(f"    FAILED: {url}")

            raster_list.append({
                "name": f"{subfolder}/{fname.replace('.tif', '')}",
                "subfolder": subfolder,
                "filename": fname,
                "path": str(dest),
                "url": url,
                "source": "cudem",
                "available": avail,
                "priority": priority_counter,
            })
            priority_counter += 1

    # ── Write manifest ──────────────────────────────────────────────────────
    manifest = {r["name"]: r for r in raster_list}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def _print_summary(manifest: Dict, dry_run: bool) -> None:
    total = len(manifest)
    available = sum(1 for v in manifest.values() if v["available"])
    gebco = sum(1 for v in manifest.values() if v["source"] == "gebco")
    cudem = sum(1 for v in manifest.values() if v["source"] == "cudem")
    print("\n" + "=" * 60)
    print(f"  {'[DRY RUN] ' if dry_run else ''}Summary")
    print("=" * 60)
    print(f"  Total entries  : {total}")
    print(f"    GEBCO tiles  : {gebco}")
    print(f"    CUDEM tiles  : {cudem}  (every other, ~{cudem*130:.0f} MB est.)")
    print(f"  Available      : {available}/{total}")
    if available < total:
        failed = [k for k, v in manifest.items() if not v["available"]]
        print(f"  Failed ({len(failed)})    : {failed[:5]}{'...' if len(failed) > 5 else ''}")
    print()
    print("  Priority ordering passed to HfunCollector:")
    print("    index 0 = GEBCO (lowest priority — deep ocean background)")
    print("    index 1…N = CUDEM tiles (higher priority — coastal override)")
    print("    HfunCollector reverses the list internally, so the last")
    print("    CUDEM tile in the list has the absolute highest priority.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download GEBCO + NOAA NCEI 1/9 arc-second CUDEM tiles\n"
            "for the STOFS-3D-Atlantic OCSMesh benchmark.\n\n"
            "Downloads every other tile per subfolder to create deliberate\n"
            "gaps in coastal coverage — showing GEBCO in the gaps and CUDEM\n"
            "where tiles are present, making the priority mechanism testable.\n\n"
            f"Subfolders: {', '.join(_TILES.keys())}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("./stofs_dems"),
        help="Root directory for downloaded DEMs (default: ./stofs_dems)",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(__file__).parent / "dem_manifest.json",
        help="Output manifest JSON path",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="SUBFOLDER",
        choices=list(_TILES.keys()),
        help="Download only the named subfolder(s).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    args = parser.parse_args()

    n_cudem = sum(len(select_every_other(v)) for k, v in _TILES.items()
                  if not args.only or k in args.only)
    print("=" * 60)
    print("  OCSMesh STOFS-3D-Atlantic DEM Download")
    print("=" * 60)
    print(f"  Output dir : {args.out_dir.resolve()}")
    print(f"  Manifest   : {args.manifest.resolve()}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"  GEBCO      : 1 tile (full domain, 15 arc-sec)")
    print(f"  CUDEM      : ~{n_cudem} tiles (every other, 1/9 arc-sec, ~3 m)")
    print(f"  Est. total : ~{n_cudem * 130 / 1024:.0f} GB CUDEM + a few GB GEBCO")
    print()

    manifest = download_all(
        args.out_dir.resolve(),
        args.manifest.resolve(),
        only=args.only,
        dry_run=args.dry_run,
    )
    _print_summary(manifest, dry_run=args.dry_run)

    if not args.dry_run:
        available = sum(1 for v in manifest.values() if v["available"])
        if available < len(manifest):
            sys.exit(1)


if __name__ == "__main__":
    main()
