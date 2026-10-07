"""NJ_test/tile_utils.py

Utility functions shared by the NJ DEM download and manifest scripts.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from dem_config import (
    DOMAIN_LAT_MIN, DOMAIN_LAT_MAX,
    DOMAIN_LON_MIN, DOMAIN_LON_MAX,
    TILE_INTERSECT_BUFFER_DEG,
)

# ---------------------------------------------------------------------------
# GEBCO split-tile intersection (reads actual GeoTIFF bounds)
# ---------------------------------------------------------------------------

def gebco_tile_intersects_domain(tif_path: Path) -> bool:
    """Return True if a GEBCO split GeoTIFF intersects the NJ domain.

    Reads bounds directly from file metadata via rasterio.
    """
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            b = src.bounds
            tile_w, tile_s, tile_e, tile_n = (
                b.left, b.bottom, b.right, b.top
            )
    except Exception as exc:
        print(f"  WARNING: could not read bounds of {tif_path.name}: {exc}")
        return True   # include conservatively

    buf = TILE_INTERSECT_BUFFER_DEG
    dom_w = DOMAIN_LON_MIN - buf
    dom_e = DOMAIN_LON_MAX + buf
    dom_s = DOMAIN_LAT_MIN - buf
    dom_n = DOMAIN_LAT_MAX + buf

    no_overlap = (
        tile_e < dom_w or tile_w > dom_e or
        tile_n < dom_s or tile_s > dom_n
    )
    return not no_overlap


# ---------------------------------------------------------------------------
# CUDEM / CoNED filename-based intersection
# ---------------------------------------------------------------------------

# Matches NCEI tile filenames:
#   ncei19_n39x50_w074x75_2018v2.tif
_NCEI_TIF_RE = re.compile(
    r"ncei\d+_[nN](\d+)[xX](\d+)_[wWeE](\d+)[xX](\d+)_\d{4}v\d+\.tif",
    re.IGNORECASE,
)

# Matches href links on index pages — captures only the href value
# We only want bare filenames or relative paths, not full http:// URLs
_HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)


def parse_tile_sw_corner(filename: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) of the SW corner from an NCEI tile filename.

    e.g. ncei19_n39x50_w074x75_2018v2.tif -> (39.50, -74.75)
    Returns None if filename does not match.
    """
    m = _NCEI_TIF_RE.search(filename)
    if m is None:
        return None
    lat_deg, lat_frac, lon_deg, lon_frac = (int(x) for x in m.groups())
    lat = lat_deg + lat_frac / 100.0
    lon = -(lon_deg + lon_frac / 100.0)   # West → negative
    return lat, lon


def tile_extent(
    filename: str,
    tile_size_deg: float = 0.25,
) -> Optional[Tuple[float, float, float, float]]:
    """Return (west, south, east, north) extent of an NCEI tile.

    Returns None if filename cannot be parsed.
    """
    corner = parse_tile_sw_corner(filename)
    if corner is None:
        return None
    lat, lon = corner
    return (lon, lat, lon + tile_size_deg, lat + tile_size_deg)


def tile_intersects_domain(
    filename: str,
    tile_size_deg: float = 0.25,
    buf: float = TILE_INTERSECT_BUFFER_DEG,
) -> bool:
    """Return True if the NCEI tile overlaps the NJ domain.

    Falls back to True (include) if filename cannot be parsed.
    """
    ext = tile_extent(filename, tile_size_deg)
    if ext is None:
        return True   # conservative

    tile_w, tile_s, tile_e, tile_n = ext

    dom_w = DOMAIN_LON_MIN - buf
    dom_e = DOMAIN_LON_MAX + buf
    dom_s = DOMAIN_LAT_MIN - buf
    dom_n = DOMAIN_LAT_MAX + buf

    no_overlap = (
        tile_e < dom_w or tile_w > dom_e or
        tile_n < dom_s or tile_s > dom_n
    )
    return not no_overlap


# ---------------------------------------------------------------------------
# Index-page scraping
# ---------------------------------------------------------------------------

_INDEX_CACHE: Dict[str, List[str]] = {}


def scrape_tif_links(index_url: str, timeout: int = 120) -> List[str]:
    """Scrape bare *.tif filenames from an HTML index page.

    Extracts href values, keeps only those that:
      - end with .tif
      - match the NCEI tile filename pattern
      - are bare filenames (no http:// — avoids S3 redirect links)

    Returns a sorted, deduplicated list of bare filenames.
    Returns [] on failure.
    """
    if index_url in _INDEX_CACHE:
        return _INDEX_CACHE[index_url]

    try:
        resp = requests.get(index_url, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  WARNING: could not fetch {index_url}: {exc}")
        _INDEX_CACHE[index_url] = []
        return []

    raw_hrefs = _HREF_RE.findall(resp.text)

    names = []
    for href in raw_hrefs:
        # Strip any leading path components — keep only the filename
        bare = href.split("/")[-1]
        # Must match NCEI tile pattern and end in .tif
        if bare.endswith(".tif") and _NCEI_TIF_RE.search(bare):
            names.append(bare)

    names = sorted(set(names))
    _INDEX_CACHE[index_url] = names
    return names


def get_subfolder_tiles(
    index_base: str,
    subfolder: str,
    timeout: int = 120,
) -> List[str]:
    """Return sorted list of bare *.tif filenames in a subfolder index."""
    url = f"{index_base}/{subfolder}/index.html"
    return scrape_tif_links(url, timeout=timeout)


# ---------------------------------------------------------------------------
# File download
# ---------------------------------------------------------------------------

def download_file(
    url: str,
    dest: Path,
    retries: int = 4,
    timeout: int = 300,
    verbose: bool = True,
) -> bool:
    """Stream-download url → dest. Returns True on success.

    Skips if dest already exists. Uses a .tmp staging file.
    """
    if dest.exists():
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=2 << 20):
                    fh.write(chunk)
            tmp.rename(dest)
            return True
        except Exception as exc:
            if verbose:
                print(f"      attempt {attempt}/{retries} failed: {exc}")
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(6 * attempt)

    return False
