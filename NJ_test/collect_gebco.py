"""NJ_test/collect_gebco.py

Collect the GEBCO split tiles that intersect the NJ domain.

No download is performed. The tiles were already created by
split_gebco.py and live in:
    ../stofs_dems/gebco_split/gebco_tile_*.tif

This script:
  1. Lists all *.tif files in GEBCO_SPLIT_DIR.
  2. Reads each tile's actual geographic bounds via rasterio.
  3. Returns only tiles whose bounds intersect the NJ domain
     (39–40.5 N, -75 to -73 lon).

Override the search directory via the GEBCO_SPLIT_DIR environment
variable if your layout differs.

Usage
-----
    python collect_gebco.py
    python collect_gebco.py --list-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from dem_config import (
    GEBCO_SPLIT_DIR,
    GEBCO_TILE_PATTERN,
    SOURCE_GEBCO,
    DEFAULT_DEM_DIR,
    DOMAIN_LAT_MIN, DOMAIN_LAT_MAX,
    DOMAIN_LON_MIN, DOMAIN_LON_MAX,
)
from tile_utils import gebco_tile_intersects_domain


def collect_gebco_entries(verbose: bool = True) -> List[Dict]:
    """Return manifest entries for GEBCO split tiles in the NJ domain.

    Reads actual GeoTIFF bounds via rasterio — no filename parsing needed.

    Returns
    -------
    List of manifest entry dicts, one per intersecting tile.
    Empty list if the directory does not exist or no tiles are found.
    """
    if not GEBCO_SPLIT_DIR.exists():
        print(f"  ERROR: GEBCO split directory not found: {GEBCO_SPLIT_DIR}")
        print(f"  Set GEBCO_SPLIT_DIR env var or check the path in dem_config.py")
        return []

    all_tiles = sorted(GEBCO_SPLIT_DIR.glob(GEBCO_TILE_PATTERN))
    if not all_tiles:
        print(f"  ERROR: No files matching '{GEBCO_TILE_PATTERN}' "
              f"in {GEBCO_SPLIT_DIR}")
        return []

    if verbose:
        print()
        print("─" * 60)
        print("  GEBCO split tiles  (lowest priority — ocean background)")
        print("─" * 60)
        print(f"  Directory : {GEBCO_SPLIT_DIR}")
        print(f"  All tiles : {len(all_tiles)}")

    entries: List[Dict] = []
    for tif_path in all_tiles:
        if gebco_tile_intersects_domain(tif_path):
            name = f"gebco_split/{tif_path.stem}"
            entries.append({
                "name":      name,
                "subfolder": "gebco_split",
                "filename":  tif_path.name,
                "path":      str(tif_path),
                "url":       "n/a",
                "source":    SOURCE_GEBCO,
                "available": True,
            })
            if verbose:
                print(f"  ✓  {tif_path.name}")

    if verbose:
        print(f"  NJ tiles  : {len(entries)} / {len(all_tiles)} intersect domain")

    if not entries:
        print(f"  WARNING: No GEBCO tiles intersect the NJ domain "
              f"({DOMAIN_LAT_MIN}–{DOMAIN_LAT_MAX} N, "
              f"{DOMAIN_LON_MIN}–{DOMAIN_LON_MAX} E).")
        print(f"  Check GEBCO_SPLIT_DIR: {GEBCO_SPLIT_DIR}")

    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="List GEBCO split tiles that intersect the NJ domain."
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="Print intersecting tiles and exit (no manifest output).",
    )
    args = parser.parse_args()

    entries = collect_gebco_entries(verbose=True)

    if not entries:
        sys.exit(1)

    if args.list_only:
        print(f"\nIntersecting tiles ({len(entries)}):")
        for e in entries:
            print(f"  {e['path']}")
        return

    print(f"\n{len(entries)} GEBCO tile(s) will be included in the NJ manifest.")


if __name__ == "__main__":
    main()
