"""NJ_test/download_third.py

Download NCEI third arc-second (1/3", ~10m) topobathy tiles for the NJ domain.

Dataset: NCEI_third_Topobathy_2014_8580
Priority: MIDDLE (above GEBCO, below CUDEM ninth arc-second)

20 tiles intersect the NJ domain from the northeast_sandy subfolder.

Usage
-----
    python download_third.py --dem-dir ./NJ_dems
    python download_third.py --dem-dir ./NJ_dems --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from dem_config import (
    THIRD_S3_BASE,
    THIRD_INDEX_BASE,
    THIRD_SUBFOLDERS,
    THIRD_TILE_SIZE_DEG,
    DEFAULT_DEM_DIR,
    SOURCE_THIRD,
)
from tile_utils import (
    download_file,
    get_subfolder_tiles,
    tile_intersects_domain,
)


def download_third(
    dem_dir: Path,
    subfolders: Optional[List[str]] = None,
    dry_run: bool = False,
) -> List[Dict]:
    """Download NCEI third arc-second tiles intersecting the NJ domain.

    Returns a list of manifest entry dicts.
    """
    if subfolders is None:
        subfolders = list(THIRD_SUBFOLDERS)

    entries: List[Dict] = []

    for subfolder in subfolders:
        print()
        print("─" * 60)
        print(f"  NCEI third arc-second / {subfolder}")
        print("─" * 60)

        all_tiles = get_subfolder_tiles(THIRD_INDEX_BASE, subfolder)
        if not all_tiles:
            print(f"  WARNING: no tiles scraped for {subfolder} — skipping.")
            continue

        nj_tiles = [
            t for t in all_tiles
            if tile_intersects_domain(t, tile_size_deg=THIRD_TILE_SIZE_DEG)
        ]
        print(f"  {len(nj_tiles)} / {len(all_tiles)} tiles intersect NJ domain")

        sf_dir = dem_dir / f"third_{subfolder}"
        sf_dir.mkdir(parents=True, exist_ok=True)

        for fname in nj_tiles:
            url  = f"{THIRD_S3_BASE}/{subfolder}/{fname}"
            dest = sf_dir / fname
            avail = False

            if dry_run:
                print(f"  [dry-run] third_{subfolder}/{fname}")
                avail = True
            elif dest.exists():
                print(f"  {fname}  ✓ cached")
                avail = True
            else:
                print(f"  {fname}")
                avail = download_file(url, dest, verbose=True)
                if not avail:
                    print(f"    FAILED: {url}")

            entries.append({
                "name":      f"third_{subfolder}/{fname.replace('.tif', '')}",
                "subfolder": f"third_{subfolder}",
                "filename":  fname,
                "path":      str(dest),
                "url":       url,
                "source":    SOURCE_THIRD,
                "available": avail,
            })

    total_avail = sum(1 for e in entries if e["available"])
    print()
    print(f"  NCEI third summary: {total_avail} / {len(entries)} tiles available")
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NCEI third arc-second tiles for the NJ domain."
    )
    parser.add_argument(
        "--dem-dir", type=Path, default=Path(DEFAULT_DEM_DIR),
    )
    parser.add_argument(
        "--only", nargs="+", metavar="SUBFOLDER",
        choices=list(THIRD_SUBFOLDERS),
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    entries = download_third(
        dem_dir=args.dem_dir.resolve(),
        subfolders=args.only,
        dry_run=args.dry_run,
    )

    failed = [e for e in entries if not e["available"]]
    if failed:
        print(f"\n  {len(failed)} tiles failed:")
        for e in failed:
            print(f"    {e['name']}")


if __name__ == "__main__":
    main()
