"""split_gebco.py — Split the GEBCO background tile into an NxN grid.

Splits the single large GEBCO GeoTIFF (12000×12240 px, full Atlantic domain)
into an NxN grid of smaller overlapping tiles. Each sub-tile is saved as a
separate GeoTIFF and added to a new manifest.

Why:
    With 387 CUDEM ranks each finishing in ~89 min, the single GEBCO rank
    (estimated ~131 min for Config F) becomes the critical path that makes
    all CUDEM ranks wait. An 8×8 split reduces each GEBCO sub-tile to ~2 min,
    eliminating the bottleneck.

Output:
    <out-dir>/gebco_split/gebco_tile_<row>_<col>.tif   (NxN tiles)
    <out-dir>/dem_manifest_full_split.json             (387 CUDEM + NxN GEBCO)

Usage:
    # Default: 8x8 split, 0.1 degree overlap
    python split_gebco.py \
        --manifest dem_manifest.json \
        --out-dir  /work2/noaa/nos-surge/felicioc/OCSMesh_MPI/stofs_dems \
        --n        8 \
        --overlap  0.1

    # Dry run: print tile extents without writing files
    python split_gebco.py \
        --manifest dem_manifest.json \
        --out-dir  /work2/noaa/nos-surge/felicioc/OCSMesh_MPI/stofs_dems \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds


# ---------------------------------------------------------------------------
# Core splitting logic
# ---------------------------------------------------------------------------

def compute_tile_extents(
    bounds: Tuple[float, float, float, float],
    n: int,
    overlap: float,
) -> List[Tuple[float, float, float, float]]:
    """Compute (left, bottom, right, top) extents for an NxN grid.

    Parameters
    ----------
    bounds : (left, bottom, right, top)
        Geographic extent of the source raster in its native CRS.
    n : int
        Number of tiles per dimension (n×n total tiles).
    overlap : float
        Overlap in degrees between adjacent tiles. Prevents seam
        artifacts at tile boundaries when OCSMesh clips rasters.

    Returns
    -------
    list of (left, bottom, right, top) tuples, length n*n,
    ordered row-major (top to bottom, left to right).
    """
    left, bottom, right, top = bounds
    lon_step = (right - left) / n
    lat_step = (top - bottom) / n

    extents = []
    for row in range(n):
        for col in range(n):
            tile_left   = left   + col * lon_step - overlap
            tile_right  = left   + (col + 1) * lon_step + overlap
            tile_bottom = bottom + row * lat_step - overlap
            tile_top    = bottom + (row + 1) * lat_step + overlap

            # Clamp to source bounds
            tile_left   = max(tile_left,   left)
            tile_right  = min(tile_right,  right)
            tile_bottom = max(tile_bottom, bottom)
            tile_top    = min(tile_top,    top)

            extents.append((tile_left, tile_bottom, tile_right, tile_top))

    return extents


def split_gebco(
    src_path: Path,
    out_dir: Path,
    n: int = 8,
    overlap: float = 0.1,
    dry_run: bool = False,
) -> List[Path]:
    """Split a GeoTIFF into an NxN grid of overlapping tiles.

    Parameters
    ----------
    src_path : Path
        Source GEBCO GeoTIFF.
    out_dir : Path
        Directory to write sub-tiles into.
    n : int
        Grid size (n×n tiles).
    overlap : float
        Overlap in degrees.
    dry_run : bool
        If True, print extents without writing files.

    Returns
    -------
    List of paths to written sub-tile files (empty if dry_run).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    with rasterio.open(src_path) as src:
        bounds  = src.bounds          # (left, bottom, right, top)
        profile = src.profile.copy()
        crs     = src.crs

        print(f"Source GEBCO tile:")
        print(f"  Path   : {src_path}")
        print(f"  Size   : {src.width}×{src.height} px")
        print(f"  CRS    : {crs}")
        print(f"  Bounds : lon [{bounds.left:.4f}, {bounds.right:.4f}]  "
              f"lat [{bounds.bottom:.4f}, {bounds.top:.4f}]")
        print(f"  Splitting into {n}×{n} = {n*n} tiles "
              f"with {overlap}° overlap")
        print()

        extents = compute_tile_extents(
            (bounds.left, bounds.bottom, bounds.right, bounds.top),
            n, overlap,
        )

        for idx, (tile_left, tile_bottom, tile_right, tile_top) in \
                enumerate(extents):
            row = idx // n
            col = idx  % n
            out_path = out_dir / f"gebco_tile_{row:02d}_{col:02d}.tif"

            # Pixel extent of this tile
            win = window_from_bounds(
                tile_left, tile_bottom, tile_right, tile_top,
                transform=src.transform,
            )
            win_data = src.read(window=win)
            win_h, win_w = win_data.shape[1], win_data.shape[2]

            print(f"  Tile [{row:02d},{col:02d}]  "
                  f"lon [{tile_left:.4f}, {tile_right:.4f}]  "
                  f"lat [{tile_bottom:.4f}, {tile_top:.4f}]  "
                  f"{win_w}×{win_h} px")

            if dry_run:
                continue

            if out_path.exists():
                print(f"    -> already exists, skipping")
                written.append(out_path)
                continue

            new_transform = from_bounds(
                tile_left, tile_bottom, tile_right, tile_top,
                win_w, win_h,
            )
            new_profile = profile.copy()
            new_profile.update({
                "width":     win_w,
                "height":    win_h,
                "transform": new_transform,
                "driver":    "GTiff",
                "compress":  "lzw",
                "tiled":     True,
                "blockxsize": 256,
                "blockysize": 256,
            })

            with rasterio.open(out_path, "w", **new_profile) as dst:
                dst.write(win_data)

            print(f"    -> written: {out_path.name}")
            written.append(out_path)

    return written


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_split_manifest(
    original_manifest_path: Path,
    gebco_tile_paths: List[Path],
    out_manifest_path: Path,
) -> Dict:
    """Build a new manifest replacing the single GEBCO entry with NxN tiles.

    The split GEBCO tiles are assigned priority 0..N-1 (lowest), and all
    CUDEM tiles are renumbered to priorities N..N+387 (higher). This
    preserves the GEBCO-as-background-tile priority ordering.

    Parameters
    ----------
    original_manifest_path : Path
        The full 387-tile manifest (dem_manifest.json).
    gebco_tile_paths : List[Path]
        Paths to the written GEBCO sub-tiles.
    out_manifest_path : Path
        Where to write the new manifest.

    Returns
    -------
    The new manifest dict.
    """
    original = json.loads(original_manifest_path.read_text())

    # Separate GEBCO and CUDEM entries
    cudem_entries = [
        (k, v) for k, v in original.items()
        if v.get("source") == "cudem"
    ]
    cudem_entries.sort(key=lambda kv: kv[1].get("priority", 99))

    new_manifest: Dict = {}
    priority = 0

    # Add split GEBCO tiles (lowest priority — background)
    for i, tile_path in enumerate(sorted(gebco_tile_paths)):
        name = f"gebco_split/{tile_path.stem}"
        new_manifest[name] = {
            "name":      name,
            "subfolder": "gebco_split",
            "filename":  tile_path.name,
            "path":      str(tile_path),
            "url":       "n/a",
            "source":    "gebco",
            "available": tile_path.exists(),
            "priority":  priority,
        }
        priority += 1

    # Add CUDEM tiles (higher priority — coastal override)
    for k, v in cudem_entries:
        v = dict(v)
        v["priority"] = priority
        new_manifest[k] = v
        priority += 1

    out_manifest_path.write_text(json.dumps(new_manifest, indent=2))

    n_gebco = sum(1 for v in new_manifest.values()
                  if v.get("source") == "gebco")
    n_cudem = sum(1 for v in new_manifest.values()
                  if v.get("source") == "cudem")
    print(f"\nManifest written to: {out_manifest_path}")
    print(f"  GEBCO sub-tiles : {n_gebco}")
    print(f"  CUDEM tiles     : {n_cudem}")
    print(f"  Total           : {n_gebco + n_cudem}")

    return new_manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split GEBCO tile into NxN grid for MPI benchmarking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(__file__).parent / "dem_manifest.json",
        help="Full DEM manifest (dem_manifest.json with 387 CUDEM tiles)",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/stofs_dems"),
        help="Root directory for DEM storage. Sub-tiles written to "
             "<out-dir>/gebco_split/",
    )
    parser.add_argument(
        "--n", type=int, default=8,
        help="Grid size: split into NxN tiles (default: 8, giving 64 tiles)",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.1,
        help="Overlap between adjacent tiles in degrees (default: 0.1)",
    )
    parser.add_argument(
        "--out-manifest", type=Path, default=None,
        help="Output manifest path. Default: "
             "<manifest_dir>/dem_manifest_full_split.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print tile extents without writing files or manifest.",
    )
    args = parser.parse_args()

    # Resolve output manifest path
    if args.out_manifest is None:
        args.out_manifest = args.manifest.parent / "dem_manifest_full_split.json"

    # Find the GEBCO tile in the manifest
    manifest = json.loads(args.manifest.read_text())
    gebco_entries = [
        v for v in manifest.values()
        if v.get("source") == "gebco" and v.get("available")
    ]
    if not gebco_entries:
        print("ERROR: No available GEBCO tile found in manifest.")
        raise SystemExit(1)
    if len(gebco_entries) > 1:
        print(f"WARNING: {len(gebco_entries)} GEBCO entries found. "
              f"Using the first one.")
    gebco_meta = gebco_entries[0]
    gebco_path = Path(gebco_meta["path"])

    if not gebco_path.exists():
        print(f"ERROR: GEBCO file not found: {gebco_path}")
        raise SystemExit(1)

    print("=" * 60)
    print(f"  GEBCO Split Tool")
    print(f"  Grid       : {args.n}×{args.n} = {args.n*args.n} tiles")
    print(f"  Overlap    : {args.overlap}°")
    print(f"  Output dir : {args.out_dir}/gebco_split/")
    print(f"  Dry run    : {args.dry_run}")
    print("=" * 60)
    print()

    # Split
    split_dir = args.out_dir / "gebco_split"
    tile_paths = split_gebco(
        src_path=gebco_path,
        out_dir=split_dir,
        n=args.n,
        overlap=args.overlap,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"\nDry run complete. {args.n*args.n} tiles would be written to:")
        print(f"  {split_dir}")
        return

    if not tile_paths:
        print("No tiles written.")
        return

    # Build new manifest
    print()
    print("=" * 60)
    print("  Building split manifest")
    print("=" * 60)
    build_split_manifest(
        original_manifest_path=args.manifest,
        gebco_tile_paths=tile_paths,
        out_manifest_path=args.out_manifest,
    )

    print()
    print("=" * 60)
    print("  Done")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Verify tiles look correct:")
    print(f"       python split_gebco.py --manifest {args.manifest} "
          f"--out-dir {args.out_dir} --dry-run")
    print(f"  2. Use the split manifest for the 300+ DEM benchmark:")
    print(f"       --manifest {args.out_manifest}")
    print(f"  3. Expected: {args.n*args.n} GEBCO sub-tiles each ~2 min "
          f"vs ~131 min for the full tile")


if __name__ == "__main__":
    main()
