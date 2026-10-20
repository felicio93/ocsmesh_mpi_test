"""NJ_test/time_channels_per_tile.py

Time Raster.get_channels() on each of the 18 AOI tiles, one at a time, to see
whether channel detection is dominated by a few tiles (caps per-raster
parallelism) or spread evenly.

    python time_channels_per_tile.py --tolerance 50
    python time_channels_per_tile.py --tolerance none --indices 9,10   # slow
"""

from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path

import rasterio

from ocsmesh import Raster

warnings.filterwarnings("ignore", message="shapes are outside bounds")

from diagnose_channels import pick_many, load_domain
from nj_mesh_config_gaea import MANIFEST_PATH, CHANNELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", default="50",
                    help="metres, or 'none' for no simplification")
    ap.add_argument("--indices", default=None,
                    help="comma list of tile indices to run (default: all)")
    ap.add_argument("--out", type=Path, default=Path("results_gaea/diagnose"))
    args = ap.parse_args()

    tol = None if args.tolerance.lower() == "none" else float(args.tolerance)
    width = CHANNELS[0][0]
    shape, _ = load_domain()
    paths = pick_many(MANIFEST_PATH, "gebco:1,third:8,ninth:9", shape)
    idx = (range(len(paths)) if args.indices is None
           else [int(i) for i in args.indices.split(",")])

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"channels_per_tile_tol{args.tolerance}.csv"
    rows = []
    total = 0.0
    for i in idx:
        p = paths[i]
        with rasterio.open(p) as src:
            npx = src.width * src.height
        t0 = time.perf_counter()
        ch = Raster(p).get_channels(0.0, width, tol)
        dt = time.perf_counter() - t0
        total += dt
        nverts = 0 if ch is None else sum(
            len(g.exterior.coords) for g in getattr(ch, "geoms", [ch]))
        print(f"[{i:2d}] {Path(p).name:40s} {npx/1e6:8.1f} Mpx  "
              f"{dt:9.1f}s  verts={nverts}", flush=True)
        rows.append((i, Path(p).name, npx, round(dt, 1), nverts))
    print(f"=== tol={tol} sum={total:.1f}s max={max(r[3] for r in rows):.1f}s")
    with open(csv_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["idx", "tile", "pixels", "seconds", "verts"])
        w.writerows(rows)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
