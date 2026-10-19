"""NJ_test/check_tiles_clip.py

Mimic what HfunCollector.__init__ does to every manifest tile (clip to the
AOI) WITHOUT building any hfun, and report which tiles survive and what is
in them. Used to find tiles that pass Raster.clip() but contain no real data.

    python check_tiles_clip.py                       # all available manifest tiles
    python check_tiles_clip.py --manifest NJ_dem_manifest.json --out tiles_clip.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
from pyproj import CRS, Transformer
from shapely import ops
from shapely.geometry import box

from ocsmesh import Raster

from nj_mesh_config_gaea import MANIFEST_PATH, DOMAIN_SHP

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--out", type=Path, default=Path("results_gaea/tiles_clip.csv"))
    ap.add_argument("--decimate", type=int, default=20,
                    help="read every Nth pixel when computing stats")
    args = ap.parse_args()

    gdf = gpd.read_file(DOMAIN_SHP).to_crs("EPSG:4326")
    aoi = gdf.union_all()
    manifest = json.load(open(args.manifest))
    entries = sorted((e for e in manifest.values() if e.get("available")),
                     key=lambda e: e["priority"])

    rows = []
    hdr = (f"{'name':<36}{'src':<11}{'in_poly':>8}{'clip':>8}{'valid%':>8}"
           f"{'nodata':>12}{'min':>12}{'max':>12}")
    print(hdr); print("-" * len(hdr))
    for e in entries:
        r = Raster(e["path"])
        b = r.get_bbox().bounds
        in_poly = box(*b).intersects(aoi)
        shape = aoi
        if not CRS.from_user_input("EPSG:4326").equals(r.crs):
            tr = Transformer.from_crs("EPSG:4326", r.crs, always_xy=True)
            shape = ops.transform(tr.transform, shape)
        row = dict(name=e["name"], source=e["source"], in_polygon=in_poly,
                   clip="", valid_pct="", nodata="", vmin="", vmax="",
                   west=b[0], south=b[1], east=b[2], north=b[3])
        try:
            r.clip(shape)
            row["clip"] = "ok"
            src = r.src
            h, w = src.height, src.width
            d = max(1, args.decimate)
            data = src.read(1, out_shape=(max(1, h // d), max(1, w // d)),
                            masked=True)
            row["nodata"] = src.nodata
            row["valid_pct"] = round(100.0 * (1 - np.ma.getmaskarray(data).mean()), 1)
            if data.count():
                row["vmin"] = float(data.min()); row["vmax"] = float(data.max())
        except ValueError:
            row["clip"] = "DROPPED"
        except Exception as exc:                     # keep going, show the error
            row["clip"] = f"ERR:{type(exc).__name__}"
        rows.append(row)
        print(f"{row['name']:<36}{row['source']:<11}{str(in_poly):>8}{row['clip']:>8}"
              f"{str(row['valid_pct']):>8}{str(row['nodata']):>12}"
              f"{str(row['vmin'])[:10]:>12}{str(row['vmax'])[:10]:>12}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    kept_out = [r for r in rows if not r["in_polygon"] and r["clip"] == "ok"]
    print(f"\n{len(rows)} tiles | in AOI polygon: {sum(r['in_polygon'] for r in rows)}"
          f" | survive clip: {sum(r['clip'] == 'ok' for r in rows)}"
          f" | OUTSIDE polygon but survive clip: {len(kept_out)}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
