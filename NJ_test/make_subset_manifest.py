"""NJ_test/make_subset_manifest.py

Write a small manifest containing the first n AOI-intersecting tiles of each
source, for fast debugging of build_nj_mesh_gaea.py.

    python make_subset_manifest.py gebco:1,third:2,ninth:2 NJ_dem_manifest_small.json

Suffix a source with _out to take tiles that do NOT intersect the AOI polygon
(bounds only overlap its bounding box), e.g. gebco_out:1,third_out:12.
The count may be a slice a-b (by priority order), e.g. ninth_out:0-6, to bisect.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box

from nj_mesh_config_gaea import MANIFEST_PATH, DOMAIN_SHP

SRC = {"gebco": "gebco", "third": "ncei_third", "ninth": "ncei_ninth"}


def main(spec: str, out: str) -> None:
    gdf = gpd.read_file(DOMAIN_SHP).to_crs("EPSG:4326")
    aoi = gdf.union_all()
    manifest = json.load(open(MANIFEST_PATH))
    keep = {}
    for part in spec.split(","):
        src, n = part.split(":")
        outside = src.endswith("_out")
        if outside:
            src = src[: -len("_out")]
        entries = sorted((e for e in manifest.values()
                          if e.get("available") and e["source"] == SRC[src]),
                         key=lambda e: e["priority"])
        sel = []
        for e in entries:
            with rasterio.open(e["path"]) as r:
                b = r.bounds
            if box(b.left, b.bottom, b.right, b.top).intersects(aoi) != outside:
                sel.append(e)
        if "-" in n:
            a, b = (int(x) for x in n.split("-"))
        else:
            a, b = 0, int(n)
        for e in sel[a:b]:
            keep[e["name"]] = e
        print(f"{src}{'_out' if outside else ''}: {len(sel)} tiles, "
              f"kept {len(sel[a:b])} ({a}-{b})")
        for e in sel[a:b]:
            print("   ", e["name"])
    Path(out).write_text(json.dumps(keep, indent=2))
    print(f"wrote {out} ({len(keep)} tiles)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
