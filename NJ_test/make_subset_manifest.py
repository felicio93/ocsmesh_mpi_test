"""NJ_test/make_subset_manifest.py

Write a small manifest containing the first n AOI-intersecting tiles of each
source, for fast debugging of build_nj_mesh_gaea.py.

    python make_subset_manifest.py gebco:1,third:2,ninth:2 NJ_dem_manifest_small.json
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
        entries = sorted((e for e in manifest.values()
                          if e.get("available") and e["source"] == SRC[src]),
                         key=lambda e: e["priority"])
        sel = []
        for e in entries:
            with rasterio.open(e["path"]) as r:
                b = r.bounds
            if box(b.left, b.bottom, b.right, b.top).intersects(aoi):
                sel.append(e)
        for e in sel[: int(n)]:
            keep[e["name"]] = e
        print(f"{src}: {len(sel)} AOI tiles, kept {min(int(n), len(sel))}")
    Path(out).write_text(json.dumps(keep, indent=2))
    print(f"wrote {out} ({len(keep)} tiles)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
