"""NJ_test/diagnose_channels.py

Debug the channel refinement on ONE manifest tile (minutes, not hours).

For the chosen tile it
  1. runs the same Raster.get_channels(level, width, tolerance) call that
     Hfun.add_channel uses and reports channel area vs. water area,
  2. runs a single-tile Hfun with add_channel and reports size statistics,
  3. writes a GeoPackage (tile bounds, water polygon, channels) for QGIS.

Usage
-----
    python diagnose_channels.py --source ninth --index 0
    python diagnose_channels.py --source gebco --index 0 --tolerance 50
    python diagnose_channels.py --source third --index 3 --skip-hfun

Multi-tile mode (reproduces the full-pipeline assembly on a subset; skips
the per-tile polygon step). Tiles are the FIRST n of each source, by
manifest priority; --domain clips them to the benchmark AOI like the real
run does:

    python diagnose_channels.py --tiles gebco:2,third:2,ninth:2 --domain --tolerance 50
    python diagnose_channels.py --tiles gebco:2,third:2,ninth:2 --domain   # tolerance=None
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.geometry import box

from ocsmesh import Raster, Hfun

from nj_mesh_config_gaea import MANIFEST_PATH, DOMAIN_SHP, HMIN, HMAX, CHANNELS

_SOURCES = {"gebco": "gebco", "third": "ncei_third", "ninth": "ncei_ninth"}


def pick_tile(manifest_path: Path, source: str, index: int) -> str:
    manifest = json.load(open(manifest_path))
    entries = sorted(
        (e for e in manifest.values()
         if e.get("available") and e["source"] == _SOURCES[source]),
        key=lambda e: e["priority"],
    )
    if not entries:
        raise SystemExit(f"No available '{source}' tiles in manifest")
    print(f"{len(entries)} '{source}' tiles; using index {index}")
    return entries[index]["path"]


def pick_many(manifest_path: Path, spec: str):
    """'gebco:2,third:2' -> first n tile paths of each source."""
    paths = []
    for part in spec.split(","):
        src, n = part.split(":")
        manifest = json.load(open(manifest_path))
        entries = sorted(
            (e for e in manifest.values()
             if e.get("available") and e["source"] == _SOURCES[src]),
            key=lambda e: e["priority"])
        paths += [e["path"] for e in entries[: int(n)]]
    return paths


def load_domain():
    gdf = gpd.read_file(DOMAIN_SHP)
    if gdf.crs and not gdf.crs.equals("EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")
    return gdf.union_all(), gdf.crs


def run_multi(paths, args, width, target, rate) -> None:
    print(f"{len(paths)} tiles:", *[Path(p).name for p in paths], sep="\n  ")
    kw = {}
    if args.domain:
        shape, crs = load_domain()
        kw = dict(base_shape=shape, base_shape_crs=crs)
    for with_channel in (False, True):
        hfun = Hfun(paths, hmin=HMIN, hmax=HMAX, nprocs=4,
                    method="exact", **kw)
        hfun.execution_mode = "serial"
        if with_channel:
            hfun.add_channel(level=0.0, width=width, target_size=target,
                             expansion_rate=rate, tolerance=args.tolerance)
        t0 = time.perf_counter()
        md = hfun.meshdata()
        v = np.asarray(md.values).ravel()
        q = np.nanpercentile(v, [0, 5, 25, 50, 75, 95, 100])
        print(f"\nhfun {'WITH' if with_channel else 'WITHOUT'} channel "
              f"({time.perf_counter() - t0:.1f}s, {md.num_nodes} nodes)")
        print("  percentiles 0/5/25/50/75/95/100:", np.round(q, 1).tolist())
        print(f"  fraction at HMIN({HMIN:g}): {(v <= HMIN).mean():.2%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=list(_SOURCES), default="ninth")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--tolerance", type=float, default=None,
                    help="polygon simplify tolerance (m); None = what the "
                         "benchmark currently uses")
    ap.add_argument("--out", type=Path, default=Path("results_gaea/diagnose"))
    ap.add_argument("--skip-hfun", action="store_true")
    ap.add_argument("--tiles", default=None,
                    help="multi-tile mode, e.g. gebco:2,third:2,ninth:2")
    ap.add_argument("--domain", action="store_true",
                    help="multi-tile mode: clip to the benchmark AOI")
    args = ap.parse_args()

    width, target, rate, _ = CHANNELS[0]
    if args.tiles:
        run_multi(pick_many(args.manifest, args.tiles), args,
                  width, target, rate)
        return
    path = pick_tile(args.manifest, args.source, args.index)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.source}_{args.index}_tol{args.tolerance}"
    print(f"tile: {path}\nwidth={width} target={target} rate={rate} "
          f"tolerance={args.tolerance}")

    rast = Raster(path)
    bounds = rast.get_bbox().bounds
    print("tile bounds (lon/lat):", [round(b, 4) for b in bounds])

    # 1. channel polygons ------------------------------------------------
    t0 = time.perf_counter()
    water = rast.get_multipolygon(zmax=0.0)
    ch = rast.get_channels(0.0, width, args.tolerance)
    dt = time.perf_counter() - t0
    gdf_w = gpd.GeoSeries([water], crs=rast.crs)
    utm = gdf_w.estimate_utm_crs()
    a_water = gdf_w.to_crs(utm).area.sum() / 1e6
    if ch is None:
        a_ch = 0.0
        print("get_channels returned None (no channels)")
    else:
        gdf_c = gpd.GeoSeries([ch], crs=rast.crs)
        a_ch = gdf_c.to_crs(utm).area.sum() / 1e6
    a_tile = gpd.GeoSeries([box(*bounds)], crs=rast.crs).to_crs(utm).area.sum() / 1e6
    print(f"get_channels: {dt:.1f}s")
    print(f"area km2  tile={a_tile:.1f}  water(z<=0)={a_water:.1f}  "
          f"channels={a_ch:.1f}  channels/water={a_ch / max(a_water, 1e-9):.2%}  "
          f"channels/tile={a_ch / a_tile:.2%}")

    gpkg = args.out / f"channels_{stem}.gpkg"
    gpd.GeoDataFrame({"geometry": [box(*bounds)]}, crs=rast.crs).to_file(
        gpkg, layer="tile", driver="GPKG")
    gpd.GeoDataFrame({"geometry": [water]}, crs=rast.crs).to_file(
        gpkg, layer="water", driver="GPKG")
    if ch is not None:
        gpd.GeoDataFrame({"geometry": [ch]}, crs=rast.crs).to_file(
            gpkg, layer="channels", driver="GPKG")
    print("wrote", gpkg)

    if args.skip_hfun:
        return

    # 2. single-tile hfun ------------------------------------------------
    for with_channel in (False, True):
        hfun = Hfun([path], hmin=HMIN, hmax=HMAX, nprocs=4, method="exact")
        hfun.execution_mode = "serial"
        if with_channel:
            hfun.add_channel(level=0.0, width=width, target_size=target,
                             expansion_rate=rate, tolerance=args.tolerance)
        t0 = time.perf_counter()
        md = hfun.meshdata()
        v = np.asarray(md.values).ravel()
        q = np.nanpercentile(v, [0, 5, 25, 50, 75, 95, 100])
        print(f"\nhfun {'WITH' if with_channel else 'WITHOUT'} channel "
              f"({time.perf_counter() - t0:.1f}s, {md.num_nodes} nodes)")
        print("  percentiles 0/5/25/50/75/95/100:", np.round(q, 1).tolist())
        print(f"  fraction at HMIN({HMIN:g}): {(v <= HMIN).mean():.2%}")


if __name__ == "__main__":
    main()
