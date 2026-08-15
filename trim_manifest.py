"""Trim a DEM manifest to the first N available CUDEM tiles (+ GEBCO).

Used to shrink the smoke test so serial_mp fits in the 8h wall clock.
Keeps full DEM resolution — just reduces the tile count.

The kept tiles are re-indexed with sequential priorities so the
index-modulo refinement assignment in build_geom_and_hfun.py still
spreads them across all 5 refinement classes.

Usage
-----
    python trim_manifest.py \
        --in  dem_manifest_smoke.json \
        --out dem_manifest_smoke15.json \
        --n-cudem 14
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Trim DEM manifest to N CUDEM tiles + GEBCO.")
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", dest="out", type=Path, required=True)
    p.add_argument("--n-cudem", type=int, default=14,
                   help="Number of CUDEM tiles to keep (default: 14).")
    args = p.parse_args()

    manifest = json.loads(args.inp.read_text())

    # Split GEBCO vs CUDEM, keep only available + on-disk entries.
    gebco = {k: v for k, v in manifest.items()
             if v.get("source") == "gebco" and v.get("available")}
    cudem = [(k, v) for k, v in manifest.items()
             if v.get("source") == "cudem" and v.get("available")]

    # Sort CUDEM by existing priority for determinism, keep first N.
    cudem.sort(key=lambda kv: kv[1].get("priority", 99))
    kept_cudem = cudem[:args.n_cudem]

    # Rebuild manifest: GEBCO first (priority 0), then CUDEM (priority 1..N).
    out_manifest = {}
    prio = 0
    for k, v in gebco.items():
        v = dict(v)
        v["priority"] = prio
        out_manifest[k] = v
        prio += 1
    for k, v in kept_cudem:
        v = dict(v)
        v["priority"] = prio
        out_manifest[k] = v
        prio += 1

    args.out.write_text(json.dumps(out_manifest, indent=2))

    n_gebco = len(gebco)
    n_cudem = len(kept_cudem)
    print(f"Wrote {args.out}")
    print(f"  GEBCO tiles : {n_gebco}")
    print(f"  CUDEM tiles : {n_cudem}  (from {len(cudem)} available)")
    print(f"  Total       : {n_gebco + n_cudem}")


if __name__ == "__main__":
    main()
