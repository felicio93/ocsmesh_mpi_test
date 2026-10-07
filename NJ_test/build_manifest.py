"""NJ_test/build_manifest.py

Assemble the NJ DEM manifest from three sources:

    GEBCO split tiles   (priority: lowest  — ~450m ocean background)
    NCEI third arc-sec  (priority: middle  — ~10m coastal)
    NCEI ninth arc-sec  (priority: highest — ~3m finest resolution)

Resolution note:
    ncei13 (1/3") < ncei19 (1/9") in arc-second fraction,
    BUT 1/9 arc-second is a FINER grid than 1/3 arc-second.
    So ninth arc-second (CUDEM) has the highest priority.

Usage
-----
    python build_manifest.py            # full download + manifest
    python build_manifest.py --dry-run  # preview without downloading
    python build_manifest.py --no-third # skip third arc-second
    python build_manifest.py --no-ninth # skip ninth arc-second
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from dem_config import (
    DEFAULT_DEM_DIR,
    DEFAULT_MANIFEST,
    SOURCE_GEBCO,
    SOURCE_THIRD,
    SOURCE_NINTH,
    DOMAIN_LAT_MIN, DOMAIN_LAT_MAX,
    DOMAIN_LON_MIN, DOMAIN_LON_MAX,
    GEBCO_SPLIT_DIR,
)
from collect_gebco  import collect_gebco_entries
from download_third import download_third
from download_ninth import download_ninth


# ---------------------------------------------------------------------------
# Priority assignment
# ---------------------------------------------------------------------------

def _assign_priorities(
    gebco_entries: List[Dict],
    third_entries: List[Dict],
    ninth_entries: List[Dict],
) -> Dict:
    """Build manifest with sequential priorities.

    Ordering passed to HfunCollector (lowest index = lowest priority):
        index 0..N_gebco-1                    : GEBCO  (lowest)
        index N_gebco..N_gebco+N_third-1      : NCEI third (~10m)
        index N_gebco+N_third..end            : NCEI ninth (~3m, highest)

    HfunCollector reverses the list internally so the last entry
    (ninth arc-second) has absolute highest priority and overrides
    all lower-resolution sources where tiles overlap.
    """
    manifest: Dict = {}
    counter = 0

    for e in gebco_entries:
        entry = dict(e)
        entry["priority"] = counter
        manifest[entry["name"]] = entry
        counter += 1

    for e in third_entries:
        entry = dict(e)
        entry["priority"] = counter
        manifest[entry["name"]] = entry
        counter += 1

    for e in ninth_entries:
        entry = dict(e)
        entry["priority"] = counter
        manifest[entry["name"]] = entry
        counter += 1

    return manifest


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(manifest: Dict, out_path: Path, dry_run: bool) -> None:
    total   = len(manifest)
    avail   = sum(1 for v in manifest.values() if v["available"])
    n_gebco = sum(1 for v in manifest.values() if v["source"] == SOURCE_GEBCO)
    n_third = sum(1 for v in manifest.values() if v["source"] == SOURCE_THIRD)
    n_ninth = sum(1 for v in manifest.values() if v["source"] == SOURCE_NINTH)

    tag = "[DRY RUN] " if dry_run else ""
    print()
    print("=" * 60)
    print(f"  {tag}NJ DEM Manifest Summary")
    print("=" * 60)
    print(f"  Domain      : {DOMAIN_LAT_MIN}–{DOMAIN_LAT_MAX} N, "
          f"{DOMAIN_LON_MIN}–{DOMAIN_LON_MAX} E")
    print(f"  GEBCO       : {n_gebco} split tile(s)  [lowest  ~450m]")
    print(f"  NCEI third  : {n_third} tile(s)         [middle  ~10m ]")
    print(f"  NCEI ninth  : {n_ninth} tile(s)         [highest ~3m  ]")
    print(f"  Total       : {total}  ({avail} available on disk)")
    if not dry_run:
        print(f"  Output      : {out_path}")
    print()
    print("  HfunCollector priority (low → high):")
    print("    GEBCO (~450m) → NCEI third (~10m) → NCEI ninth (~3m)")
    print("    Ninth arc-second overrides all lower-res sources.")
    print("=" * 60)

    failed = [k for k, v in manifest.items() if not v["available"]]
    if failed:
        print(f"\n  WARNING: {len(failed)} unavailable entries:")
        for k in failed[:10]:
            print(f"    {k}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_manifest(
    dem_dir: Path,
    out_path: Path,
    include_third: bool = True,
    include_ninth: bool = True,
    dry_run: bool = False,
) -> Dict:
    """Collect GEBCO + download NCEI third + ninth, write manifest."""

    print("=" * 60)
    print("  NJ DEM Manifest Builder")
    print("=" * 60)
    print(f"  GEBCO split dir : {GEBCO_SPLIT_DIR}")
    print(f"  DEM directory   : {dem_dir}")
    print(f"  Manifest path   : {out_path}")
    print(f"  Dry run         : {dry_run}")
    print(f"  Include third   : {include_third}  (~10m NCEI third arc-sec)")
    print(f"  Include ninth   : {include_ninth}  (~3m  NCEI ninth arc-sec)")

    # ── GEBCO ─────────────────────────────────────────────────────────────
    gebco_entries = collect_gebco_entries(verbose=True)
    if not gebco_entries:
        print("\nERROR: No GEBCO tiles found. Check GEBCO_SPLIT_DIR.")
        sys.exit(1)

    # ── NCEI third arc-second (~10m, middle priority) ─────────────────────
    third_entries: List[Dict] = []
    if include_third:
        third_entries = download_third(dem_dir, dry_run=dry_run)
    else:
        print("\n  NCEI third: skipped (--no-third)")

    # ── NCEI ninth arc-second (~3m, highest priority) ─────────────────────
    ninth_entries: List[Dict] = []
    if include_ninth:
        ninth_entries = download_ninth(dem_dir, dry_run=dry_run)
    else:
        print("\n  NCEI ninth: skipped (--no-ninth)")

    # ── Assemble and write ─────────────────────────────────────────────────
    manifest = _assign_priorities(gebco_entries, third_entries, ninth_entries)

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2))
        print(f"\n  Manifest written to: {out_path}")

    _print_summary(manifest, out_path, dry_run=dry_run)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect GEBCO split tiles + download NCEI third and ninth "
            "arc-second tiles for the NJ domain, then write manifest JSON.\n\n"
            "Priority (low to high): GEBCO → NCEI third (~10m) → NCEI ninth (~3m)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python build_manifest.py\n"
            "  python build_manifest.py --dry-run\n"
            "  python build_manifest.py --no-third\n"
            "  GEBCO_SPLIT_DIR=/other/path python build_manifest.py\n"
        ),
    )
    parser.add_argument(
        "--dem-dir", type=Path, default=Path(DEFAULT_DEM_DIR),
        help=f"Root directory for DEM downloads (default: {DEFAULT_DEM_DIR})",
    )
    parser.add_argument(
        "--out", type=Path, default=Path(DEFAULT_MANIFEST),
        help=f"Output manifest JSON (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument("--no-third", action="store_true",
                        help="Skip NCEI third arc-second download.")
    parser.add_argument("--no-ninth", action="store_true",
                        help="Skip NCEI ninth arc-second download.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be included without downloading.")
    args = parser.parse_args()

    build_manifest(
        dem_dir=args.dem_dir.resolve(),
        out_path=args.out.resolve(),
        include_third=not args.no_third,
        include_ninth=not args.no_ninth,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
