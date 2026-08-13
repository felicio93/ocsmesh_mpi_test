"""Post-run profiling analysis and benchmark report generator.

Reads:
  - ``benchmark_results.json`` (written by run_benchmark.py)
  - ``profile_serial.prof``, ``profile_parallel.prof``, ``profile_mpi.prof``
    (cProfile binary files written by run_benchmark.py)

Produces:
  - A human-readable text report (``benchmark_report.txt``)
  - A per-mode profile summary (top hotspots) embedded in the report
  - An optional CSV speedup table for easy copy-paste into papers/slides

Usage
-----
    # After slurm_single_node.sh completes:
    python analyze_profile.py \\
        --results-dir /work/noaa/<user>/results/single_node_<jobid> \\
        --out         benchmark_report.txt

    # After slurm_multi_node.sh:
    python analyze_profile.py \\
        --results-dir /work/noaa/<user>/results/multi_node_<jobid> \\
        --out         benchmark_report_multinode.txt

    # Merge single-node and multi-node results for a combined speedup table:
    python analyze_profile.py \\
        --results-dir /work/noaa/<user>/results/single_node_<jobid> \\
                      /work/noaa/<user>/results/multi_node_<jobid> \\
        --out         combined_report.txt
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pstats
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Dict]:
    if path.exists():
        return json.loads(path.read_text())
    return None


def _load_profile_stats(prof_path: Path, n_top: int = 25) -> str:
    """Load a cProfile .prof file and return top-N hotspots as a string."""
    if not prof_path.exists():
        return f"  [profile not found: {prof_path}]"
    sio = io.StringIO()
    ps = pstats.Stats(str(prof_path), stream=sio)
    ps.strip_dirs()
    ps.sort_stats("cumulative")
    ps.print_stats(n_top)
    return sio.getvalue()


def _status_icon(status: str) -> str:
    return {"success": "OK", "failed": "FAIL", "pending": "---"}.get(status, status)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    results_dirs: List[Path],
    output_path: Path,
    n_profile_top: int = 25,
) -> None:
    """Build a combined benchmark report from one or more results directories."""

    lines: List[str] = []
    sep = "=" * 70

    def h(title: str) -> None:
        lines.append("")
        lines.append(sep)
        lines.append(f"  {title}")
        lines.append(sep)

    # ── Header ──────────────────────────────────────────────────────────────
    lines.append(sep)
    lines.append("  OCSMesh MPI Benchmark Report")
    lines.append(f"  Generated : {datetime.now().isoformat(timespec='seconds')}")
    lines.append(sep)

    all_run_results = []

    for results_dir in results_dirs:
        h(f"Results directory: {results_dir}")

        # Look for benchmark_results.json in this dir and sub-dirs
        json_candidates = [
            results_dir / "benchmark_results.json",
            results_dir / "serial_parallel" / "benchmark_results.json",
            results_dir / "mpi" / "benchmark_results.json",
        ]
        summaries = [_load_json(p) for p in json_candidates if p and _load_json(p)]

        if not summaries:
            lines.append("  [No benchmark_results.json found in this directory]")
            continue

        for summary in summaries:
            hostname = summary.get("hostname", "unknown")
            mpi_size = summary.get("mpi_size", 1)
            nprocs = summary.get("nprocs_parallel", "?")
            n_dems = summary.get("n_dems", "?")
            hmin = summary.get("hmin", "?")
            hmax = summary.get("hmax", "?")

            lines.append(f"  Hostname      : {hostname}")
            lines.append(f"  MPI size      : {mpi_size} ranks")
            lines.append(f"  nprocs (para) : {nprocs}")
            lines.append(f"  DEMs loaded   : {n_dems}")
            lines.append(f"  hmin / hmax   : {hmin} m / {hmax} m")
            lines.append("")

            # ── Timing table ────────────────────────────────────────────────
            lines.append(
                f"  {'Mode':<14} {'Status':<8} {'Wall (s)':>10}  "
                f"{'Speedup':>9}  {'Nodes':>10}  {'Tria':>10}"
            )
            lines.append(
                f"  {'-'*14} {'-'*8} {'-'*10}  {'-'*9}  {'-'*10}  {'-'*10}"
            )
            for r in summary.get("results", []):
                sp = (
                    f"{r.get('speedup_vs_serial', 1.0):.2f}x"
                    if "speedup_vs_serial" in r else " — "
                )
                nodes = f"{r.get('n_nodes', 0):,}"
                tria = f"{r.get('n_triangles', 0):,}"
                lines.append(
                    f"  {r['mode']:<14} "
                    f"{_status_icon(r['status']):<8} "
                    f"{r.get('wall_time_s', 0):>10.2f}  "
                    f"{sp:>9}  {nodes:>10}  {tria:>10}"
                )
                if r["status"] == "failed":
                    lines.append(f"    ERROR: {r.get('error', 'unknown')}")
            lines.append("")

            # ── Mesh quality stats ───────────────────────────────────────────
            for r in summary.get("results", []):
                if r["status"] != "success":
                    continue
                lines.append(f"  [{r['mode']}] Mesh size function stats:")
                lines.append(f"    hfun min  = {r.get('hfun_min', 'N/A'):.1f} m")
                lines.append(f"    hfun max  = {r.get('hfun_max', 'N/A'):.1f} m")
                lines.append(f"    hfun mean = {r.get('hfun_mean', 'N/A'):.1f} m")
                lines.append(f"    hfun std  = {r.get('hfun_std', 'N/A'):.1f} m")
                lines.append("")

            all_run_results.extend(summary.get("results", []))

            # ── cProfile hotspots ────────────────────────────────────────────
            h("cProfile Hotspots (cumulative time)")
            for mode in ("serial", "parallel", "mpi"):
                # Search this dir and sub-dirs
                candidates = [
                    results_dir / f"profile_{mode}.prof",
                    results_dir / "serial_parallel" / f"profile_{mode}.prof",
                    results_dir / "mpi" / f"profile_{mode}.prof",
                ]
                for prof_path in candidates:
                    if prof_path.exists():
                        lines.append(f"\n--- {mode.upper()} ({prof_path}) ---")
                        lines.append(_load_profile_stats(prof_path, n_top=n_profile_top))
                        break
                else:
                    lines.append(f"\n--- {mode.upper()} ---")
                    lines.append("  [no .prof file found]\n")

    # ── Numerical equivalence check ─────────────────────────────────────────
    h("Numerical Equivalence Check (serial baseline)")
    serial_result = next(
        (r for r in all_run_results
         if r["mode"] == "serial" and r["status"] == "success"),
        None,
    )
    if serial_result:
        s_nodes = serial_result.get("n_nodes", 0)
        s_min = serial_result.get("hfun_min", float("nan"))
        s_max = serial_result.get("hfun_max", float("nan"))
        s_mean = serial_result.get("hfun_mean", float("nan"))
        lines.append(f"  Baseline (serial): {s_nodes:,} nodes")
        lines.append("")
        lines.append(
            f"  {'Mode':<14} {'Nodes':>10}  {'|Δnodes|':>10}  "
            f"{'|Δmin|':>10}  {'|Δmax|':>10}  {'|Δmean|':>10}"
        )
        lines.append(
            f"  {'-'*14} {'-'*10}  {'-'*10}  "
            f"{'-'*10}  {'-'*10}  {'-'*10}"
        )
        for r in all_run_results:
            if r["mode"] == "serial" or r["status"] != "success":
                continue
            r_nodes = r.get("n_nodes", 0)
            d_nodes = abs(r_nodes - s_nodes)
            d_min = abs(r.get("hfun_min", float("nan")) - s_min)
            d_max = abs(r.get("hfun_max", float("nan")) - s_max)
            d_mean = abs(r.get("hfun_mean", float("nan")) - s_mean)
            # flag if node count differs by more than 1%
            flag = " *** NODE MISMATCH ***" if d_nodes > 0.01 * s_nodes else ""
            lines.append(
                f"  {r['mode']:<14} {r_nodes:>10,}  "
                f"{d_nodes:>10,}  "
                f"{d_min:>10.2f}  {d_max:>10.2f}  {d_mean:>10.2f}{flag}"
            )
    else:
        lines.append("  Serial result not found — equivalence check skipped.")

    # ── Speedup summary CSV ──────────────────────────────────────────────────
    h("Speedup Summary (CSV — copy-paste to spreadsheet)")
    serial_time = next(
        (r["wall_time_s"] for r in all_run_results
         if r["mode"] == "serial" and r["status"] == "success"),
        None,
    )
    csv_io = io.StringIO()
    writer = csv.writer(csv_io)
    writer.writerow(["mode", "wall_time_s", "speedup_vs_serial", "n_nodes"])
    for r in all_run_results:
        if r["status"] != "success":
            continue
        sp = (serial_time / r["wall_time_s"]) if serial_time else None
        writer.writerow([
            r["mode"],
            f"{r.get('wall_time_s', 0):.3f}",
            f"{sp:.3f}" if sp else "",
            r.get("n_nodes", 0),
        ])
    lines.append(csv_io.getvalue())

    # ── Footer ──────────────────────────────────────────────────────────────
    lines.append(sep)
    lines.append("  End of Report")
    lines.append(sep)

    report_text = "\n".join(lines)
    output_path.write_text(report_text)
    print(f"Report written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate benchmark report from OCSMesh profiling results."
    )
    parser.add_argument(
        "--results-dir",
        nargs="+",
        type=Path,
        required=True,
        help="One or more results directories (output of run_benchmark.py).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_report.txt"),
        help="Output report file (default: benchmark_report.txt)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of top hotspots to show per profile (default: 25)",
    )
    args = parser.parse_args()

    build_report(args.results_dir, args.out, n_profile_top=args.top)


if __name__ == "__main__":
    main()
