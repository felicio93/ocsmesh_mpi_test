"""Post-run profiling analysis and benchmark report generator.

Reads:
  - ``benchmark_results.json`` (written by run_benchmark.py)
  - ``profile_serial_true.prof``, ``profile_serial_mp.prof``,
    ``profile_parallel.prof``, ``profile_mpi.prof``,
    ``profile_mpi_no_pool.prof``, ``profile_mpi_hybrid.prof``
    (cProfile binary files written by run_benchmark.py)

Produces:
  - A human-readable text report (``benchmark_report.txt``)
  - A per-mode profile summary (top hotspots) embedded in the report
  - CPU utilization table (wall time, CPU seconds, busy cores, util%)
  - Pixel-exact correctness check results
  - Speedup table comparing all modes vs serial_mp baseline
  - An optional CSV speedup table for copy-paste into papers/slides

Usage
-----
    # After a single-node run:
    python analyze_profile.py \\
        --results-dir /work/noaa/<user>/results/single_node_<jobid> \\
        --out         benchmark_report.txt

    # After smoke matrix (multiple configs):
    python analyze_profile.py \\
        --results-dir /work/noaa/<user>/results/smoke_matrix_<jobid>/config_F/mpi \\
                      /work/noaa/<user>/results/smoke_matrix_<jobid>/config_F/serial_mp \\
        --out         report_config_F.txt

    # Merge single-node and multi-node results:
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
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
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
    return {"success": "OK", "failed": "FAIL", "pending": "---"}.get(
        status, status
    )


# ---------------------------------------------------------------------------
# Pipeline-stage bucketing
# ---------------------------------------------------------------------------
_STAGE_MARKERS = {
    "geom_build":          ["build_geom", "GeomCollector", "combine_geom"],
    "hfun_construct":      ["build_hfun", "HfunCollector.__init__", "clip"],
    "apply_flow_limiter":  ["_apply_flow_limiter", "_flow_limiter_task_worker",
                            "add_subtidal_flow_limiter"],
    "apply_const_val":     ["_apply_const_val", "_const_val_task_worker",
                            "add_constant_value"],
    "apply_constraints":   ["_apply_constraints", "_constraints_task_worker",
                            "add_topo_bound", "add_topo_func",
                            "add_courant", "TopoFuncConstraint",
                            "TopoConstConstraint"],
    "apply_contours":      ["_apply_contours", "_contours_task_worker",
                            "add_contour"],
    "apply_channels":      ["_apply_channels", "add_channel"],
    "apply_user_shapes":   ["_apply_patch", "_apply_linefeatures",
                            "_user_shapes_task_worker",
                            "add_patch", "add_feature",
                            "_resolve_and_build_shape_tasks"],
    "calc_write_meshdata": ["_calculate_and_write_hfun_to_disk",
                            "meshdata", "_meshdata_task_worker",
                            "triangulate", "msh_t", "interpolate"],
    "composite":           ["_get_hfun_composite", "vstack",
                            "project_to_utm"],
    "mesh_write_2dm":      ["sms2dm", "meshdata_to_2dm", "write"],
    "raster_io":           ["rasterio", "Raster", "resampl",
                            "warp", "reproject"],
    "kdtree":              ["cKDTree", "KDTree", "query"],
}

# All modes including Anas's new ones
_ALL_MODES = [
    "serial_true",
    "serial_mp",
    "parallel",
    "mpi",
    "mpi_no_pool",
    "mpi_hybrid",
]


def _bucket_profile_by_stage(prof_path: Path) -> Optional[Dict[str, float]]:
    """Bucket a cProfile .prof into pipeline stages by self-time."""
    if not prof_path.exists():
        return None

    ps = pstats.Stats(str(prof_path))
    stage_self: Dict[str, float] = {s: 0.0 for s in _STAGE_MARKERS}

    for (fname, _lineno, funcname), (_cc, _nc, tt, ct, _callers) \
            in ps.stats.items():
        key = f"{fname}:{funcname}"
        for stage, markers in _STAGE_MARKERS.items():
            if any(m in key for m in markers):
                stage_self[stage] += tt
                break

    return stage_self


def _find_profile(results_dir: Path, mode: str) -> Optional[Path]:
    """Search for a profile file in a results directory and sub-dirs."""
    candidates = [
        results_dir / f"profile_{mode}.prof",
        results_dir / "serial_parallel" / f"profile_{mode}.prof",
        results_dir / "mpi"             / f"profile_{mode}.prof",
        results_dir / mode              / f"profile_{mode}.prof",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_json(results_dir: Path) -> Optional[Dict]:
    """Search for benchmark_results.json in a results directory."""
    candidates = [
        results_dir / "benchmark_results.json",
        results_dir / "serial_parallel" / "benchmark_results.json",
        results_dir / "mpi"             / "benchmark_results.json",
    ]
    for p in candidates:
        d = _load_json(p)
        if d is not None:
            return d
    return None


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    results_dirs: List[Path],
    output_path: Path,
    n_profile_top: int = 25,
) -> None:
    """Build a combined benchmark report from one or more results dirs."""

    lines: List[str] = []
    sep = "=" * 72

    def h(title: str) -> None:
        lines.append("")
        lines.append(sep)
        lines.append(f"  {title}")
        lines.append(sep)

    # ── Header ──────────────────────────────────────────────────────────
    lines.append(sep)
    lines.append("  OCSMesh MPI Benchmark Report")
    lines.append(f"  Generated : {datetime.now().isoformat(timespec='seconds')}")
    lines.append(sep)

    all_run_results: List[Dict] = []

    for results_dir in results_dirs:
        h(f"Results directory: {results_dir}")

        summary = _find_json(results_dir)
        if summary is None:
            lines.append(
                "  [No benchmark_results.json found in this directory]"
            )
            continue

        hostname  = summary.get("hostname", "unknown")
        mpi_size  = summary.get("mpi_size", 1)
        nprocs    = summary.get("nprocs_requested", "?")
        n_dems    = summary.get("n_dems", "?")
        hmin      = summary.get("hmin", "?")
        hmax      = summary.get("hmax", "?")

        lines.append(f"  Hostname      : {hostname}")
        lines.append(f"  MPI size      : {mpi_size} ranks")
        lines.append(f"  nprocs (para) : {nprocs}")
        lines.append(f"  DEMs loaded   : {n_dems}")
        lines.append(f"  hmin / hmax   : {hmin} m / {hmax} m")
        lines.append("")

        # ── Timing + utilization table ───────────────────────────────
        lines.append(
            f"  {'Mode':<14} {'nprocs':>6}  {'Status':<8} "
            f"{'Wall (s)':>10}  {'Speedup':>9}  "
            f"{'CPU (s)':>9}  {'BusyCores':>10}  {'Util%':>6}  "
            f"{'Nodes':>10}  {'Tria':>10}"
        )
        lines.append(
            f"  {'-'*14} {'-'*6}  {'-'*8} "
            f"{'-'*10}  {'-'*9}  "
            f"{'-'*9}  {'-'*10}  {'-'*6}  "
            f"{'-'*10}  {'-'*10}"
        )
        for r in summary.get("results", []):
            sp  = (
                f"{r.get('speedup_vs_baseline', 1.0):.2f}x"
                if "speedup_vs_baseline" in r else " — "
            )
            nd    = f"{r.get('n_nodes', 0):,}"
            tria  = f"{r.get('n_triangles', 0):,}"
            cpu   = f"{r.get('cpu_s', 0.0):.1f}"
            bc    = f"{r.get('busy_cores', 0.0):.1f}"
            ut    = f"{r.get('utilization_pct', 0.0):.0f}%"
            np_   = r.get("effective_nprocs", "?")
            lines.append(
                f"  {r['mode']:<14} {np_:>6}  "
                f"{_status_icon(r['status']):<8} "
                f"{r.get('wall_time_s', 0):>10.2f}  {sp:>9}  "
                f"{cpu:>9}  {bc:>10}  {ut:>6}  "
                f"{nd:>10}  {tria:>10}"
            )
            if r["status"] == "failed":
                lines.append(f"    ERROR: {r.get('error', 'unknown')}")
        lines.append("")

        # ── Correctness check results ────────────────────────────────
        checks = summary.get("correctness_checks", [])
        if checks:
            lines.append("  Correctness checks (pixel-exact vs serial_mp):")
            for c in checks:
                icon = "OK  " if c.get("match") else "FAIL"
                lines.append(
                    f"    {icon}  {c.get('vs_baseline','?')} vs "
                    f"{c.get('mode','?')}: {c.get('reason','?')}"
                )
            lines.append("")

        # ── Mesh quality stats ───────────────────────────────────────
        for r in summary.get("results", []):
            if r["status"] != "success":
                continue
            lines.append(f"  [{r['mode']}] Mesh size function stats:")
            lines.append(f"    hfun min  = {r.get('hfun_min', 'N/A'):.1f} m")
            lines.append(f"    hfun max  = {r.get('hfun_max', 'N/A'):.1f} m")
            lines.append(f"    hfun mean = {r.get('hfun_mean', 'N/A'):.1f} m")
            lines.append(f"    hfun std  = {r.get('hfun_std', 'N/A'):.1f} m")
            lines.append("")

        # ── Per-stage wall-clock breakdown ───────────────────────────
        any_stage = any(
            r.get("stage_times_s")
            for r in summary.get("results", [])
        )
        if any_stage:
            h("Per-stage wall-clock time (seconds)")
            stage_keys = [
                "geom_build_s", "hfun_build_s",
                "hfun_meshdata_s", "meshdriver_run_s",
            ]
            header = "  {:<14}".format("mode") + "".join(
                f"{k.replace('_s',''):>20}" for k in stage_keys
            ) + f"{'total':>12}"
            lines.append(header)
            lines.append("  " + "-" * (14 + 20 * len(stage_keys) + 12))
            for r in summary.get("results", []):
                st  = r.get("stage_times_s") or {}
                row = "  {:<14}".format(r["mode"])
                for k in stage_keys:
                    v = st.get(k)
                    row += f"{'—' if v is None else f'{v:.1f}':>20}"
                row += f"{r.get('wall_time_s', 0):>12.1f}"
                lines.append(row)
            lines.append("")
            lines.append(
                "  Note: hfun_meshdata is the MPI-parallelized stage."
            )
            lines.append(
                "  meshdriver_run (final mesh) is serial/global."
            )
            lines.append("")

        all_run_results.extend(summary.get("results", []))

        # ── cProfile hotspots ────────────────────────────────────────
        h("cProfile Hotspots (cumulative time)")
        for mode in _ALL_MODES:
            prof_path = _find_profile(results_dir, mode)
            if prof_path is not None:
                lines.append(f"\n--- {mode.upper()} ({prof_path}) ---")
                lines.append(
                    _load_profile_stats(prof_path, n_top=n_profile_top)
                )
            else:
                lines.append(f"\n--- {mode.upper()} ---")
                lines.append("  [no .prof file found]\n")

        # ── Per-stage cProfile breakdown ─────────────────────────────
        h("Pipeline Stage Breakdown (self-time seconds, from cProfile)")
        lines.append(
            "  Buckets cProfile self-time into pipeline stages.\n"
            "  Newly parallelized stages (Anas PR): apply_flow_limiter,\n"
            "  apply_const_val, apply_constraints, apply_contours,\n"
            "  apply_user_shapes.\n"
        )
        stage_names = list(_STAGE_MARKERS.keys())
        header = "  {:<22}".format("stage") + "".join(
            f"{m:>16}" for m in _ALL_MODES
        )
        lines.append(header)
        lines.append("  " + "-" * (22 + 16 * len(_ALL_MODES)))

        mode_buckets: Dict[str, Optional[Dict]] = {}
        for mode in _ALL_MODES:
            prof_path = _find_profile(results_dir, mode)
            mode_buckets[mode] = (
                _bucket_profile_by_stage(prof_path)
                if prof_path else None
            )

        for stage in stage_names:
            row = "  {:<22}".format(stage)
            for mode in _ALL_MODES:
                b = mode_buckets.get(mode)
                if b is None:
                    row += f"{'—':>16}"
                else:
                    row += f"{b.get(stage, 0.0):>16.2f}"
            lines.append(row)
        lines.append("")

    # ── Numerical equivalence check ─────────────────────────────────────
    h("Numerical Equivalence Check (serial_mp baseline)")
    serial_result = next(
        (r for r in all_run_results
         if r["mode"] == "serial_mp" and r["status"] == "success"),
        None,
    )
    if serial_result is None:
        serial_result = next(
            (r for r in all_run_results
             if r["mode"] == "serial_true" and r["status"] == "success"),
            None,
        )
    if serial_result:
        s_nodes = serial_result.get("n_nodes", 0)
        s_min   = serial_result.get("hfun_min",  float("nan"))
        s_max   = serial_result.get("hfun_max",  float("nan"))
        s_mean  = serial_result.get("hfun_mean", float("nan"))
        lines.append(
            f"  Baseline ({serial_result['mode']}): {s_nodes:,} nodes"
        )
        lines.append("")
        lines.append(
            f"  {'Mode':<14} {'Nodes':>10}  {'|Δnodes|':>10}  "
            f"{'|Δmin|':>10}  {'|Δmax|':>10}  {'|Δmean|':>10}  "
            f"{'Match':>8}"
        )
        lines.append(
            f"  {'-'*14} {'-'*10}  {'-'*10}  "
            f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}"
        )
        baseline_mode = serial_result["mode"]

        # Collect pixel-exact results from JSON correctness_checks
        # keyed by mode for quick lookup
        px_checks: Dict[str, bool] = {}
        for results_dir in results_dirs:
            s = _find_json(results_dir)
            if s:
                for c in s.get("correctness_checks", []):
                    px_checks[c.get("mode", "")] = c.get("match", False)

        for r in all_run_results:
            if r["mode"] == baseline_mode or r["status"] != "success":
                continue
            r_nodes = r.get("n_nodes", 0)
            d_nodes = abs(r_nodes - s_nodes)
            d_min   = abs(r.get("hfun_min",  float("nan")) - s_min)
            d_max   = abs(r.get("hfun_max",  float("nan")) - s_max)
            d_mean  = abs(r.get("hfun_mean", float("nan")) - s_mean)
            flag    = (
                " *** NODE MISMATCH ***"
                if d_nodes > 0.01 * s_nodes else ""
            )
            px = px_checks.get(r["mode"])
            px_str = (
                "pixel-OK" if px is True
                else "pixel-FAIL" if px is False
                else "—"
            )
            lines.append(
                f"  {r['mode']:<14} {r_nodes:>10,}  "
                f"{d_nodes:>10,}  "
                f"{d_min:>10.2f}  {d_max:>10.2f}  "
                f"{d_mean:>10.2f}  {px_str:>8}{flag}"
            )
    else:
        lines.append(
            "  Serial result not found — equivalence check skipped."
        )

    # ── Speedup summary CSV ──────────────────────────────────────────────
    h("Speedup Summary (CSV — copy-paste to spreadsheet)")
    serial_time = next(
        (r["wall_time_s"] for r in all_run_results
         if r["mode"] == "serial_mp" and r["status"] == "success"),
        None,
    )
    if serial_time is None:
        serial_time = next(
            (r["wall_time_s"] for r in all_run_results
             if r["mode"] == "serial_true" and r["status"] == "success"),
            None,
        )
    csv_io = io.StringIO()
    writer = csv.writer(csv_io)
    writer.writerow([
        "mode", "wall_time_s", "speedup_vs_serial_mp",
        "cpu_s", "busy_cores", "utilization_pct", "n_nodes",
    ])
    for r in all_run_results:
        if r["status"] != "success":
            continue
        sp = (serial_time / r["wall_time_s"]) if serial_time else None
        writer.writerow([
            r["mode"],
            f"{r.get('wall_time_s', 0):.3f}",
            f"{sp:.3f}" if sp else "",
            f"{r.get('cpu_s', 0.0):.1f}",
            f"{r.get('busy_cores', 0.0):.1f}",
            f"{r.get('utilization_pct', 0.0):.1f}",
            r.get("n_nodes", 0),
        ])
    lines.append(csv_io.getvalue())

    # ── Footer ───────────────────────────────────────────────────────────
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
        "--results-dir", nargs="+", type=Path, required=True,
        help="One or more results directories (output of run_benchmark.py).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("benchmark_report.txt"),
        help="Output report file (default: benchmark_report.txt)",
    )
    parser.add_argument(
        "--top", type=int, default=25,
        help="Top hotspots per profile (default: 25)",
    )
    args = parser.parse_args()
    build_report(args.results_dir, args.out, n_profile_top=args.top)


if __name__ == "__main__":
    main()
