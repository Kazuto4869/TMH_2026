"""Compare saved solver result files without rerunning solvers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vrp_weekly.benchmark import BenchmarkTable
from vrp_weekly.config import METRIC_COLUMNS, SORT_BY
from vrp_weekly.export import export_benchmark_plots

DISPLAY_NAMES = {
    "cp_rolling_repair": "CP Rolling + Weekly Repair",
}


def compare_saved_results(
    results_dir: str | Path = "results",
    output_dir: str | Path | None = None,
    export_plots: bool = False,
) -> BenchmarkTable:
    """Read saved solver results and write a comparison summary."""
    results_path = Path(results_dir)
    comparison_dir = Path(output_dir) if output_dir is not None else results_path / "comparison"
    rows = [_row_from_result(path) for path in _result_files(results_path)]
    rows.sort(key=lambda row: tuple(row[column] for column in SORT_BY))

    frame = BenchmarkTable(rows)
    summary_path = comparison_dir / "benchmark_summary.csv"
    frame.to_csv(summary_path)
    if export_plots:
        export_benchmark_plots(summary_path, comparison_dir)
    return frame


def build_parser() -> argparse.ArgumentParser:
    """Build the saved-result comparison CLI parser."""
    parser = argparse.ArgumentParser(description="Compare saved weekly VRP solver results.")
    parser.add_argument("--results-dir", default="results", help="Directory containing schedules/{solver}/result.json files.")
    parser.add_argument("--output-dir", default=None, help="Directory for comparison output. Defaults to results/comparison.")
    parser.add_argument("--export-plots", action="store_true", help="Export comparison bar plots.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run saved-result comparison CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    frame = compare_saved_results(args.results_dir, args.output_dir, args.export_plots)
    print(frame.to_string(index=False))
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(args.results_dir) / "comparison"
    print(f"comparison_summary={output_dir / 'benchmark_summary.csv'}")
    return 0


def _result_files(results_dir: Path) -> list[Path]:
    """Return saved solver result files in deterministic solver order."""
    schedules_dir = results_dir / "schedules"
    paths = sorted(schedules_dir.glob("*/result.json"))
    if not paths:
        raise FileNotFoundError(f"No saved result files found under {schedules_dir}")
    return paths


def _row_from_result(path: Path) -> dict[str, Any]:
    """Convert one saved result JSON into a comparison table row."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = dict(payload["metrics"])
    solver_status = payload.get("solver_status", {})
    row: dict[str, Any] = {
        "solver": payload.get("solver", path.parent.name),
        "delivered_count": metrics["delivered_count"],
        "incomplete_count": metrics["incomplete_count"],
        "active_days": metrics.get("active_days", metrics.get("number_of_active_days", 0)),
        "total_deferral_days": metrics["total_deferral_days"],
        "total_distance_km": metrics["total_distance_km"],
        "total_travel_time_min": metrics["total_travel_time_min"],
        "total_waiting_time_min": metrics["total_waiting_time_min"],
        "total_service_time_min": metrics["total_service_time_min"],
        "total_route_duration_min": metrics["total_route_duration_min"],
        "objective_value": metrics["objective_value"],
        "runtime_sec": metrics.get("runtime_sec", solver_status.get("runtime_sec", "")),
        "max_day_gap_percent": solver_status.get("max_day_gap_percent", ""),
        "total_fixed_impossible_arcs": solver_status.get("total_fixed_impossible_arcs", ""),
        "average_fixed_arc_ratio": solver_status.get("average_fixed_arc_ratio", ""),
        "total_route_interval_count": solver_status.get("total_route_interval_count", ""),
        "route_no_overlap_days": solver_status.get("route_no_overlap_days", ""),
        "total_remaining_after_week": solver_status.get("total_remaining_after_week", ""),
        "hard_feasible": metrics["hard_feasible"],
    }
    if row["solver"] in DISPLAY_NAMES:
        row["solver_display_name"] = DISPLAY_NAMES[row["solver"]]
    return {column: row[column] for column in METRIC_COLUMNS}


if __name__ == "__main__":
    raise SystemExit(main())
