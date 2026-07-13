"""Compare saved solver result files without rerunning solvers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vrp_weekly.benchmark import BenchmarkTable
from vrp_weekly.config import METRIC_COLUMNS, OBJECTIVE_VERSION, SORT_BY
from vrp_weekly.evaluator import calculate_objective_value
from vrp_weekly.export import export_benchmark_plots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
BASELINE_SOLVERS = {"nearest", "deadline", "min_deferral"}
BASELINE_ORDER = {"nearest": 0, "deadline": 1, "min_deferral": 2}
CP_REFERENCE_SOLVER = "cp_rolling"
DISPLAY_COLUMNS = [
    ("solver", "solver"),
    ("delivered_count", "delivered"),
    ("incomplete_count", "incomplete"),
    ("total_deferral_days", "deferral"),
    ("total_distance_km", "distance"),
    ("total_waiting_time_min", "waiting"),
    ("objective_value", "objective"),
    ("gap_to_cp_percent", "gap_cp%"),
    ("runtime_sec", "runtime"),
]


def compare_saved_results(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    output_dir: str | Path | None = None,
    export_plots: bool = False,
) -> BenchmarkTable:
    """Read saved solver results and write a comparison summary."""
    results_path = Path(results_dir)
    comparison_dir = Path(output_dir) if output_dir is not None else results_path / "comparison"
    rows = [_row_from_result(path) for path in _result_files(results_path)]
    rows.sort(key=lambda row: tuple(row[column] for column in SORT_BY))
    reference_solver, reference_objective = _add_cp_reference_gaps(rows)

    frame = BenchmarkTable(rows)
    summary_path = comparison_dir / "benchmark_summary.csv"
    frame.to_csv(summary_path)
    report = build_detailed_comparison_report(frame, results_path, reference_solver, reference_objective)
    (comparison_dir / "comparison_table.txt").write_text(report, encoding="utf-8")
    if export_plots:
        export_benchmark_plots(summary_path, comparison_dir)
    return frame


def build_parser() -> argparse.ArgumentParser:
    """Build the saved-result comparison CLI parser."""
    parser = argparse.ArgumentParser(description="Compare saved weekly VRP solver results.")
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing schedules/{solver}/result.json files. Defaults to the project results directory.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for comparison output. Defaults to results/comparison.")
    parser.add_argument("--export-plots", action="store_true", help="Export comparison bar plots.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run saved-result comparison CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    frame = compare_saved_results(args.results_dir, args.output_dir, args.export_plots)
    print(format_comparison_table(frame))
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(args.results_dir) / "comparison"
    print(f"comparison_summary={output_dir / 'benchmark_summary.csv'}")
    print(f"comparison_report={output_dir / 'comparison_table.txt'}")
    return 0


def format_comparison_table(frame: BenchmarkTable) -> str:
    """Return a compact bordered table suitable for a normal terminal."""
    if not frame.rows:
        return "No saved solver results"
    headers = [label for _, label in DISPLAY_COLUMNS]
    display_rows = sorted(
        frame.rows,
        key=lambda row: (
            0 if row.get("solver") in BASELINE_SOLVERS else 1,
            BASELINE_ORDER.get(str(row.get("solver")), 0),
            float(row.get("objective_value", 0.0)),
        ),
    )
    values = [
        [_format_display_value(key, row.get(key, "")) for key, _ in DISPLAY_COLUMNS]
        for row in display_rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render_row(row: list[str], *, header: bool = False) -> str:
        cells = []
        for index, value in enumerate(row):
            if header or index == 0:
                cells.append(value.ljust(widths[index]))
            else:
                cells.append(value.rjust(widths[index]))
        return "| " + " | ".join(cells) + " |"

    lines = [separator, render_row(headers, header=True), separator]
    lines.extend(render_row(row) for row in values)
    lines.append(separator)
    lines.append("[B] baseline")
    return "\n".join(lines)


def _format_display_value(key: str, value: Any) -> str:
    """Format compact comparison cells without losing saved CSV precision."""
    if value in (None, ""):
        return "-"
    if key == "solver":
        solver = str(value)
        return f"{solver} [B]" if solver in BASELINE_SOLVERS else solver
    if key in {"total_distance_km", "objective_value", "runtime_sec"}:
        return f"{float(value):.3f}"
    if key == "gap_to_cp_percent":
        return f"{float(value):.2f}%"
    return str(value)


def build_detailed_comparison_report(
    frame: BenchmarkTable,
    results_dir: str | Path,
    reference_solver: str | None = None,
    reference_objective: float | None = None,
) -> str:
    """Build the saved text report with ranking, baseline deltas, and daily routes."""
    results_path = Path(results_dir)
    payloads = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in _result_files(results_path)
    }
    if reference_objective is None:
        reference_solver, reference_objective = _add_cp_reference_gaps(frame.rows)
    reference_text = "N/A" if reference_objective is None else f"{reference_objective:.6f}"
    reference_solver_text = reference_solver or "N/A"
    lines = [
        "WEEKLY VRP SAVED-RESULT COMPARISON",
        "Official objective = 1000*incomplete + 100*deferral + 10*distance + waiting",
        "Lower objective is better.",
        f"CP reference solver = {reference_solver_text}",
        f"CP incumbent objective = {reference_text}",
        "gap_cp% = 100 * (model objective - CP incumbent objective) / CP incumbent objective.",
        "Negative gap_cp% means the model has a lower objective than the current CP incumbent.",
        "gap_cp% is not a certified optimality gap because cp_rolling is not globally exact.",
        "The reference is always cp_rolling's final seven-day evaluator objective, never a daily CP objective.",
        "",
        "1. SUMMARY",
        format_comparison_table(frame),
        "",
        "2. MODELS COMPARED WITH BASELINES",
        "Delta convention: model - baseline. Negative objective delta means the model is better.",
        "",
        _format_baseline_comparisons(frame),
        "",
        "3. MODEL DETAILS AND DAILY ROUTES",
    ]
    rows_by_solver = {str(row["solver"]): row for row in frame.rows}
    ordered_solvers = [
        str(row["solver"])
        for row in sorted(
            frame.rows,
            key=lambda row: (
                0 if row["solver"] in BASELINE_SOLVERS else 1,
                BASELINE_ORDER.get(str(row["solver"]), 0),
                float(row["objective_value"]),
            ),
        )
    ]
    for solver in ordered_solvers:
        payload = payloads.get(solver)
        if payload is None:
            continue
        lines.extend(["", _format_model_details(solver, rows_by_solver[solver], payload)])
    return "\n".join(lines).rstrip() + "\n"


def _format_baseline_comparisons(frame: BenchmarkTable) -> str:
    """Return one compact delta table for every non-baseline/baseline pair."""
    rows_by_solver = {str(row["solver"]): row for row in frame.rows}
    baselines = [solver for solver in BASELINE_ORDER if solver in rows_by_solver]
    models = sorted(
        (row for row in frame.rows if row["solver"] not in BASELINE_SOLVERS),
        key=lambda row: float(row["objective_value"]),
    )
    headers = ["model", "baseline", "d_deliv", "d_incomp", "d_defer", "d_dist", "d_wait", "d_obj", "result"]
    values: list[list[str]] = []
    for model in models:
        for baseline_name in baselines:
            baseline = rows_by_solver[baseline_name]
            objective_delta = float(model["objective_value"]) - float(baseline["objective_value"])
            values.append(
                [
                    str(model["solver"]),
                    baseline_name,
                    _format_delta(model["delivered_count"], baseline["delivered_count"], 0),
                    _format_delta(model["incomplete_count"], baseline["incomplete_count"], 0),
                    _format_delta(model["total_deferral_days"], baseline["total_deferral_days"], 0),
                    _format_delta(model["total_distance_km"], baseline["total_distance_km"], 3),
                    _format_delta(model["total_waiting_time_min"], baseline["total_waiting_time_min"], 0),
                    f"{objective_delta:+.3f}",
                    "BETTER" if objective_delta < 0 else "WORSE" if objective_delta > 0 else "TIE",
                ]
            )
    return _render_bordered_table(headers, values, left_columns={0, 1, 8}) if values else "No baseline comparisons available."


def _format_model_details(solver: str, row: dict[str, Any], payload: dict[str, Any]) -> str:
    """Return aggregate metrics and all saved daily routes for one model."""
    baseline_suffix = " [BASELINE]" if solver in BASELINE_SOLVERS else ""
    runtime = row.get("runtime_sec", "")
    runtime_text = "-" if runtime in (None, "") else f"{float(runtime):.3f} sec"
    lines = [
        "=" * 100,
        f"MODEL: {solver}{baseline_suffix}",
        f"delivered={row['delivered_count']} | incomplete={row['incomplete_count']} | "
        f"deferral={row['total_deferral_days']} | distance={float(row['total_distance_km']):.3f} km | "
        f"waiting={row['total_waiting_time_min']} min | objective={float(row['objective_value']):.3f} | "
        f"gap_cp={_format_gap_for_detail(row.get('gap_to_cp_percent', ''))} | "
        f"runtime={runtime_text}",
        f"active_days={row['active_days']} | route_duration={row['total_route_duration_min']} min | "
        f"travel={row['total_travel_time_min']} min | service={row['total_service_time_min']} min | "
        f"hard_feasible={row['hard_feasible']}",
        "Daily routes:",
    ]
    routes = sorted(payload.get("schedule", {}).get("routes", []), key=lambda route: int(route.get("day", 0)))
    routes_by_day = {int(route.get("day", 0)): route for route in routes}
    for day in range(1, 8):
        route = routes_by_day.get(day, {})
        stops = [stop for stop in route.get("stops", []) if stop.get("hard_feasible", True)]
        sequence = " -> ".join(["DEPOT", *(str(stop.get("customer_id", "")) for stop in stops), "DEPOT"])
        lines.append(
            f"Day {day}: stops={len(stops)} | distance={float(route.get('route_distance_km', 0.0)):.3f} km | "
            f"travel={int(route.get('route_travel_time_min', 0))} min | "
            f"waiting={int(route.get('route_waiting_time_min', 0))} min | "
            f"service={int(route.get('route_service_time_min', 0))} min | "
            f"duration={int(route.get('route_duration_min', 0))} min | "
            f"return={_format_minutes(route.get('return_to_depot_time', 0))} | "
            f"feasible={route.get('hard_feasible', True)}"
        )
        lines.append(f"  route: {sequence}")
    return "\n".join(lines)


def _add_cp_reference_gaps(rows: list[dict[str, Any]]) -> tuple[str | None, float | None]:
    """Add gaps from the saved final seven-day CP rolling result."""
    reference_row = next((row for row in rows if row.get("solver") == CP_REFERENCE_SOLVER), None)
    if reference_row is None:
        for row in rows:
            row["gap_to_cp_percent"] = ""
        return None, None
    reference = float(reference_row["objective_value"])
    for row in rows:
        row["gap_to_cp_percent"] = 100.0 * (float(row["objective_value"]) - reference) / max(abs(reference), 1e-12)
    return CP_REFERENCE_SOLVER, reference


def _format_gap_for_detail(value: Any) -> str:
    """Format an optional best-known gap for the detailed model section."""
    return "N/A" if value in (None, "") else f"{float(value):.2f}%"


def _format_delta(model_value: Any, baseline_value: Any, decimals: int) -> str:
    """Format a signed model-minus-baseline metric delta."""
    delta = float(model_value) - float(baseline_value)
    return f"{delta:+.{decimals}f}"


def _format_minutes(value: Any) -> str:
    """Format a minute-of-day value as HH:MM when possible."""
    minutes = int(value or 0)
    if 0 <= minutes <= 24 * 60:
        return f"{minutes // 60:02d}:{minutes % 60:02d}"
    return f"{minutes}min"


def _render_bordered_table(headers: list[str], values: list[list[str]], left_columns: set[int]) -> str:
    """Render a fixed-width ASCII table with selected left-aligned columns."""
    widths = [max(len(headers[index]), *(len(row[index]) for row in values)) for index in range(len(headers))]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render(row: list[str], header: bool = False) -> str:
        cells = [
            value.ljust(widths[index]) if header or index in left_columns else value.rjust(widths[index])
            for index, value in enumerate(row)
        ]
        return "| " + " | ".join(cells) + " |"

    return "\n".join([separator, render(headers, header=True), separator, *(render(row) for row in values), separator])


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
    if payload.get("objective_version") == OBJECTIVE_VERSION or metrics.get("objective_version") == OBJECTIVE_VERSION:
        objective_value = metrics.get("objective_value", payload.get("objective_value"))
    else:
        objective_value = calculate_objective_value(
            incomplete_count=int(metrics["incomplete_count"]),
            total_deferral_days=int(metrics["total_deferral_days"]),
            total_distance_km=float(metrics["total_distance_km"]),
            total_waiting_time_min=float(metrics["total_waiting_time_min"]),
        )
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
        "objective_value": objective_value,
        "runtime_sec": metrics.get("runtime_sec", solver_status.get("runtime_sec", "")),
        "max_day_gap_percent": solver_status.get("max_day_gap_percent", ""),
        "total_fixed_impossible_arcs": solver_status.get("total_fixed_impossible_arcs", ""),
        "average_fixed_arc_ratio": solver_status.get("average_fixed_arc_ratio", ""),
        "total_route_interval_count": solver_status.get("total_route_interval_count", ""),
        "route_no_overlap_days": solver_status.get("route_no_overlap_days", ""),
        "total_remaining_after_week": solver_status.get("total_remaining_after_week", ""),
        "hard_feasible": metrics["hard_feasible"],
        "cp_reference_eligible": bool(
            payload.get("solver", path.parent.name) == CP_REFERENCE_SOLVER
            and solver_status.get("status") in {"FEASIBLE", "OPTIMAL"}
        ),
    }
    if row["solver"] in DISPLAY_NAMES:
        row["solver_display_name"] = DISPLAY_NAMES[row["solver"]]
    return {**{column: row[column] for column in METRIC_COLUMNS}, "cp_reference_eligible": row["cp_reference_eligible"]}


if __name__ == "__main__":
    raise SystemExit(main())
