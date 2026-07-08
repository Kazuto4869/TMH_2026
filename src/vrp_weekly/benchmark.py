"""Benchmark runner for comparing weekly VRP solvers."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

from vrp_weekly.config import (
    CP_TIME_LIMIT_PER_DAY_SEC,
    INSERTION_WEIGHT,
    METRIC_COLUMNS,
    REGRET_WEIGHT,
    SORT_BY,
    URGENCY_WEIGHT,
    WAITING_WEIGHT,
)
from vrp_weekly.evaluator import calculate_objective, evaluate_weekly_schedule
from vrp_weekly.export import export_benchmark_plots, export_report_files, save_result_json, solver_results_dir
from vrp_weekly.io import load_instance
from vrp_weekly.core import Instance
from vrp_weekly.model_factory import create_solver, solver_names


class BenchmarkTable:
    """Small DataFrame-like table used to avoid pandas/OR-Tools import conflicts."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        """Create a table from row dictionaries."""
        self.rows = rows
        self.loc = _LocIndexer(self)

    def to_csv(self, path: str | Path, index: bool = False) -> None:
        """Write the table to a CSV file."""
        del index
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.rows[0]) if self.rows else []
        with output_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def to_string(self, index: bool = False) -> str:
        """Return a fixed-width table string similar to pandas."""
        del index
        if not self.rows:
            return "Empty benchmark table"
        headers = list(self.rows[0])
        string_rows = [[_format_cell(row.get(header, "")) for header in headers] for row in self.rows]
        widths = [
            max(len(header), *(len(row[column_index]) for row in string_rows))
            for column_index, header in enumerate(headers)
        ]
        lines = [" ".join(header.rjust(widths[index]) for index, header in enumerate(headers))]
        for row in string_rows:
            lines.append(" ".join(value.rjust(widths[index]) for index, value in enumerate(row)))
        return "\n".join(lines)


class _LocIndexer:
    """Minimal `.loc[row, column]` accessor for tests and simple scripts."""

    def __init__(self, table: BenchmarkTable) -> None:
        self.table = table

    def __getitem__(self, key: tuple[int, str]) -> Any:
        row_index, column = key
        return self.table.rows[row_index][column]


def run_benchmark(
    instance: Instance,
    solver_names_to_run: list[str],
    results_dir: str | Path = "results",
    export_report: bool = False,
    **solver_kwargs: Any,
) -> BenchmarkTable:
    """Run selected solvers and return a metrics table."""
    rows: list[dict[str, Any]] = []
    output_dir = Path(results_dir)
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for solver_name in solver_names_to_run:
        solver = create_solver(solver_name, **solver_kwargs)
        start_time = time.perf_counter()
        schedule = solver.solve(instance)
        runtime_sec = time.perf_counter() - start_time
        metrics = evaluate_weekly_schedule(instance, schedule)
        save_result_json(solver_results_dir(output_dir, solver.name) / "result.json", solver.name, schedule, metrics)
        if export_report:
            export_report_files(output_dir, solver.name, instance, schedule)

        row = metrics.to_dict()
        row.update(
            {
                "solver": solver.name,
                "active_days": metrics.number_of_active_days,
                "runtime_sec": runtime_sec,
                "objective_value": calculate_objective(
                    metrics.incomplete_count,
                    metrics.total_deferral_days,
                    metrics.total_distance_km,
                    metrics.total_waiting_time_min,
                    active_days=metrics.number_of_active_days,
                    total_route_duration_min=metrics.total_route_duration_min,
                ),
            }
        )
        row.pop("violations", None)
        rows.append({column: row[column] for column in METRIC_COLUMNS})

    rows.sort(key=lambda row: tuple(row[column] for column in SORT_BY))
    frame = BenchmarkTable(rows)
    summary_path = comparison_dir / "benchmark_summary.csv"
    frame.to_csv(summary_path)
    if export_report:
        export_benchmark_plots(summary_path, comparison_dir)
    return frame


def build_parser() -> argparse.ArgumentParser:
    """Build benchmark CLI parser."""
    parser = argparse.ArgumentParser(description="Benchmark weekly VRP solvers.")
    parser.add_argument("--locations", required=True, help="Path to locations.csv")
    parser.add_argument("--time-windows", required=True, help="Path to time_windows.csv")
    parser.add_argument("--solvers", nargs="+", choices=solver_names() + ["earliest"], default=["nearest", "deadline", "regret"])
    parser.add_argument("--results-dir", default="results", help="Directory for benchmark outputs.")
    parser.add_argument("--export-report", action="store_true", help="Export report CSV and PNG files.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--urgency-weight", type=float, default=URGENCY_WEIGHT, help="Regret solver urgency weight.")
    parser.add_argument("--regret-weight", type=float, default=REGRET_WEIGHT, help="Regret solver regret weight.")
    parser.add_argument("--insertion-weight", type=float, default=INSERTION_WEIGHT, help="Regret solver insertion weight.")
    parser.add_argument("--waiting-weight", type=float, default=WAITING_WEIGHT, help="Waiting penalty weight.")
    parser.add_argument("--cp-time-limit-per-day", type=int, default=CP_TIME_LIMIT_PER_DAY_SEC, help="CP solver time limit per day.")
    parser.add_argument("--log-level", default="WARNING", help="Python logging level.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run benchmark CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING), format="%(levelname)s:%(name)s:%(message)s")
    instance = load_instance(args.locations, args.time_windows)
    frame = run_benchmark(
        instance,
        args.solvers,
        results_dir=args.results_dir,
        export_report=args.export_report,
        regret_weight=args.regret_weight,
        insertion_weight=args.insertion_weight,
        urgency_weight=args.urgency_weight,
        waiting_weight=args.waiting_weight,
        cp_time_limit_per_day=args.cp_time_limit_per_day,
        seed=args.seed,
    )
    print(frame.to_string(index=False))
    print(f"benchmark_summary={Path(args.results_dir) / 'comparison' / 'benchmark_summary.csv'}")
    return 0


def _format_cell(value: Any) -> str:
    """Format one table cell for display."""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())

