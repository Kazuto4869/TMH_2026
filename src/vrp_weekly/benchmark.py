"""Benchmark runner for comparing weekly VRP solvers."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

from vrp_weekly.config import (
    METRIC_COLUMNS,
    SORT_BY,
)
from vrp_weekly.evaluator import calculate_objective, evaluate_weekly_schedule
from vrp_weekly.export import export_benchmark_plots, export_report_files, save_result_json, solver_results_dir
from vrp_weekly.io import load_instance
from vrp_weekly.core import Instance
from vrp_weekly.model_factory import create_solver, solver_names

CP_GRID_CANDIDATE_LIMITS = [30, 40, 50, 60, 80]
CP_GRID_TIME_LIMITS_PER_DAY = [30, 60, 120]


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
            export_report_files(output_dir, solver.name, instance, schedule, metrics)

        row = metrics.to_dict()
        row.update(
            {
                "solver": solver.name,
                "active_days": metrics.number_of_active_days,
                "runtime_sec": runtime_sec,
                "max_day_gap_percent": _blank_if_missing(schedule.solver_status.get("max_day_gap_percent", "")),
                "total_fixed_impossible_arcs": _blank_if_missing(schedule.solver_status.get("total_fixed_impossible_arcs", "")),
                "average_fixed_arc_ratio": _blank_if_missing(schedule.solver_status.get("average_fixed_arc_ratio", "")),
                "total_route_interval_count": _blank_if_missing(schedule.solver_status.get("total_route_interval_count", "")),
                "route_no_overlap_days": _blank_if_missing(schedule.solver_status.get("route_no_overlap_days", "")),
                "total_remaining_after_week": _blank_if_missing(schedule.solver_status.get("total_remaining_after_week", "")),
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


def run_cp_diagnostic_grid(
    instance: Instance,
    results_dir: str | Path = "results",
    candidate_limits: list[int] | None = None,
    time_limits_per_day: list[int] | None = None,
    **solver_kwargs: Any,
) -> BenchmarkTable:
    """Run rolling CP over a candidate/time diagnostic grid."""
    output_dir = Path(results_dir)
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    grid_path = comparison_dir / "cp_diagnostic_grid.csv"
    rows: list[dict[str, Any]] = []
    if grid_path.exists():
        with grid_path.open(newline="", encoding="utf-8") as file_obj:
            rows = list(csv.DictReader(file_obj))
    completed_profiles = {
        (int(row["candidate_limit"]), int(row["time_limit_per_day_sec"]))
        for row in rows
        if row.get("candidate_limit") and row.get("time_limit_per_day_sec")
    }
    candidate_limits = CP_GRID_CANDIDATE_LIMITS if candidate_limits is None else candidate_limits
    time_limits_per_day = CP_GRID_TIME_LIMITS_PER_DAY if time_limits_per_day is None else time_limits_per_day

    for candidate_limit in candidate_limits:
        for time_limit_per_day_sec in time_limits_per_day:
            if (candidate_limit, time_limit_per_day_sec) in completed_profiles:
                print(
                    f"cp_diagnostic_grid skip completed candidate_limit={candidate_limit} "
                    f"time_limit_per_day_sec={time_limit_per_day_sec}",
                    flush=True,
                )
                continue
            print(
                f"cp_diagnostic_grid profile candidate_limit={candidate_limit} "
                f"time_limit_per_day_sec={time_limit_per_day_sec}",
                flush=True,
            )
            kwargs = dict(solver_kwargs)
            kwargs.update(
                {
                    "cp_max_candidates_per_day": candidate_limit,
                    "cp_time_limit_per_day_sec": time_limit_per_day_sec,
                    "cp_two_phase_objective": True,
                }
            )
            solver = create_solver("cp_rolling", **kwargs)
            start_time = time.perf_counter()
            schedule = solver.solve(instance)
            runtime_sec = time.perf_counter() - start_time
            metrics = evaluate_weekly_schedule(instance, schedule)
            statuses = schedule.solver_status.get("day_statuses", {})
            day_statuses = statuses if isinstance(statuses, dict) else {}
            rows.append(
                {
                    "candidate_limit": candidate_limit,
                    "time_limit_per_day_sec": time_limit_per_day_sec,
                    "delivered_count": metrics.delivered_count,
                    "incomplete_count": metrics.incomplete_count,
                    "total_deferral_days": metrics.total_deferral_days,
                    "total_distance_km": metrics.total_distance_km,
                    "phase1_optimal_days": _count_day_status(day_statuses, "phase1_status", "OPTIMAL"),
                    "phase2_optimal_days": _count_day_status(day_statuses, "phase2_status", "OPTIMAL"),
                    "feasible_days": sum(
                        1 for day_status in day_statuses.values() if day_status.get("status") in {"OPTIMAL", "FEASIBLE"}
                    ),
                    "unknown_or_infeasible_days": sum(
                        1
                        for day_status in day_statuses.values()
                        if day_status.get("status") not in {"NO_CANDIDATES", "OPTIMAL", "FEASIBLE"}
                    ),
                    "total_runtime_sec": runtime_sec,
                    "daily_phase1_delivered_count": _join_daily(day_statuses, "phase1_delivered_count"),
                    "daily_phase1_best_bound": _join_daily(day_statuses, "phase1_best_bound"),
                    "daily_phase1_gap": _join_daily(day_statuses, "phase1_gap_percent"),
                    "daily_phase2_gap": _join_daily(day_statuses, "phase2_gap_percent"),
                }
            )
            BenchmarkTable(rows).to_csv(grid_path)
            completed_profiles.add((candidate_limit, time_limit_per_day_sec))

    frame = BenchmarkTable(rows)
    frame.to_csv(grid_path)
    return frame


def build_parser() -> argparse.ArgumentParser:
    """Build benchmark CLI parser."""
    parser = argparse.ArgumentParser(description="Benchmark weekly VRP solvers.")
    parser.add_argument("--locations", required=True, help="Path to locations.csv")
    parser.add_argument("--time-windows", required=True, help="Path to time_windows.csv")
    parser.add_argument("--solvers", nargs="+", choices=solver_names() + ["earliest"], default=["nearest", "deadline", "min_deferral"])
    parser.add_argument("--results-dir", default="results", help="Directory for benchmark outputs.")
    parser.add_argument("--export-report", action="store_true", help="Export report CSV and PNG files.")
    parser.add_argument("--cp-diagnostic-grid", action="store_true", help="Run cp_rolling candidate/time diagnostic grid.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--heuristic-max-candidates-per-day", type=int, default=None, help="Limit daily candidates for heuristic insertion solvers.")
    parser.add_argument("--heuristic-random-seed", type=int, default=1, help="Random seed for heuristic solvers.")
    parser.add_argument("--use-local-search", dest="heuristic_use_local_search", action="store_true", default=None, help="Enable local search for compatible heuristics.")
    parser.add_argument("--no-local-search", dest="heuristic_use_local_search", action="store_false", help="Disable optional local search for compatible heuristics.")
    parser.add_argument("--local-search-time-limit-sec", type=int, default=10, help="Local search time limit per route in seconds.")
    parser.add_argument("--local-search-max-iterations", type=int, default=100, help="Local search iteration limit per route.")
    parser.add_argument("--ga-population-size", type=int, default=30, help="Hybrid genetic VNS population size.")
    parser.add_argument("--ga-generations", type=int, default=50, help="Hybrid genetic VNS generation limit.")
    parser.add_argument("--ga-elite-size", type=int, default=5, help="Hybrid genetic VNS elite count.")
    parser.add_argument("--ga-mutation-rate", type=float, default=0.10, help="Hybrid genetic VNS mutation probability.")
    parser.add_argument("--ga-crossover-rate", type=float, default=0.80, help="Hybrid genetic VNS crossover probability.")
    parser.add_argument("--ga-time-limit-sec", type=int, default=120, help="Hybrid genetic VNS time limit in seconds.")
    parser.add_argument("--cp-time-limit-sec", type=int, default=60, help="Full-week CP-SAT time limit in seconds.")
    parser.add_argument("--cp-time-limit-per-day-sec", type=int, default=10, help="Rolling CP-SAT time limit per day.")
    parser.add_argument("--cp-time-limit-per-day", type=int, default=None, help="Deprecated alias for --cp-time-limit-per-day-sec.")
    parser.add_argument("--cp-max-customers", type=int, default=40, help="Limit customers for full-week CP-SAT.")
    parser.add_argument("--cp-max-candidates-per-day", type=int, default=None, help="Limit daily candidates for rolling CP-SAT.")
    parser.add_argument("--cp-workers", type=int, default=8, help="CP-SAT worker count.")
    parser.add_argument("--cp-log-search", action="store_true", help="Print CP-SAT search logs.")
    parser.add_argument("--cp-two-phase-objective", dest="cp_two_phase_objective", action="store_true", default=True, help="Use pure two-phase rolling CP objective.")
    parser.add_argument("--cp-single-phase-objective", dest="cp_two_phase_objective", action="store_false", help="Use pure single-phase rolling CP objective.")
    parser.add_argument("--cp-random-seed", type=int, default=1, help="CP-SAT random seed.")
    parser.add_argument("--cp-use-decision-strategy", dest="cp_use_decision_strategy", action="store_true", default=True, help="Add an optional y-first CP decision strategy.")
    parser.add_argument("--cp-no-decision-strategy", dest="cp_use_decision_strategy", action="store_false", help="Disable the optional CP decision strategy.")
    parser.add_argument("--cp-use-service-no-overlap", dest="cp_use_service_no_overlap", action="store_true", default=True, help="Add optional service intervals and NoOverlap.")
    parser.add_argument("--cp-no-service-no-overlap", dest="cp_use_service_no_overlap", action="store_false", help="Disable service NoOverlap intervals.")
    parser.add_argument("--cp-candidate-strategy", choices=["urgent", "hybrid"], default="hybrid", help="Daily candidate filtering strategy for rolling CP.")
    parser.add_argument("--cp-phase1-only", dest="cp_solve_phase2", action="store_false", default=True, help="Run only phase 1 delivered-count CP for diagnostics.")
    parser.add_argument("--cp-solve-phase2", dest="cp_solve_phase2", action="store_true", help="Run phase 2 route-cost CP after phase 1.")
    parser.add_argument("--log-level", default="WARNING", help="Python logging level.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run benchmark CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING), format="%(levelname)s:%(name)s:%(message)s")
    instance = load_instance(args.locations, args.time_windows)
    cp_time_limit_per_day_sec = (
        args.cp_time_limit_per_day if args.cp_time_limit_per_day is not None else args.cp_time_limit_per_day_sec
    )
    common_kwargs = {
        "cp_time_limit_sec": args.cp_time_limit_sec,
        "cp_time_limit_per_day_sec": cp_time_limit_per_day_sec,
        "cp_max_customers": args.cp_max_customers,
        "cp_max_candidates_per_day": args.cp_max_candidates_per_day,
        "cp_workers": args.cp_workers,
        "cp_log_search": args.cp_log_search,
        "cp_two_phase_objective": args.cp_two_phase_objective,
        "cp_random_seed": args.cp_random_seed,
        "cp_use_decision_strategy": args.cp_use_decision_strategy,
        "cp_use_service_no_overlap": args.cp_use_service_no_overlap,
        "cp_candidate_strategy": args.cp_candidate_strategy,
        "cp_solve_phase2": args.cp_solve_phase2,
        "heuristic_max_candidates_per_day": args.heuristic_max_candidates_per_day,
        "heuristic_random_seed": args.heuristic_random_seed,
        "heuristic_use_local_search": args.heuristic_use_local_search,
        "local_search_time_limit_sec": args.local_search_time_limit_sec,
        "local_search_max_iterations": args.local_search_max_iterations,
        "ga_population_size": args.ga_population_size,
        "ga_generations": args.ga_generations,
        "ga_elite_size": args.ga_elite_size,
        "ga_mutation_rate": args.ga_mutation_rate,
        "ga_crossover_rate": args.ga_crossover_rate,
        "ga_time_limit_sec": args.ga_time_limit_sec,
        "seed": args.seed,
    }
    if args.cp_diagnostic_grid:
        frame = run_cp_diagnostic_grid(
            instance,
            results_dir=args.results_dir,
            **common_kwargs,
        )
        print(frame.to_string(index=False))
        print(f"cp_diagnostic_grid={Path(args.results_dir) / 'comparison' / 'cp_diagnostic_grid.csv'}")
        return 0

    frame = run_benchmark(
        instance,
        args.solvers,
        results_dir=args.results_dir,
        export_report=args.export_report,
        **common_kwargs,
    )
    print(frame.to_string(index=False))
    print(f"benchmark_summary={Path(args.results_dir) / 'comparison' / 'benchmark_summary.csv'}")
    return 0


def _format_cell(value: Any) -> str:
    """Format one table cell for display."""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _blank_if_missing(value: Any) -> Any:
    """Return blanks for missing optional solver-status fields."""
    return "" if value is None else value


def _count_day_status(day_statuses: dict[Any, Any], field: str, value: str) -> int:
    """Count day statuses where a field equals a value."""
    return sum(1 for day_status in day_statuses.values() if isinstance(day_status, dict) and day_status.get(field) == value)


def _join_daily(day_statuses: dict[Any, Any], field: str) -> str:
    """Return a semicolon-separated day:value diagnostic string."""
    values: list[str] = []
    for day in sorted(day_statuses, key=lambda raw_day: int(raw_day)):
        day_status = day_statuses[day]
        if isinstance(day_status, dict):
            values.append(f"{day}:{day_status.get(field, '')}")
    return ";".join(values)


if __name__ == "__main__":
    raise SystemExit(main())

