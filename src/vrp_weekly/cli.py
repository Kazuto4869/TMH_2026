"""Command-line entry point for running weekly VRP solvers."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vrp_weekly.config import (
    CP_TIME_LIMIT_PER_DAY_SEC,
    INSERTION_WEIGHT,
    REGRET_WEIGHT,
    URGENCY_WEIGHT,
    WAITING_WEIGHT,
)
from vrp_weekly.evaluator import evaluate_weekly_schedule, print_metrics, print_schedule
from vrp_weekly.export import export_report_files, save_result_json, solver_results_dir
from vrp_weekly.io import load_instance, summarize_instance
from vrp_weekly.model_factory import create_solver, solver_names


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run a weekly VRP solver.")
    parser.add_argument("--locations", required=True, help="Path to locations.csv")
    parser.add_argument("--time-windows", required=True, help="Path to time_windows.csv")
    parser.add_argument("--solver", choices=solver_names() + ["earliest"], default="nearest", help="Solver to run.")
    parser.add_argument("--summary", action="store_true", help="Print input data summary and exit unless --solver is explicit.")
    parser.add_argument("--save-results", action="store_true", help="Save result JSON and report CSV files under results/.")
    parser.add_argument("--results-dir", default="results", help="Directory for saved results.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for solvers that need one.")
    parser.add_argument("--urgency-weight", type=float, default=URGENCY_WEIGHT, help="Regret solver urgency weight.")
    parser.add_argument("--regret-weight", type=float, default=REGRET_WEIGHT, help="Regret solver regret weight.")
    parser.add_argument("--insertion-weight", type=float, default=INSERTION_WEIGHT, help="Regret solver insertion weight.")
    parser.add_argument("--waiting-weight", type=float, default=WAITING_WEIGHT, help="Regret solver waiting penalty weight.")
    parser.add_argument("--cp-time-limit-per-day", type=int, default=CP_TIME_LIMIT_PER_DAY_SEC, help="CP solver time limit per day in seconds.")
    parser.add_argument("--cp-threads", type=int, default=1, help="CP solver worker thread count when supported by OR-Tools.")
    parser.add_argument("--cp-log-search", action="store_true", help="Print OR-Tools CP search log to the terminal.")
    parser.add_argument("--log-level", default="WARNING", help="Python logging level.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_args)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING), format="%(levelname)s:%(name)s:%(message)s")

    instance = load_instance(args.locations, args.time_windows)
    if args.summary:
        print_summary(summarize_instance(instance))
        if "--solver" not in raw_args:
            return 0

    solver = create_solver(
        args.solver,
        regret_weight=args.regret_weight,
        insertion_weight=args.insertion_weight,
        urgency_weight=args.urgency_weight,
        waiting_weight=args.waiting_weight,
        cp_time_limit_per_day=args.cp_time_limit_per_day,
        cp_threads=args.cp_threads,
        cp_log_search=args.cp_log_search,
        seed=args.seed,
    )
    schedule = solver.solve(instance)
    metrics = evaluate_weekly_schedule(instance, schedule)

    print(f"solver={solver.name}")
    print_schedule(schedule)
    print_metrics(metrics)

    if args.save_results:
        results_dir = Path(args.results_dir)
        save_result_json(solver_results_dir(results_dir, solver.name) / "result.json", solver.name, schedule, metrics)
        export_report_files(results_dir, solver.name, instance, schedule)
        print(f"saved_results={results_dir}")

    return 0


def print_summary(summary: dict[str, object]) -> None:
    """Print a clear data summary."""
    print("Data summary")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}={value:.3f}")
        else:
            print(f"{key}={value}")


if __name__ == "__main__":
    raise SystemExit(main())
