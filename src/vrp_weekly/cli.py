"""Command-line entry point for running weekly VRP solvers."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vrp_weekly.evaluator import evaluate_weekly_schedule, print_metrics, print_schedule
from vrp_weekly.export import (
    export_report_files,
    format_gap_percent,
    save_result_json,
    solver_results_dir,
    solver_status_summary,
)
from vrp_weekly.io import load_instance, summarize_instance
from vrp_weekly.model_factory import create_solver, solver_names


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run a weekly VRP model.")
    parser.add_argument("--locations", required=True, help="Path to locations.csv")
    parser.add_argument("--time-windows", required=True, help="Path to time_windows.csv")
    parser.add_argument("--solver", choices=solver_names() + ["earliest", "cp"], default="nearest", help="Model to run.")
    parser.add_argument("--summary", action="store_true", help="Print input data summary and exit unless --solver is explicit.")
    parser.add_argument("--save-results", action="store_true", help="Save result JSON and report CSV files under results/.")
    parser.add_argument("--results-dir", default="results", help="Directory for saved results.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for solvers that need one.")
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
    parser.add_argument("--cp-time-limit-per-day-sec", type=int, default=10, help="Rolling CP-SAT time limit per day in seconds.")
    parser.add_argument("--cp-time-limit-per-day", type=int, default=None, help="Deprecated alias for --cp-time-limit-per-day-sec.")
    parser.add_argument("--cp-max-customers", type=int, default=40, help="Limit customers for full-week CP-SAT.")
    parser.add_argument("--cp-max-candidates-per-day", type=int, default=None, help="Limit daily candidates for rolling CP-SAT.")
    parser.add_argument("--cp-workers", type=int, default=8, help="CP-SAT worker count.")
    parser.add_argument("--cp-threads", type=int, default=None, help="Deprecated alias for --cp-workers.")
    parser.add_argument("--cp-log-search", action="store_true", help="Print OR-Tools CP-SAT search log to the terminal.")
    parser.add_argument("--cp-two-phase-objective", dest="cp_two_phase_objective", action="store_true", default=True, help="Use pure two-phase rolling CP objective.")
    parser.add_argument("--cp-single-phase-objective", dest="cp_two_phase_objective", action="store_false", help="Use the pure single-phase rolling CP objective.")
    parser.add_argument("--cp-random-seed", type=int, default=1, help="CP-SAT random seed.")
    parser.add_argument("--cp-use-decision-strategy", dest="cp_use_decision_strategy", action="store_true", default=True, help="Add an optional y-first CP decision strategy.")
    parser.add_argument("--cp-no-decision-strategy", dest="cp_use_decision_strategy", action="store_false", help="Disable the optional CP decision strategy.")
    parser.add_argument("--cp-use-service-no-overlap", dest="cp_use_service_no_overlap", action="store_true", default=False, help="Add optional service intervals and NoOverlap.")
    parser.add_argument("--cp-no-service-no-overlap", dest="cp_use_service_no_overlap", action="store_false", help="Disable service NoOverlap intervals.")
    parser.add_argument("--cp-candidate-strategy", choices=["urgent", "hybrid"], default="hybrid", help="Daily candidate filtering strategy for rolling CP.")
    parser.add_argument("--cp-phase1-only", dest="cp_solve_phase2", action="store_false", default=True, help="Run only phase 1 delivered-count CP for diagnostics.")
    parser.add_argument("--cp-solve-phase2", dest="cp_solve_phase2", action="store_true", help="Run phase 2 route-cost CP after phase 1.")
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

    cp_workers = args.cp_threads if args.cp_threads is not None else args.cp_workers
    cp_time_limit_per_day_sec = (
        args.cp_time_limit_per_day if args.cp_time_limit_per_day is not None else args.cp_time_limit_per_day_sec
    )
    solver = create_solver(
        args.solver,
        cp_time_limit_sec=args.cp_time_limit_sec,
        cp_time_limit_per_day_sec=cp_time_limit_per_day_sec,
        cp_max_customers=args.cp_max_customers,
        cp_max_candidates_per_day=args.cp_max_candidates_per_day,
        cp_workers=cp_workers,
        cp_log_search=args.cp_log_search,
        cp_two_phase_objective=args.cp_two_phase_objective,
        cp_random_seed=args.cp_random_seed,
        cp_use_decision_strategy=args.cp_use_decision_strategy,
        cp_use_service_no_overlap=args.cp_use_service_no_overlap,
        cp_candidate_strategy=args.cp_candidate_strategy,
        cp_solve_phase2=args.cp_solve_phase2,
        heuristic_max_candidates_per_day=args.heuristic_max_candidates_per_day,
        heuristic_random_seed=args.heuristic_random_seed,
        heuristic_use_local_search=args.heuristic_use_local_search,
        local_search_time_limit_sec=args.local_search_time_limit_sec,
        local_search_max_iterations=args.local_search_max_iterations,
        ga_population_size=args.ga_population_size,
        ga_generations=args.ga_generations,
        ga_elite_size=args.ga_elite_size,
        ga_mutation_rate=args.ga_mutation_rate,
        ga_crossover_rate=args.ga_crossover_rate,
        ga_time_limit_sec=args.ga_time_limit_sec,
        seed=args.seed,
    )
    schedule = solver.solve(instance)
    metrics = evaluate_weekly_schedule(instance, schedule)
    solver_status = solver_status_summary(schedule, metrics)

    print(f"solver={solver.name}")
    print(f"solver_status={solver_status.get('status', '')}")
    print(f"gap_percent={format_gap_percent(solver_status.get('gap_percent', ''))}")
    print_schedule(schedule)
    print_metrics(metrics)

    if args.save_results:
        results_dir = Path(args.results_dir)
        save_result_json(solver_results_dir(results_dir, solver.name) / "result.json", solver.name, schedule, metrics)
        export_report_files(results_dir, solver.name, instance, schedule, metrics)
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
