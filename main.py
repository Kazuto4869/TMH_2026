"""Interactive runner for weekly VRP models."""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vrp_weekly.evaluator import evaluate_weekly_schedule, print_metrics, print_schedule  # noqa: E402
from vrp_weekly.export import (  # noqa: E402
    export_report_files,
    format_gap_percent,
    save_result_json,
    save_run_log_csv,
    solver_results_dir,
    solver_status_summary,
)
from vrp_weekly.io import load_instance  # noqa: E402
from vrp_weekly.model_factory import create_solver, solver_names  # noqa: E402


DEFAULT_LOCATIONS = ROOT_DIR / "data" / "locations.csv"
DEFAULT_TIME_WINDOWS = ROOT_DIR / "data" / "time_windows.csv"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"


def main() -> int:
    """Run selected models from an interactive terminal menu."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stdout,
        force=True,
    )

    print("Weekly VRP model runner", flush=True)
    print("=======================", flush=True)
    locations_path = DEFAULT_LOCATIONS
    time_windows_path = DEFAULT_TIME_WINDOWS
    results_dir = DEFAULT_RESULTS_DIR

    available_models = solver_names()
    _ensure_result_dirs(results_dir, available_models)
    selected_models = _prompt_models(available_models)
    has_cp_full_week = "cp_full_week" in selected_models
    has_cp_rolling = "cp_rolling" in selected_models
    has_cp_model = has_cp_full_week or has_cp_rolling
    cp_time_limit_sec = 60
    cp_time_limit_per_day_sec = 10
    cp_max_customers: int | None = None
    cp_max_candidates_per_day: int | None = None
    cp_threads = 8
    cp_log_search = False
    cp_two_phase_objective = True
    cp_random_seed = 1
    cp_use_decision_strategy = True
    cp_use_service_no_overlap = True
    cp_candidate_strategy = "hybrid"
    cp_solve_phase2 = True
    heuristic_max_candidates_per_day: int | None = None
    heuristic_random_seed = 1
    heuristic_use_local_search = False
    local_search_time_limit_sec = 10
    local_search_max_iterations = 100
    ga_population_size = 30
    ga_generations = 50
    ga_elite_size = 5
    ga_mutation_rate = 0.10
    ga_crossover_rate = 0.80
    ga_time_limit_sec = 120
    has_ls_model = any(model_name.endswith("_ls") or model_name == "hybrid_genetic_vns" for model_name in selected_models)
    if has_cp_full_week:
        cp_time_limit_sec = _prompt_int("Full-week CP time limit in seconds", 60, minimum=1)
        cp_max_customers = _prompt_optional_int("Full-week CP max customers", default=40, minimum=1)
    if has_cp_rolling:
        cp_time_limit_per_day_sec = _prompt_int("Rolling CP time limit per day in seconds", 10, minimum=1)
        cp_max_candidates_per_day = _prompt_optional_int("Rolling CP max candidates per day", default=80, minimum=1)
        cp_two_phase_objective = _prompt_yes_no("Use pure two-phase rolling CP objective", default=True)
        cp_random_seed = _prompt_int("CP random seed", 1, minimum=0)
        cp_use_decision_strategy = _prompt_yes_no("Use optional y-first CP decision strategy", default=True)
        cp_use_service_no_overlap = _prompt_yes_no("Use service NoOverlap intervals", default=True)
        cp_candidate_strategy = _prompt_choice("Rolling CP candidate strategy", ["hybrid", "urgent"], default="hybrid")
        cp_solve_phase2 = not _prompt_yes_no("Run phase 1 only", default=False)
    if has_cp_model:
        cp_threads = _prompt_int("CP workers", 8, minimum=1)
        cp_log_search = _prompt_yes_no("Show CP-SAT run log in terminal", default=False)
    if any(model_name in {"inferior_insertion", "inferior_insertion_ls", "regret_dispatch", "regret_dispatch_ls"} for model_name in selected_models):
        heuristic_max_candidates_per_day = _prompt_optional_int("Heuristic max candidates per day", default=0, minimum=1)
        if heuristic_max_candidates_per_day == 0:
            heuristic_max_candidates_per_day = None
        heuristic_random_seed = _prompt_int("Heuristic random seed", 1, minimum=0)
    if has_ls_model:
        heuristic_use_local_search = True
        local_search_time_limit_sec = _prompt_int("Local search time limit per route in seconds", 10, minimum=1)
        local_search_max_iterations = _prompt_int("Local search max iterations per route", 100, minimum=1)
    if "hybrid_genetic_vns" in selected_models:
        ga_population_size = _prompt_int("Hybrid GA population size", 30, minimum=1)
        ga_generations = _prompt_int("Hybrid GA generations", 50, minimum=1)
        ga_elite_size = _prompt_int("Hybrid GA elite size", 5, minimum=1)
        ga_mutation_rate = float(_prompt_int("Hybrid GA mutation rate percent", 10, minimum=0)) / 100.0
        ga_crossover_rate = float(_prompt_int("Hybrid GA crossover rate percent", 80, minimum=0)) / 100.0
        ga_time_limit_sec = _prompt_int("Hybrid GA time limit in seconds", 120, minimum=1)

    instance = load_instance(locations_path, time_windows_path)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name in selected_models:
        print(flush=True)
        print(f"Running model: {model_name}", flush=True)
        start_time = time.perf_counter()
        model = create_solver(
            model_name,
            cp_time_limit_sec=cp_time_limit_sec,
            cp_time_limit_per_day_sec=cp_time_limit_per_day_sec,
            cp_max_customers=cp_max_customers,
            cp_max_candidates_per_day=cp_max_candidates_per_day,
            cp_workers=cp_threads,
            cp_log_search=cp_log_search,
            cp_two_phase_objective=cp_two_phase_objective,
            cp_random_seed=cp_random_seed,
            cp_use_decision_strategy=cp_use_decision_strategy,
            cp_use_service_no_overlap=cp_use_service_no_overlap,
            cp_candidate_strategy=cp_candidate_strategy,
            cp_solve_phase2=cp_solve_phase2,
            heuristic_max_candidates_per_day=heuristic_max_candidates_per_day,
            heuristic_random_seed=heuristic_random_seed,
            heuristic_use_local_search=heuristic_use_local_search,
            local_search_time_limit_sec=local_search_time_limit_sec,
            local_search_max_iterations=local_search_max_iterations,
            ga_population_size=ga_population_size,
            ga_generations=ga_generations,
            ga_elite_size=ga_elite_size,
            ga_mutation_rate=ga_mutation_rate,
            ga_crossover_rate=ga_crossover_rate,
            ga_time_limit_sec=ga_time_limit_sec,
        )
        schedule = model.solve(instance)
        runtime_sec = time.perf_counter() - start_time
        metrics = evaluate_weekly_schedule(instance, schedule)
        solver_status = solver_status_summary(schedule, metrics)

        print(f"model={model.name}", flush=True)
        print(f"runtime_sec={runtime_sec:.3f}", flush=True)
        print(f"solver_status={solver_status.get('status', '')}", flush=True)
        print(f"gap_percent={format_gap_percent(solver_status.get('gap_percent', ''))}", flush=True)
        print_schedule(schedule)
        print_metrics(metrics)

        output_dir = solver_results_dir(results_dir, model.name)
        result_path = output_dir / "result.json"
        result_txt_path = output_dir / "result.txt"
        daily_schedule_path = output_dir / "daily_schedule.csv"
        incomplete_orders_path = output_dir / "incomplete_orders.csv"
        save_result_json(result_path, model.name, schedule, metrics)
        export_report_files(results_dir, model.name, instance, schedule, metrics)
        run_log_path = output_dir / f"run_log_{model.name}_{run_timestamp}.csv"
        if model.name == "cp_rolling":
            save_run_log_csv(
                run_log_path,
                _build_rolling_run_log_rows(
                    timestamp=run_timestamp,
                    runtime_sec=runtime_sec,
                    solver_status=solver_status,
                    metrics=metrics.to_dict(),
                    cp_time_limit_per_day_sec=cp_time_limit_per_day_sec,
                    cp_max_candidates_per_day=cp_max_candidates_per_day,
                    cp_threads=cp_threads,
                    cp_log_search=cp_log_search,
                    cp_two_phase_objective=cp_two_phase_objective,
                    cp_random_seed=cp_random_seed,
                    cp_use_decision_strategy=cp_use_decision_strategy,
                    cp_use_service_no_overlap=cp_use_service_no_overlap,
                    cp_candidate_strategy=cp_candidate_strategy,
                    cp_solve_phase2=cp_solve_phase2,
                ),
            )
        else:
            save_run_log_csv(
                run_log_path,
                _build_run_log_row(
                    model_name=model.name,
                    timestamp=run_timestamp,
                    runtime_sec=runtime_sec,
                    solver_status=solver_status,
                    metrics=metrics.to_dict(),
                    cp_time_limit_sec=cp_time_limit_sec if model.name == "cp_full_week" else "",
                    cp_time_limit_per_day_sec="",
                    cp_max_customers=cp_max_customers if model.name == "cp_full_week" else "",
                    cp_max_candidates_per_day="",
                    cp_threads=cp_threads if model.name in ("cp_full_week", "cp_rolling") else "",
                    cp_log_search=cp_log_search if model.name in ("cp_full_week", "cp_rolling") else "",
                ),
            )
        print(f"saved_results={model.name}", flush=True)
        print("updated_files=result.json,result.txt,daily_schedule.csv,incomplete_orders.csv", flush=True)
        print(f"run_log={run_log_path.name}", flush=True)

    print(flush=True)
    print("Done.", flush=True)
    return 0


def _prompt_path(label: str, default: Path) -> Path:
    """Prompt for a path with a default."""
    raw_value = input(f"{label} [{default}]: ").strip()
    return Path(raw_value) if raw_value else default


def _ensure_result_dirs(results_dir: Path, model_names: list[str]) -> None:
    """Create result folders for all registered models."""
    for model_name in model_names:
        solver_results_dir(results_dir, model_name).mkdir(parents=True, exist_ok=True)


def _prompt_models(available_models: list[str]) -> list[str]:
    """Prompt for one or more model names."""
    print(flush=True)
    print("Available models:", flush=True)
    for index, model_name in enumerate(available_models, start=1):
        print(f"{index}. {model_name}", flush=True)
    print("Type a number, model name, comma-separated list, or 'all'.", flush=True)

    while True:
        raw_value = input("Models to run [all]: ").strip().lower()
        if raw_value in ("", "all"):
            return available_models

        selected: list[str] = []
        invalid: list[str] = []
        for token in [part.strip() for part in raw_value.split(",") if part.strip()]:
            if token.isdigit():
                index = int(token)
                if 1 <= index <= len(available_models):
                    selected.append(available_models[index - 1])
                else:
                    invalid.append(token)
            elif token in available_models:
                selected.append(token)
            else:
                invalid.append(token)

        if selected and not invalid:
            return list(dict.fromkeys(selected))
        print(f"Invalid model selection: {', '.join(invalid) if invalid else raw_value}")


def _prompt_int(label: str, default: int, minimum: int | None = None) -> int:
    """Prompt for an integer with validation."""
    while True:
        raw_value = input(f"{label} [{default}]: ").strip()
        if not raw_value:
            return default
        try:
            value = int(raw_value)
        except ValueError:
            print("Please enter an integer.")
            continue
        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def _prompt_optional_int(label: str, default: int, minimum: int | None = None) -> int | None:
    """Prompt for an optional integer; blank keeps the recommended default."""
    while True:
        raw_value = input(f"{label} [{default}, blank uses default, 0 disables]: ").strip()
        if not raw_value:
            return default
        try:
            value = int(raw_value)
        except ValueError:
            print("Please enter an integer.")
            continue
        if value == 0:
            return None
        if minimum is not None and value < minimum:
            print(f"Please enter 0 or a value >= {minimum}.")
            continue
        return value


def _prompt_yes_no(label: str, default: bool = False) -> bool:
    """Prompt for a yes/no option."""
    default_text = "Y/n" if default else "y/N"
    while True:
        raw_value = input(f"{label} [{default_text}]: ").strip().lower()
        if not raw_value:
            return default
        if raw_value in ("y", "yes"):
            return True
        if raw_value in ("n", "no"):
            return False
        print("Please answer y or n.")


def _prompt_choice(label: str, choices: list[str], default: str) -> str:
    """Prompt for one value from a small option set."""
    choices_text = "/".join(choices)
    while True:
        raw_value = input(f"{label} [{default}; {choices_text}]: ").strip().lower()
        if not raw_value:
            return default
        if raw_value in choices:
            return raw_value
        print(f"Please enter one of: {', '.join(choices)}")


def _build_run_log_row(
    model_name: str,
    timestamp: str,
    runtime_sec: float,
    solver_status: dict[str, object],
    metrics: dict[str, object],
    cp_time_limit_sec: int | str,
    cp_time_limit_per_day_sec: int | str,
    cp_max_customers: int | str | None,
    cp_max_candidates_per_day: int | str | None,
    cp_threads: int | str,
    cp_log_search: bool | str,
    row_type: str = "SUMMARY",
    day: int | str = "",
) -> dict[str, object]:
    """Build one CSV row describing a completed model run."""
    row: dict[str, object] = {
        "timestamp": timestamp,
        "model": model_name,
        "row_type": row_type,
        "day": day,
        "runtime_sec": f"{runtime_sec:.6f}",
        "solver_status": solver_status.get("status", ""),
        "gap_percent": format_gap_percent(solver_status.get("gap_percent", "")),
        "cp_time_limit_sec": cp_time_limit_sec,
        "cp_time_limit_per_day_sec": cp_time_limit_per_day_sec,
        "cp_max_customers": "" if cp_max_customers is None else cp_max_customers,
        "cp_max_candidates_per_day": "" if cp_max_candidates_per_day is None else cp_max_candidates_per_day,
        "cp_threads": cp_threads,
        "cp_log_search": cp_log_search,
    }
    for key, value in metrics.items():
        if key == "violations":
            row["violation_count"] = len(value) if isinstance(value, list) else 0
        else:
            row[key] = value
    return row


def _build_rolling_run_log_rows(
    timestamp: str,
    runtime_sec: float,
    solver_status: dict[str, object],
    metrics: dict[str, object],
    cp_time_limit_per_day_sec: int | str,
    cp_max_candidates_per_day: int | str | None,
    cp_threads: int | str,
    cp_log_search: bool | str,
    cp_two_phase_objective: bool | str,
    cp_random_seed: int | str,
    cp_use_decision_strategy: bool | str,
    cp_use_service_no_overlap: bool | str,
    cp_candidate_strategy: str,
    cp_solve_phase2: bool | str,
) -> list[dict[str, object]]:
    """Build day-level rows plus a summary row for rolling-horizon CP."""
    rows: list[dict[str, object]] = []
    day_statuses = solver_status.get("day_statuses", {})
    if isinstance(day_statuses, dict):
        for day in range(1, 8):
            day_status = day_statuses.get(day, {})
            if not isinstance(day_status, dict):
                day_status = {}
            rows.append(
                {
                    "timestamp": timestamp,
                    "model": "cp_rolling",
                    "row_type": "DAY",
                    "day": day,
                    "runtime_sec": f"{float(day_status.get('runtime_sec', 0.0)):.6f}",
                    "solver_status": day_status.get("status", ""),
                    "gap_percent": format_gap_percent(day_status.get("gap_percent", "")),
                    "objective": day_status.get("objective", ""),
                    "best_bound": day_status.get("best_bound", ""),
                    "complete_count": day_status.get("complete_count", ""),
                    "carried_over_count": day_status.get("carried_over_count", ""),
                    "cp_time_limit_sec": "",
                    "cp_time_limit_per_day_sec": cp_time_limit_per_day_sec,
                    "cp_max_customers": "",
                    "cp_max_candidates_per_day": "" if cp_max_candidates_per_day is None else cp_max_candidates_per_day,
                    "cp_threads": cp_threads,
                    "cp_log_search": cp_log_search,
                    "cp_two_phase_objective": cp_two_phase_objective,
                    "cp_random_seed": cp_random_seed,
                    "cp_use_decision_strategy": cp_use_decision_strategy,
                    "cp_use_service_no_overlap": cp_use_service_no_overlap,
                    "cp_candidate_strategy": cp_candidate_strategy,
                    "cp_solve_phase2": cp_solve_phase2,
                    "objective_mode": day_status.get("objective_mode", ""),
                    "daily_optimal_for": day_status.get("daily_optimal_for", ""),
                    "fixed_impossible_arcs": day_status.get("fixed_impossible_arcs", ""),
                    "total_nonself_arcs": day_status.get("total_nonself_arcs", ""),
                    "fixed_arc_ratio": day_status.get("fixed_arc_ratio", ""),
                    "distance_objective_scale": day_status.get("distance_objective_scale", ""),
                    "drop_penalty": day_status.get("drop_penalty", ""),
                    "distance_weight": day_status.get("distance_weight", ""),
                    "degree_linking_constraints_count": day_status.get("degree_linking_constraints_count", ""),
                    "arc_linking_constraints_count": day_status.get("arc_linking_constraints_count", ""),
                    "window_pair_cuts_count": day_status.get("window_pair_cuts_count", ""),
                    "pair_conflict_cuts_count": day_status.get("pair_conflict_cuts_count", ""),
                    "depot_window_cuts_count": day_status.get("depot_window_cuts_count", ""),
                    "dominated_window_cuts_count": day_status.get("dominated_window_cuts_count", ""),
                    "precedence_cuts_count": day_status.get("precedence_cuts_count", ""),
                    "phase1_status": day_status.get("phase1_status", ""),
                    "phase2_status": day_status.get("phase2_status", ""),
                    "service_interval_count": day_status.get("service_interval_count", ""),
                    "roundtrip_duration_lb_count": day_status.get("roundtrip_duration_lb_count", ""),
                    "fixed_impossible_customers_count": day_status.get("fixed_impossible_customers_count", ""),
                    "candidate_strategy": day_status.get("candidate_strategy", ""),
                    "delivered_count": "",
                    "incomplete_count": "",
                    "active_days": "",
                    "total_deferral_days": "",
                    "total_distance_km": "",
                    "total_travel_time_min": "",
                    "total_waiting_time_min": "",
                    "total_service_time_min": "",
                    "total_route_duration_min": "",
                    "objective_value": "",
                    "violation_count": "",
                }
            )

    summary_row = _build_run_log_row(
        model_name="cp_rolling",
        timestamp=timestamp,
        runtime_sec=runtime_sec,
        solver_status=solver_status,
        metrics=metrics,
        cp_time_limit_sec="",
        cp_time_limit_per_day_sec=cp_time_limit_per_day_sec,
        cp_max_customers="",
        cp_max_candidates_per_day=cp_max_candidates_per_day,
        cp_threads=cp_threads,
        cp_log_search=cp_log_search,
    )
    summary_row["row_type"] = "SUMMARY"
    summary_row["day"] = "SUMMARY"
    summary_row["complete_count"] = metrics.get("delivered_count", "")
    summary_row["carried_over_count"] = metrics.get("incomplete_count", "")
    summary_row["cp_two_phase_objective"] = cp_two_phase_objective
    summary_row["cp_random_seed"] = cp_random_seed
    summary_row["cp_use_decision_strategy"] = cp_use_decision_strategy
    summary_row["cp_use_service_no_overlap"] = cp_use_service_no_overlap
    summary_row["cp_candidate_strategy"] = cp_candidate_strategy
    summary_row["cp_solve_phase2"] = cp_solve_phase2
    rows.append(summary_row)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
