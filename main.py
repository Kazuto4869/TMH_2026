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
    if has_cp_full_week:
        cp_time_limit_sec = _prompt_int("Full-week CP time limit in seconds", 60, minimum=1)
        cp_max_customers = _prompt_optional_int("Full-week CP max customers", default=300, minimum=1)
    if has_cp_rolling:
        cp_time_limit_per_day_sec = _prompt_int("Rolling CP time limit per day in seconds", 10, minimum=1)
        cp_max_candidates_per_day = _prompt_optional_int("Rolling CP max candidates per day", default=80, minimum=1)
    if has_cp_model:
        cp_threads = _prompt_int("CP workers", 8, minimum=1)
        cp_log_search = _prompt_yes_no("Show CP-SAT run log in terminal", default=False)

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
        save_run_log_csv(
            run_log_path,
            _build_run_log_row(
                model_name=model.name,
                timestamp=run_timestamp,
                runtime_sec=runtime_sec,
                solver_status=solver_status,
                metrics=metrics.to_dict(),
                cp_time_limit_sec=cp_time_limit_sec if model.name == "cp_full_week" else "",
                cp_time_limit_per_day_sec=cp_time_limit_per_day_sec if model.name == "cp_rolling" else "",
                cp_max_customers=cp_max_customers if model.name == "cp_full_week" else "",
                cp_max_candidates_per_day=cp_max_candidates_per_day if model.name == "cp_rolling" else "",
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
) -> dict[str, object]:
    """Build one CSV row describing a completed model run."""
    row: dict[str, object] = {
        "timestamp": timestamp,
        "model": model_name,
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


if __name__ == "__main__":
    raise SystemExit(main())
