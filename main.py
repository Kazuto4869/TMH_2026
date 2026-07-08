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

from vrp_weekly.config import CP_TIME_LIMIT_PER_DAY_SEC  # noqa: E402
from vrp_weekly.evaluator import evaluate_weekly_schedule, print_metrics, print_schedule  # noqa: E402
from vrp_weekly.export import export_report_files, save_result_json, save_run_log_csv, solver_results_dir  # noqa: E402
from vrp_weekly.io import load_instance  # noqa: E402
from vrp_weekly.model_factory import create_solver, solver_names  # noqa: E402


DEFAULT_LOCATIONS = ROOT_DIR / "data" / "locations.csv"
DEFAULT_TIME_WINDOWS = ROOT_DIR / "data" / "time_windows.csv"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"


def main() -> int:
    """Run selected models from an interactive terminal menu."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    print("Weekly VRP model runner")
    print("=======================")
    locations_path = _prompt_path("Locations CSV", DEFAULT_LOCATIONS)
    time_windows_path = _prompt_path("Time windows CSV", DEFAULT_TIME_WINDOWS)
    results_dir = DEFAULT_RESULTS_DIR
    print(f"Results directory: {results_dir}")

    available_models = solver_names()
    selected_models = _prompt_models(available_models)
    cp_time_limit = _prompt_int("CP time limit per day in seconds", CP_TIME_LIMIT_PER_DAY_SEC, minimum=1)
    cp_threads = 1
    cp_log_search = False
    if "cp" in selected_models:
        cp_threads = _prompt_int("CP threads", 1, minimum=1)
        cp_log_search = _prompt_yes_no("Show CP run log in terminal", default=True)

    instance = load_instance(locations_path, time_windows_path)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name in selected_models:
        print()
        print(f"Running model: {model_name}")
        start_time = time.perf_counter()
        model = create_solver(
            model_name,
            cp_time_limit_per_day=cp_time_limit,
            cp_threads=cp_threads,
            cp_log_search=cp_log_search,
        )
        schedule = model.solve(instance)
        runtime_sec = time.perf_counter() - start_time
        metrics = evaluate_weekly_schedule(instance, schedule)

        print(f"model={model.name}")
        print(f"runtime_sec={runtime_sec:.3f}")
        print_schedule(schedule)
        print_metrics(metrics)

        output_dir = solver_results_dir(results_dir, model.name)
        save_result_json(output_dir / "result.json", model.name, schedule, metrics)
        export_report_files(results_dir, model.name, instance, schedule)
        run_log_path = output_dir / f"run_log_{model.name}_{run_timestamp}.csv"
        save_run_log_csv(
            run_log_path,
            _build_run_log_row(
                model_name=model.name,
                timestamp=run_timestamp,
                runtime_sec=runtime_sec,
                metrics=metrics.to_dict(),
                locations_path=locations_path,
                time_windows_path=time_windows_path,
                cp_time_limit=cp_time_limit,
                cp_threads=cp_threads if model.name == "cp" else "",
                cp_log_search=cp_log_search if model.name == "cp" else "",
            ),
        )
        print(f"saved_results={output_dir}")
        print(f"run_log={run_log_path}")

    print()
    print("Done.")
    return 0


def _prompt_path(label: str, default: Path) -> Path:
    """Prompt for a path with a default."""
    raw_value = input(f"{label} [{default}]: ").strip()
    return Path(raw_value) if raw_value else default


def _prompt_models(available_models: list[str]) -> list[str]:
    """Prompt for one or more model names."""
    print()
    print("Available models:")
    for index, model_name in enumerate(available_models, start=1):
        print(f"{index}. {model_name}")
    print("Type a number, model name, comma-separated list, or 'all'.")

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
    metrics: dict[str, object],
    locations_path: Path,
    time_windows_path: Path,
    cp_time_limit: int,
    cp_threads: int | str,
    cp_log_search: bool | str,
) -> dict[str, object]:
    """Build one CSV row describing a completed model run."""
    row: dict[str, object] = {
        "timestamp": timestamp,
        "model": model_name,
        "runtime_sec": f"{runtime_sec:.6f}",
        "locations_path": str(locations_path),
        "time_windows_path": str(time_windows_path),
        "cp_time_limit_per_day": cp_time_limit,
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
