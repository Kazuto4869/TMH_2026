"""Diagnostic utilities for rolling CP-SAT profiles."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

from vrp_weekly.evaluator import evaluate_weekly_schedule
from vrp_weekly.io import load_instance
from vrp_weekly.model_factory import create_solver


def run_candidate_cap_ablation(
    locations_path: str | Path,
    time_windows_path: str | Path,
    results_dir: str | Path = "results",
    workers: int = 4,
    random_seed: int = 1,
) -> tuple[Path, Path]:
    """Run 60s cap profiles, then the two best profiles at 300s/day."""
    instance = load_instance(locations_path, time_windows_path)
    output_dir = Path(results_dir) / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    cap_60_path = output_dir / "cp_rolling_candidate_cap_60s.csv"
    cap_300_path = output_dir / "cp_rolling_candidate_cap_300s.csv"
    profiles = [("cap_80", 80), ("cap_100", 100), ("cap_120", 120), ("cap_none", None)]
    rows_60 = [
        _run_cap_profile(instance, profile, cap, 60, workers, random_seed)
        for profile, cap in profiles
    ]
    _write_rows(cap_60_path, rows_60)
    best_two = sorted(
        rows_60,
        key=lambda row: (
            int(row["incomplete_count"]),
            int(row["total_deferral_days"]),
            float(row["total_distance_km"]),
            float(row["runtime_sec"]),
        ),
    )[:2]
    rows_300 = [
        _run_cap_profile(instance, str(row["profile"]).replace("cap_", "cap_300_"), _parse_cap(row), 300, workers, random_seed)
        for row in best_two
    ]
    _write_rows(cap_300_path, rows_300)
    return cap_60_path, cap_300_path


def run_repair_comparison_60s(
    locations_path: str | Path,
    time_windows_path: str | Path,
    results_dir: str | Path = "results",
    max_candidates_per_day: int | None = 80,
    workers: int = 4,
    random_seed: int = 1,
    repair_time_limit_sec: int = 300,
    repair_max_days: int = 2,
    repair_max_customers: int = 120,
) -> Path:
    """Compare cp_rolling and cp_rolling_repair at 60 seconds/day."""
    instance = load_instance(locations_path, time_windows_path)
    output_path = Path(results_dir) / "comparison" / "cp_rolling_repair_60s.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiles = [
        (
            "cp_rolling",
            {
                "cp_time_limit_per_day_sec": 60,
                "cp_max_candidates_per_day": max_candidates_per_day,
                "cp_workers": workers,
                "cp_random_seed": random_seed,
                "cp_optimization_mode": "full_three_stage",
                "cp_adaptive_daily_deadline": True,
            },
        ),
        (
            "cp_rolling_repair",
            {
                "cp_time_limit_per_day_sec": 60,
                "cp_max_candidates_per_day": max_candidates_per_day,
                "cp_workers": workers,
                "cp_random_seed": random_seed,
                "cp_optimization_mode": "full_three_stage",
                "cp_adaptive_daily_deadline": True,
                "cp_repair_time_limit_sec": repair_time_limit_sec,
                "cp_repair_max_days": repair_max_days,
                "cp_repair_max_customers": repair_max_customers,
            },
        ),
    ]
    rows: list[dict[str, Any]] = []
    for solver_name, kwargs in profiles:
        solver = create_solver(solver_name, **kwargs)
        start = time.perf_counter()
        schedule = solver.solve(instance)
        runtime_sec = time.perf_counter() - start
        metrics = evaluate_weekly_schedule(instance, schedule)
        incomplete_ids = sorted(set(instance.customer_ids()) - schedule.delivered_customer_ids())
        rows.append(
            {
                "solver": solver.name,
                "delivered_count": metrics.delivered_count,
                "incomplete_count": metrics.incomplete_count,
                "incomplete_customer_ids": " ".join(incomplete_ids),
                "total_deferral_days": metrics.total_deferral_days,
                "total_distance_km": f"{metrics.total_distance_km:.6f}",
                "total_waiting_time_min": metrics.total_waiting_time_min,
                "total_route_duration_min": metrics.total_route_duration_min,
                "runtime_sec": f"{runtime_sec:.6f}",
                "hard_feasible": metrics.hard_feasible,
                "repair_ran": schedule.solver_status.get("repair_ran", ""),
                "repair_accepted": schedule.solver_status.get("repair_accepted", ""),
                "repair_selected_days": " ".join(map(str, schedule.solver_status.get("repair_selected_days", []))),
                "repair_rescued_customer_ids": " ".join(schedule.solver_status.get("repair_rescued_customer_ids", [])),
                "repair_stage_r1_status": schedule.solver_status.get("repair_stage_r1_status", ""),
                "repair_stage_r2_status": schedule.solver_status.get("repair_stage_r2_status", ""),
                "repair_stage_r3_status": schedule.solver_status.get("repair_stage_r3_status", ""),
                "repair_total_runtime_sec": schedule.solver_status.get("repair_total_runtime_sec", ""),
            }
        )
    _write_rows(output_path, rows)
    return output_path


def _run_cap_profile(
    instance: Any,
    profile: str,
    max_candidates_per_day: int | None,
    time_limit_per_day_sec: int,
    workers: int,
    random_seed: int,
) -> dict[str, Any]:
    solver = create_solver(
        "cp_rolling",
        cp_time_limit_per_day_sec=time_limit_per_day_sec,
        cp_max_candidates_per_day=max_candidates_per_day,
        cp_workers=workers,
        cp_random_seed=random_seed,
        cp_optimization_mode="full_three_stage",
        cp_adaptive_daily_deadline=True,
        cp_stage2_max_time_fraction=0.10,
    )
    start = time.perf_counter()
    schedule = solver.solve(instance)
    runtime_sec = time.perf_counter() - start
    metrics = evaluate_weekly_schedule(instance, schedule)
    incomplete_ids = sorted(set(instance.customer_ids()) - schedule.delivered_customer_ids())
    day_statuses = schedule.solver_status.get("day_statuses", {})
    if not isinstance(day_statuses, dict):
        day_statuses = {}
    return {
        "profile": profile,
        "max_candidates_per_day": "" if max_candidates_per_day is None else max_candidates_per_day,
        "delivered_count": metrics.delivered_count,
        "incomplete_count": metrics.incomplete_count,
        "incomplete_customer_ids": " ".join(incomplete_ids),
        "total_deferral_days": metrics.total_deferral_days,
        "total_distance_km": f"{metrics.total_distance_km:.6f}",
        "total_waiting_time_min": metrics.total_waiting_time_min,
        "total_route_duration_min": metrics.total_route_duration_min,
        "runtime_sec": f"{runtime_sec:.6f}",
        "hard_feasible": metrics.hard_feasible,
        "stage1a_optimal_days": _count_day_status(day_statuses, "stage1a_status", "OPTIMAL"),
        "stage1b_optimal_days": _count_day_status(day_statuses, "stage1b_status", "OPTIMAL"),
        "stage2_optimal_days": _count_day_status(day_statuses, "stage2_status", "OPTIMAL"),
        "stage1b_skipped_days": _count_truthy_reason(day_statuses, "stage1b_skipped_reason"),
        "stage2_skipped_days": _count_truthy_reason(day_statuses, "stage2_skipped_reason"),
        "mandatory_unserved_total": sum(
            int(status.get("mandatory_unserved_count", 0) or 0)
            for status in day_statuses.values()
            if isinstance(status, dict)
        ),
        "filtered_mandatory_total": sum(
            len(set(status.get("filtered_candidate_ids", [])) & set(status.get("mandatory_candidate_ids", [])))
            for status in day_statuses.values()
            if isinstance(status, dict)
        ),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_cap(row: dict[str, Any]) -> int | None:
    raw = row.get("max_candidates_per_day", "")
    return None if raw in ("", None) else int(raw)


def _count_day_status(day_statuses: dict[Any, Any], field: str, value: str) -> int:
    return sum(1 for status in day_statuses.values() if isinstance(status, dict) and status.get(field) == value)


def _count_truthy_reason(day_statuses: dict[Any, Any], field: str) -> int:
    return sum(1 for status in day_statuses.values() if isinstance(status, dict) and bool(status.get(field)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rolling CP diagnostics.")
    parser.add_argument("--locations", default="data/locations.csv", help="Path to locations.csv")
    parser.add_argument("--time-windows", default="data/time_windows.csv", help="Path to time_windows.csv")
    parser.add_argument("--results-dir", default="results", help="Directory for diagnostic outputs")
    parser.add_argument("--workers", type=int, default=4, help="CP worker count")
    parser.add_argument("--seed", type=int, default=1, help="CP random seed")
    parser.add_argument("--repair-comparison-60s", action="store_true", help="Run cp_rolling vs cp_rolling_repair 60s comparison.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repair_comparison_60s:
        repair_path = run_repair_comparison_60s(
            args.locations,
            args.time_windows,
            args.results_dir,
            workers=args.workers,
            random_seed=args.seed,
        )
        print(f"repair_comparison_60s={repair_path}")
    else:
        cap_60_path, cap_300_path = run_candidate_cap_ablation(
            args.locations,
            args.time_windows,
            args.results_dir,
            workers=args.workers,
            random_seed=args.seed,
        )
        print(f"candidate_cap_60s={cap_60_path}")
        print(f"candidate_cap_300s={cap_300_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
