"""JSON, CSV, and plot exports for schedules and benchmark results."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from vrp_weekly.core import EvaluationMetrics, Instance, WeeklySchedule
from vrp_weekly.config import OBJECTIVE_VERSION
from vrp_weekly.evaluator import objective_breakdown
from vrp_weekly.time_utils import format_hhmm


def solver_status_summary(schedule: WeeklySchedule, metrics: EvaluationMetrics) -> dict[str, object]:
    """Return solver status suitable for display and exports."""
    status = dict(schedule.solver_status) if schedule.solver_status else {
        "status": "HEURISTIC_FEASIBLE" if metrics.hard_feasible else "HEURISTIC_INFEASIBLE",
        "gap_percent": "",
    }
    breakdown = objective_breakdown(metrics)
    status.update(
        objective_version=OBJECTIVE_VERSION,
        objective_value=metrics.objective_value,
        objective_breakdown=breakdown,
        **breakdown,
    )
    return status


def format_gap_percent(value: object) -> str:
    """Format a gap value as a percentage string."""
    if value in ("", None):
        return ""
    return f"{float(value):.2f}%"


def solver_results_dir(results_dir: str | Path, solver_name: str) -> Path:
    """Return the canonical output directory for one solver."""
    return Path(results_dir) / "schedules" / solver_name


def schedule_to_dict(schedule: WeeklySchedule) -> dict[str, Any]:
    """Convert a weekly schedule to a JSON-serializable dictionary."""
    return {
        "routes": [
            {
                "day": route.day,
                "return_to_depot_time": route.return_to_depot_time,
                "route_distance_km": route.route_distance_km,
                "route_travel_time_min": route.route_travel_time_min,
                "route_waiting_time_min": route.route_waiting_time_min,
                "route_service_time_min": route.route_service_time_min,
                "route_duration_min": route.route_duration_min,
                "hard_feasible": route.hard_feasible,
                "violations": route.violations,
                "stops": [
                    {
                        "customer_id": stop.customer_id,
                        "arrival_time": stop.arrival_time,
                        "service_start_time": stop.service_start_time,
                        "service_end_time": stop.service_end_time,
                        "selected_window_start": stop.selected_time_window.start_minute
                        if stop.selected_time_window
                        else None,
                        "selected_window_end": stop.selected_time_window.end_minute if stop.selected_time_window else None,
                        "travel_from_previous_min": stop.travel_from_previous_min,
                        "waiting_min": stop.waiting_min,
                        "distance_from_previous_km": stop.distance_from_previous_km,
                        "hard_feasible": stop.hard_feasible,
                        "violation": stop.violation,
                    }
                    for stop in route.stops
                ],
            }
            for route in schedule.ordered_routes()
        ]
    }


def save_result_json(
    path: str | Path,
    solver_name: str,
    schedule: WeeklySchedule,
    metrics: EvaluationMetrics,
    runtime_sec: float | None = None,
) -> None:
    """Save one solver result as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = metrics.to_dict()
    status = solver_status_summary(schedule, metrics)
    saved_runtime = runtime_sec if runtime_sec is not None else status.get("runtime_sec", "")
    metrics_payload.update(
        objective_version=OBJECTIVE_VERSION,
        objective_breakdown=objective_breakdown(metrics),
        runtime_sec=saved_runtime,
    )
    payload = {
        "solver": solver_name,
        "objective_version": OBJECTIVE_VERSION,
        "objective_value": metrics.objective_value,
        "objective_breakdown": objective_breakdown(metrics),
        "runtime_sec": saved_runtime,
        "solver_status": status,
        "metrics": metrics_payload,
        "schedule": schedule_to_dict(schedule),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_run_log_csv(path: str | Path, row: dict[str, Any] | list[dict[str, Any]]) -> None:
    """Save one or more run summary rows as a CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row] if isinstance(row, dict) else list(row)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = list(rows[0].keys())
    for current_row in rows[1:]:
        for key in current_row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for current_row in rows:
            writer.writerow(current_row)


def export_daily_schedule_csv(
    path: str | Path,
    instance: Instance,
    schedule: WeeklySchedule,
) -> None:
    """Export a report-ready daily schedule CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "day",
                "stop_order",
                "customer_id",
                "customer_name",
                "arrival_time",
                "service_start_time",
                "service_end_time",
                "selected_window_start",
                "selected_window_end",
                "travel_from_previous_min",
                "waiting_min",
                "distance_from_previous_km",
            ],
        )
        writer.writeheader()
        for route in schedule.ordered_routes():
            for index, stop in enumerate(route.stops, start=1):
                location = instance.locations[stop.customer_id]
                writer.writerow(
                    {
                        "day": route.day,
                        "stop_order": index,
                        "customer_id": stop.customer_id,
                        "customer_name": location.location_name,
                        "arrival_time": _format_or_blank(stop.arrival_time),
                        "service_start_time": _format_or_blank(stop.service_start_time),
                        "service_end_time": _format_or_blank(stop.service_end_time),
                        "selected_window_start": _format_or_blank(
                            stop.selected_time_window.start_minute if stop.selected_time_window else None
                        ),
                        "selected_window_end": _format_or_blank(
                            stop.selected_time_window.end_minute if stop.selected_time_window else None
                        ),
                        "travel_from_previous_min": stop.travel_from_previous_min,
                        "waiting_min": stop.waiting_min,
                        "distance_from_previous_km": f"{stop.distance_from_previous_km:.6f}",
                    }
                )


def export_incomplete_orders_csv(path: str | Path, instance: Instance, schedule: WeeklySchedule) -> None:
    """Export customers not delivered in the weekly schedule."""
    delivered = schedule.delivered_customer_ids()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["customer_id", "customer_name", "demand_kg", "available_days"],
        )
        writer.writeheader()
        for customer_id in instance.customer_ids():
            if customer_id in delivered:
                continue
            location = instance.locations[customer_id]
            writer.writerow(
                {
                    "customer_id": customer_id,
                    "customer_name": location.location_name,
                    "demand_kg": location.demand_kg,
                    "available_days": " ".join(str(day) for day in sorted(instance.available_days(customer_id))),
                }
            )


def export_incomplete_diagnostics_csv(path: str | Path, schedule: WeeklySchedule) -> None:
    """Export CP rolling incomplete-customer diagnostics when available."""
    diagnostics = schedule.solver_status.get("incomplete_customer_diagnostics", [])
    if not isinstance(diagnostics, list):
        diagnostics = []
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "customer_id",
        "available_days",
        "last_available_day",
        "selected_on_last_day",
        "mandatory_on_last_day",
        "stage1a_value",
        "stage1b_value",
        "stage2_value",
        "extracted_in_route",
        "diagnosis_reason",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in diagnostics:
            if not isinstance(row, dict):
                continue
            last_day = row.get("last_available_day", "")
            stage1a_values = row.get("stage1a_values_by_day", {})
            stage1b_values = row.get("stage1b_values_by_day", {})
            stage2_values = row.get("stage2_values_by_day", {})
            extracted_days = row.get("extracted_days", [])
            last_day_key = str(last_day)
            writer.writerow(
                {
                    "customer_id": row.get("customer_id", ""),
                    "available_days": " ".join(str(day) for day in row.get("available_days", [])),
                    "last_available_day": last_day,
                    "selected_on_last_day": row.get("selected_on_last_available_day", ""),
                    "mandatory_on_last_day": row.get("mandatory_on_last_available_day", ""),
                    "stage1a_value": stage1a_values.get(last_day_key, "") if isinstance(stage1a_values, dict) else "",
                    "stage1b_value": stage1b_values.get(last_day_key, "") if isinstance(stage1b_values, dict) else "",
                    "stage2_value": stage2_values.get(last_day_key, "") if isinstance(stage2_values, dict) else "",
                    "extracted_in_route": last_day in extracted_days if isinstance(extracted_days, list) else "",
                    "diagnosis_reason": row.get("diagnosis_reason", ""),
                }
            )


def export_result_txt(
    path: str | Path,
    solver_name: str,
    instance: Instance,
    schedule: WeeklySchedule,
    metrics: EvaluationMetrics,
) -> None:
    """Export a human-readable result report for one model run."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append(f"model={solver_name}")
    status = solver_status_summary(schedule, metrics)
    lines.append(f"solver_status={status.get('status', '')}")
    lines.append(f"gap_percent={format_gap_percent(status.get('gap_percent', ''))}")
    breakdown = objective_breakdown(metrics)
    lines.append("official_objective=1000 * incomplete + 100 * deferral + 10 * distance + waiting")
    lines.append(f"objective_value={metrics.objective_value:.6f}")
    for key, value in breakdown.items():
        lines.append(f"{key}={value:.6f}")
    if "objective" in status:
        lines.append(f"solver_objective={status['objective']}")
    if "best_bound" in status:
        lines.append(f"best_bound={status['best_bound']}")
    if "day_statuses" in status and isinstance(status["day_statuses"], dict):
        lines.append("day_solver_statuses")
        for day, day_status in sorted(status["day_statuses"].items()):
            if not isinstance(day_status, dict):
                continue
            lines.append(
                f"- day={day}; status={day_status.get('status', '')}; "
                f"gap_percent={format_gap_percent(day_status.get('gap_percent', ''))}; "
                f"objective={day_status.get('objective', '')}; best_bound={day_status.get('best_bound', '')}"
            )
    lines.append("summary")
    for key, value in metrics.to_dict().items():
        if key == "violations":
            continue
        if isinstance(value, float):
            lines.append(f"{key}={value:.6f}")
        else:
            lines.append(f"{key}={value}")
    if metrics.violations:
        lines.append("violations")
        for violation in metrics.violations:
            lines.append(f"- {violation}")
    lines.append("")

    lines.append("daily_routes")
    for route in schedule.ordered_routes():
        delivered_demand_kg = sum(instance.locations[stop.customer_id].demand_kg for stop in route.stops if stop.hard_feasible)
        sequence = " -> ".join(stop.customer_id for stop in route.stops) if route.stops else "(none)"
        lines.append(f"day={route.day}")
        lines.append(f"route={sequence}")
        lines.append(f"depot_departure={_format_or_blank(route.depot_departure_time)}")
        lines.append(f"return_to_depot={_format_or_blank(route.return_to_depot_time)}")
        lines.append(f"route_duration_min={route.route_duration_min}")
        lines.append(f"route_distance_km={route.route_distance_km:.6f}")
        lines.append(f"route_travel_time_min={route.route_travel_time_min}")
        lines.append(f"route_waiting_time_min={route.route_waiting_time_min}")
        lines.append(f"route_service_time_min={route.route_service_time_min}")
        lines.append(f"delivered_demand_kg={delivered_demand_kg:.6f}")
        lines.append("vehicle_capacity_kg=ignored")
        lines.append(f"hard_feasible={route.hard_feasible}")
        if route.violations:
            lines.append("route_violations")
            for violation in route.violations:
                lines.append(f"- {violation}")
        lines.append("stops")
        if not route.stops:
            lines.append("- none")
        for index, stop in enumerate(route.stops, start=1):
            location = instance.locations[stop.customer_id]
            window_start = _format_or_blank(stop.selected_time_window.start_minute if stop.selected_time_window else None)
            window_end = _format_or_blank(stop.selected_time_window.end_minute if stop.selected_time_window else None)
            lines.append(
                "- "
                f"order={index}; "
                f"customer_id={stop.customer_id}; "
                f"customer_name={location.location_name}; "
                f"demand_kg={location.demand_kg:.6f}; "
                f"arrival={_format_or_blank(stop.arrival_time)}; "
                f"service_start={_format_or_blank(stop.service_start_time)}; "
                f"service_end={_format_or_blank(stop.service_end_time)}; "
                f"window={window_start}-{window_end}; "
                f"travel_from_previous_min={stop.travel_from_previous_min}; "
                f"waiting_min={stop.waiting_min}; "
                f"distance_from_previous_km={stop.distance_from_previous_km:.6f}; "
                f"hard_feasible={stop.hard_feasible}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_report_files(
    results_dir: str | Path,
    solver_name: str,
    instance: Instance,
    schedule: WeeklySchedule,
    metrics: EvaluationMetrics | None = None,
) -> None:
    """Export report-ready CSV files for one solver schedule."""
    schedules_dir = solver_results_dir(results_dir, solver_name)
    export_daily_schedule_csv(schedules_dir / "daily_schedule.csv", instance, schedule)
    export_incomplete_orders_csv(schedules_dir / "incomplete_orders.csv", instance, schedule)
    if "incomplete_customer_diagnostics" in schedule.solver_status:
        export_incomplete_diagnostics_csv(schedules_dir / "incomplete_diagnostics.csv", schedule)
    if metrics is not None:
        export_result_txt(schedules_dir / "result.txt", solver_name, instance, schedule, metrics)


def export_benchmark_plots(summary_csv: str | Path, output_dir: str | Path) -> None:
    """Export simple benchmark bar plots using matplotlib."""
    temp_dir = Path(tempfile.gettempdir())
    os.environ.setdefault("MPLCONFIGDIR", str(temp_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(temp_dir))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    from matplotlib import pyplot as plt

    with Path(summary_csv).open(newline="", encoding="utf-8") as file_obj:
        summary = list(csv.DictReader(file_obj))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics = [
        "delivered_count",
        "incomplete_count",
        "total_distance_km",
        "total_waiting_time_min",
    ]
    for metric in metrics:
        plt.figure(figsize=(8, 4))
        plt.bar([row["solver"] for row in summary], [float(row[metric]) for row in summary])
        plt.xlabel("solver")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(output_path / f"{metric}_by_solver.png")
        plt.close()


def _format_or_blank(minutes: int | None) -> str:
    """Format optional minute values as HH:MM."""
    if minutes is None:
        return ""
    return format_hhmm(minutes) if 0 <= minutes <= 1440 else str(minutes)
