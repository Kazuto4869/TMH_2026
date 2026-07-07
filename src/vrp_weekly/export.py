"""JSON, CSV, and plot exports for schedules and benchmark results."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from vrp_weekly.models import EvaluationMetrics, Instance, WeeklySchedule
from vrp_weekly.time_utils import format_hhmm


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


def save_result_json(path: str | Path, solver_name: str, schedule: WeeklySchedule, metrics: EvaluationMetrics) -> None:
    """Save one solver result as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "solver": solver_name,
        "metrics": metrics.to_dict(),
        "schedule": schedule_to_dict(schedule),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def export_report_files(
    results_dir: str | Path,
    solver_name: str,
    instance: Instance,
    schedule: WeeklySchedule,
) -> None:
    """Export report-ready CSV files for one solver schedule."""
    schedules_dir = Path(results_dir) / "schedules"
    export_daily_schedule_csv(schedules_dir / f"{solver_name}_daily_schedule.csv", instance, schedule)
    export_incomplete_orders_csv(schedules_dir / f"{solver_name}_incomplete_orders.csv", instance, schedule)


def export_benchmark_plots(summary_csv: str | Path, output_dir: str | Path) -> None:
    """Export simple benchmark bar plots using matplotlib."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
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
