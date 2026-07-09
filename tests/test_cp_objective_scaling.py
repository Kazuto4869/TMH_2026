from __future__ import annotations

from vrp_weekly.core import Instance, Location, TimeWindow
from vrp_weekly.evaluator import evaluate_weekly_schedule
from vrp_weekly.models.cp_rolling_horizon import RollingHorizonCPSATSolver
from vrp_weekly.models.cp_rolling_horizon import _distance_objective_cost as rolling_distance_objective_cost


def make_instance(
    coords: dict[str, tuple[float, float]],
    windows: dict[str, list[tuple[int, int, int]]],
    service_time: int = 5,
) -> Instance:
    locations = {
        "DEPOT": Location("DEPOT", "Depot", 0.0, 0.0, service_time=0, is_depot=True),
    }
    for customer_id, (x_km, y_km) in coords.items():
        locations[customer_id] = Location(customer_id, customer_id, x_km, y_km, service_time=service_time)

    grouped: dict[str, dict[int, list[TimeWindow]]] = {}
    for customer_id, raw_windows in windows.items():
        for day, start, end in raw_windows:
            grouped.setdefault(customer_id, {}).setdefault(day, []).append(TimeWindow(customer_id, day, start, end))
    return Instance(locations=locations, time_windows=grouped)


def test_distance_objective_cost_not_meter_scaled() -> None:
    instance = make_instance({"A": (1.0, 0.0)}, {"A": [(1, 480, 700)]})

    cost = rolling_distance_objective_cost(instance, "DEPOT", "A", 10)

    assert cost == 10
    assert cost != 10_000


def test_cp_prefers_delivering_single_easy_customer() -> None:
    instance = make_instance({"A": (1.0, 0.0)}, {"A": [(1, 480, 700)]})

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)
    metrics = evaluate_weekly_schedule(instance, schedule)

    assert "A" in schedule.delivered_customer_ids()
    assert metrics.hard_feasible


def test_empty_route_not_optimal_when_easy_customer_exists() -> None:
    instance = make_instance({"A": (1.0, 0.0)}, {"A": [(1, 480, 700)]})

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)

    assert len(schedule.routes[1].stops) == 1
