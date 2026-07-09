from __future__ import annotations

from collections import Counter

from vrp_weekly.core import Instance, Location, TimeWindow
from vrp_weekly.evaluator import evaluate_weekly_schedule
from vrp_weekly.models.deadline import EarliestDeadlineSolver
from vrp_weekly.models.nearest import NearestNeighborSolver


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


def delivered_sequence(schedule, day: int) -> list[str]:
    return [stop.customer_id for stop in schedule.routes[day].stops]


def delivered_counts(schedule) -> Counter[str]:
    return Counter(stop.customer_id for route in schedule.routes.values() for stop in route.stops)


def test_nearest_chooses_closer_feasible_customer() -> None:
    instance = make_instance(
        {"NEAR": (5.0, 0.0), "FAR": (20.0, 0.0)},
        {"NEAR": [(1, 480, 900)], "FAR": [(1, 480, 900)]},
    )

    schedule = NearestNeighborSolver().solve(instance)

    assert delivered_sequence(schedule, 1)[0] == "NEAR"


def test_nearest_ties_by_selected_window_end() -> None:
    instance = make_instance(
        {"LATE": (10.0, 0.0), "EARLY": (-10.0, 0.0)},
        {"LATE": [(1, 480, 700)], "EARLY": [(1, 480, 600)]},
    )

    schedule = NearestNeighborSolver().solve(instance)

    assert delivered_sequence(schedule, 1)[0] == "EARLY"


def test_nearest_schedule_is_hard_feasible() -> None:
    instance = make_instance(
        {"A": (5.0, 0.0), "B": (10.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )

    schedule = NearestNeighborSolver().solve(instance)
    metrics = evaluate_weekly_schedule(instance, schedule)

    assert metrics.hard_feasible


def test_nearest_has_no_duplicate_delivery_across_days() -> None:
    instance = make_instance(
        {"A": (5.0, 0.0), "B": (10.0, 0.0)},
        {"A": [(1, 480, 700), (2, 480, 700)], "B": [(2, 480, 700)]},
    )

    schedule = NearestNeighborSolver().solve(instance)

    assert delivered_counts(schedule)["A"] == 1


def test_deadline_chooses_earliest_selected_window_end_even_if_farther() -> None:
    instance = make_instance(
        {"NEAR_LATE": (5.0, 0.0), "FAR_EARLY": (20.0, 0.0)},
        {"NEAR_LATE": [(1, 480, 800)], "FAR_EARLY": [(1, 480, 650)]},
    )

    schedule = EarliestDeadlineSolver().solve(instance)

    assert delivered_sequence(schedule, 1)[0] == "FAR_EARLY"


def test_deadline_ties_by_selected_window_start() -> None:
    instance = make_instance(
        {"EARLY_START": (10.0, 0.0), "LATE_START": (-10.0, 0.0)},
        {"EARLY_START": [(1, 480, 700)], "LATE_START": [(1, 520, 700)]},
    )

    schedule = EarliestDeadlineSolver().solve(instance)

    assert delivered_sequence(schedule, 1)[0] == "EARLY_START"


def test_deadline_ties_by_travel_time() -> None:
    instance = make_instance(
        {"NEAR": (5.0, 0.0), "FAR": (20.0, 0.0)},
        {"NEAR": [(1, 480, 700)], "FAR": [(1, 480, 700)]},
    )

    schedule = EarliestDeadlineSolver().solve(instance)

    assert delivered_sequence(schedule, 1)[0] == "NEAR"


def test_deadline_schedule_is_hard_feasible() -> None:
    instance = make_instance(
        {"A": (5.0, 0.0), "B": (10.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )

    schedule = EarliestDeadlineSolver().solve(instance)
    metrics = evaluate_weekly_schedule(instance, schedule)

    assert metrics.hard_feasible
