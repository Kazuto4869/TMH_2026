from __future__ import annotations

from vrp_weekly.core import Instance, Location, TimeWindow, WeeklySchedule
from vrp_weekly.evaluator import evaluate_daily_route
from vrp_weekly.heuristics.route_eval import (
    all_feasible_insertions,
    best_feasible_insertion,
    route_customer_ids,
    validate_no_duplicates,
    weekly_score,
    windows_for,
)
from vrp_weekly.heuristics.scoring import (
    available_days,
    earliest_available_day,
    is_last_available_day,
    remaining_available_days,
    window_width_loss_to_future,
)


def make_instance(
    coords: dict[str, tuple[float, float]],
    windows: dict[str, list[tuple[int, int, int]]],
    service_time: int = 5,
) -> Instance:
    locations = {"DEPOT": Location("DEPOT", "Depot", 0.0, 0.0, service_time=0, is_depot=True)}
    for customer_id, (x_km, y_km) in coords.items():
        locations[customer_id] = Location(customer_id, customer_id, x_km, y_km, service_time=service_time)
    grouped: dict[str, dict[int, list[TimeWindow]]] = {}
    for customer_id, raw_windows in windows.items():
        for day, start, end in raw_windows:
            grouped.setdefault(customer_id, {}).setdefault(day, []).append(TimeWindow(customer_id, day, start, end))
    return Instance(locations=locations, time_windows=grouped)


def test_route_eval_best_and_all_insertions_use_evaluator() -> None:
    instance = make_instance({"A": (1.0, 0.0), "B": (2.0, 0.0)}, {"A": [(1, 480, 700)], "B": [(1, 500, 800)]})

    option = best_feasible_insertion(instance, 1, ["A"], "B")
    all_options = all_feasible_insertions(instance, 1, ["A"], "B")

    assert option is not None
    assert all_options
    assert option.route.hard_feasible
    assert option.sequence == option.route.customer_sequence()


def test_route_eval_rejects_infeasible_insertion() -> None:
    instance = make_instance({"A": (1.0, 0.0)}, {"A": [(1, 480, 484)]}, service_time=5)

    assert best_feasible_insertion(instance, 1, [], "A") is None


def test_scoring_available_days_and_last_day() -> None:
    instance = make_instance({"A": (1.0, 0.0)}, {"A": [(2, 480, 700), (4, 480, 700)]})

    assert windows_for(instance, "A", 2)
    assert earliest_available_day(instance, "A") == 2
    assert available_days(instance, "A") == [2, 4]
    assert remaining_available_days(instance, "A", 3) == [4]
    assert is_last_available_day(instance, "A", 4)
    assert window_width_loss_to_future(instance, "A", 4) == 1.0


def test_validate_no_duplicates_and_weekly_score() -> None:
    instance = make_instance({"A": (1.0, 0.0)}, {"A": [(1, 480, 700)]})
    route = evaluate_daily_route(instance, 1, ["A"])
    schedule = WeeklySchedule(routes={1: route})

    assert route_customer_ids(route) == ["A"]
    assert validate_no_duplicates(schedule)
    assert weekly_score(instance, schedule) < weekly_score(instance, WeeklySchedule(routes={1: evaluate_daily_route(instance, 1, [])}))

