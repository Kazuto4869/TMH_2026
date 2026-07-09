from __future__ import annotations

from vrp_weekly.core import Instance, Location, TimeWindow, WeeklySchedule
from vrp_weekly.evaluator import evaluate_daily_route, evaluate_weekly_schedule
from vrp_weekly.heuristics.local_search import LocalSearchParams, improve_daily_route, improve_weekly_schedule
from vrp_weekly.heuristics.route_eval import route_customer_ids, route_secondary_cost


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


def test_relocate_preserves_customer_set() -> None:
    instance = make_instance({"A": (1, 0), "B": (2, 0), "C": (3, 0)}, {"A": [(1, 480, 900)], "B": [(1, 480, 900)], "C": [(1, 480, 900)]})
    route = evaluate_daily_route(instance, 1, ["C", "A", "B"])

    improved = improve_daily_route(instance, 1, route, params=LocalSearchParams(enable_swap=False, enable_two_opt=False, enable_remove_reinsert=False, enable_post_fill=False))

    assert set(route_customer_ids(improved)) == {"A", "B", "C"}


def test_swap_preserves_customer_set() -> None:
    instance = make_instance({"A": (1, 0), "B": (2, 0)}, {"A": [(1, 480, 900)], "B": [(1, 480, 900)]})
    route = evaluate_daily_route(instance, 1, ["B", "A"])

    improved = improve_daily_route(instance, 1, route, params=LocalSearchParams(enable_relocate=False, enable_two_opt=False, enable_remove_reinsert=False, enable_post_fill=False))

    assert set(route_customer_ids(improved)) == {"A", "B"}


def test_two_opt_preserves_customer_set() -> None:
    instance = make_instance({"A": (1, 0), "B": (2, 0), "C": (3, 0)}, {"A": [(1, 480, 900)], "B": [(1, 480, 900)], "C": [(1, 480, 900)]})
    route = evaluate_daily_route(instance, 1, ["C", "B", "A"])

    improved = improve_daily_route(instance, 1, route, params=LocalSearchParams(enable_relocate=False, enable_swap=False, enable_remove_reinsert=False, enable_post_fill=False))

    assert set(route_customer_ids(improved)) == {"A", "B", "C"}


def test_remove_reinsert_preserves_customer_set() -> None:
    instance = make_instance({"A": (1, 0), "B": (2, 0)}, {"A": [(1, 480, 900)], "B": [(1, 480, 900)]})
    route = evaluate_daily_route(instance, 1, ["B", "A"])

    improved = improve_daily_route(instance, 1, route, params=LocalSearchParams(enable_relocate=False, enable_swap=False, enable_two_opt=False, enable_post_fill=False))

    assert set(route_customer_ids(improved)) == {"A", "B"}


def test_local_search_never_returns_infeasible_route() -> None:
    instance = make_instance({"A": (1, 0), "B": (2, 0)}, {"A": [(1, 480, 900)], "B": [(1, 480, 900)]})
    route = evaluate_daily_route(instance, 1, ["B", "A"])

    assert improve_daily_route(instance, 1, route).hard_feasible


def test_post_fill_can_add_extra_feasible_customer() -> None:
    instance = make_instance({"A": (1, 0), "B": (2, 0)}, {"A": [(1, 480, 900)], "B": [(1, 480, 900)]})
    route = evaluate_daily_route(instance, 1, ["A"])

    improved = improve_daily_route(instance, 1, route, undelivered_today=["B"], params=LocalSearchParams(enable_relocate=False, enable_swap=False, enable_two_opt=False, enable_remove_reinsert=False))

    assert "B" in route_customer_ids(improved)


def test_post_fill_does_not_add_infeasible_customer() -> None:
    instance = make_instance({"A": (1, 0), "B": (2, 0)}, {"A": [(1, 480, 900)], "B": [(1, 480, 484)]}, service_time=5)
    route = evaluate_daily_route(instance, 1, ["A"])

    improved = improve_daily_route(instance, 1, route, undelivered_today=["B"])

    assert "B" not in route_customer_ids(improved)


def test_improve_weekly_schedule_no_duplicates() -> None:
    instance = make_instance({"A": (1, 0), "B": (2, 0)}, {"A": [(1, 480, 900), (2, 480, 900)], "B": [(2, 480, 900)]})
    schedule = WeeklySchedule(routes={1: evaluate_daily_route(instance, 1, ["A"]), 2: evaluate_daily_route(instance, 2, ["B"])})

    improved = improve_weekly_schedule(instance, schedule, params=LocalSearchParams(time_limit_sec=1))

    assert evaluate_weekly_schedule(instance, improved).hard_feasible


def test_local_search_does_not_worsen_route_cost_when_same_delivered_set() -> None:
    instance = make_instance({"A": (1, 0), "B": (2, 0)}, {"A": [(1, 480, 900)], "B": [(1, 480, 900)]})
    route = evaluate_daily_route(instance, 1, ["B", "A"])

    improved = improve_daily_route(instance, 1, route, params=LocalSearchParams(enable_post_fill=False))

    assert route_secondary_cost(improved) <= route_secondary_cost(route) + 1e-9

