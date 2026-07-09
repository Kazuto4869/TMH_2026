from __future__ import annotations

from vrp_weekly.core import Instance, Location, TimeWindow
from vrp_weekly.evaluator import evaluate_weekly_schedule
from vrp_weekly.model_factory import create_solver
from vrp_weekly.models.inferior_insertion import InferiorInsertionSolver, inferiority_score


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


def test_inferior_insertion_hard_feasible_small_instance() -> None:
    instance = make_instance({"A": (1.0, 0.0), "B": (2.0, 0.0)}, {"A": [(1, 480, 700)], "B": [(1, 500, 800)]})

    schedule = InferiorInsertionSolver().solve(instance)

    assert evaluate_weekly_schedule(instance, schedule).hard_feasible


def test_last_available_day_customer_prioritized() -> None:
    instance = make_instance({"LAST": (1.0, 0.0), "LATER": (1.0, 0.0)}, {"LAST": [(1, 480, 700)], "LATER": [(1, 480, 700), (2, 480, 700)]})

    assert inferiority_score(instance, 1, "LAST", ["LAST", "LATER"]) > inferiority_score(instance, 1, "LATER", ["LAST", "LATER"])


def test_narrower_window_prioritized_when_other_factors_tie() -> None:
    instance = make_instance({"NARROW": (1.0, 0.0), "WIDE": (1.0, 0.0)}, {"NARROW": [(1, 480, 540), (2, 480, 540)], "WIDE": [(1, 480, 900), (2, 480, 900)]})

    assert inferiority_score(instance, 1, "NARROW", ["NARROW", "WIDE"]) > inferiority_score(instance, 1, "WIDE", ["NARROW", "WIDE"])


def test_no_duplicate_delivery_across_week() -> None:
    instance = make_instance({"A": (1.0, 0.0)}, {"A": [(1, 480, 700), (2, 480, 700)]})

    schedule = InferiorInsertionSolver().solve(instance)

    assert sum("A" in route.delivered_customer_ids() for route in schedule.routes.values()) == 1


def test_customer_not_fit_today_carried_to_later_day() -> None:
    instance = make_instance({"A": (1.0, 0.0)}, {"A": [(1, 480, 484), (2, 480, 700)]}, service_time=5)

    schedule = InferiorInsertionSolver().solve(instance)

    assert "A" not in schedule.routes[1].delivered_customer_ids()
    assert "A" in schedule.routes[2].delivered_customer_ids()


def test_inferior_insertion_uses_evaluator_for_feasibility() -> None:
    instance = make_instance({"A": (1.0, 0.0)}, {"A": [(1, 480, 484)]}, service_time=5)

    schedule = InferiorInsertionSolver().solve(instance)

    assert not schedule.routes[1].stops


def test_model_factory_can_create_inferior_insertion() -> None:
    assert create_solver("inferior_insertion").name == "inferior_insertion"


def test_inferior_insertion_ls_solver_status_has_use_local_search_true() -> None:
    instance = make_instance({"A": (1.0, 0.0)}, {"A": [(1, 480, 700)]})

    schedule = create_solver("inferior_insertion_ls", local_search_time_limit_sec=1).solve(instance)

    assert schedule.solver_status["use_local_search"] is True
    assert evaluate_weekly_schedule(instance, schedule).hard_feasible

