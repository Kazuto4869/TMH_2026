from __future__ import annotations

from vrp_weekly.core import Instance, Location, TimeWindow
from vrp_weekly.evaluator import evaluate_weekly_schedule
from vrp_weekly.heuristics.route_eval import all_feasible_insertions
from vrp_weekly.model_factory import create_solver
from vrp_weekly.models.regret_dispatch_insertion import RegretDispatchInsertionSolver, defer_risk_score, insertion_regret_score


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


def test_regret_dispatch_hard_feasible_small_instance() -> None:
    instance = make_instance({"A": (1.0, 0.0), "B": (2.0, 0.0)}, {"A": [(1, 480, 700)], "B": [(1, 500, 800)]})

    schedule = RegretDispatchInsertionSolver().solve(instance)

    assert evaluate_weekly_schedule(instance, schedule).hard_feasible


def test_last_day_customer_has_high_defer_risk() -> None:
    instance = make_instance({"LAST": (1.0, 0.0), "LATER": (1.0, 0.0)}, {"LAST": [(1, 480, 700)], "LATER": [(1, 480, 700), (2, 480, 700)]})

    assert defer_risk_score(instance, 1, "LAST", ["LAST", "LATER"]) > defer_risk_score(instance, 1, "LATER", ["LAST", "LATER"])


def test_customer_with_one_feasible_insertion_gets_positive_regret() -> None:
    instance = make_instance({"A": (1.0, 0.0)}, {"A": [(1, 480, 700)]})

    assert insertion_regret_score(all_feasible_insertions(instance, 1, [], "A")) == 100.0


def test_regret_dispatch_can_insert_middle_not_only_append() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (3.0, 0.0), "C": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 480, 700)], "C": [(1, 480, 700)]},
    )

    insertions = all_feasible_insertions(instance, 1, ["A", "B"], "C")

    assert any(option.sequence == ["A", "C", "B"] for option in insertions)


def test_no_duplicate_delivery() -> None:
    instance = make_instance({"A": (1.0, 0.0)}, {"A": [(1, 480, 700), (2, 480, 700)]})

    schedule = RegretDispatchInsertionSolver().solve(instance)

    assert sum("A" in route.delivered_customer_ids() for route in schedule.routes.values()) == 1


def test_regret_dispatch_uses_evaluator_for_feasibility() -> None:
    instance = make_instance({"A": (1.0, 0.0)}, {"A": [(1, 480, 484)]}, service_time=5)

    schedule = RegretDispatchInsertionSolver().solve(instance)

    assert not schedule.routes[1].stops


def test_model_factory_can_create_regret_dispatch() -> None:
    assert create_solver("regret_dispatch").name == "regret_dispatch"


def test_regret_dispatch_ls_solver_status_has_use_local_search_true() -> None:
    instance = make_instance({"A": (1.0, 0.0)}, {"A": [(1, 480, 700)]})

    schedule = create_solver("regret_dispatch_ls", local_search_time_limit_sec=1).solve(instance)

    assert schedule.solver_status["use_local_search"] is True
    assert evaluate_weekly_schedule(instance, schedule).hard_feasible

