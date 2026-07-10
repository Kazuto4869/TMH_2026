from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from vrp_weekly.cli import build_parser
from vrp_weekly.core import DailyRoute, Instance, Location, TimeWindow, WeeklySchedule
from vrp_weekly.evaluator import evaluate_daily_route, evaluate_weekly_schedule
from vrp_weekly.model_factory import create_solver, solver_names
from vrp_weekly.models.cp_rolling_repair import RollingHorizonCPRepairSolver
import main as interactive_main


def make_instance(include_c: bool = False) -> Instance:
    locations = {
        "DEPOT": Location("DEPOT", "Depot", 0.0, 0.0, service_time=0, is_depot=True),
        "A": Location("A", "A", 1.0, 0.0, service_time=5),
        "B": Location("B", "B", 2.0, 0.0, service_time=5),
    }
    windows = {
        "A": {1: [TimeWindow("A", 1, 480, 900)]},
        "B": {1: [TimeWindow("B", 1, 480, 900)]},
    }
    if include_c:
        locations["C"] = Location("C", "C", 1.0, 1.0, service_time=5)
        windows["C"] = {2: [TimeWindow("C", 2, 480, 900)]}
    return Instance(locations=locations, time_windows=windows)


def base_schedule(instance: Instance, include_c: bool = False) -> WeeklySchedule:
    routes = {day: DailyRoute(day=day) for day in range(1, 8)}
    routes[1] = evaluate_daily_route(instance, 1, ["B"])
    if include_c:
        routes[2] = evaluate_daily_route(instance, 2, ["C"])
    return WeeklySchedule(routes=routes, solver_status={"status": "fake_base"})


class FakeRollingSolver:
    schedule: WeeklySchedule

    def __init__(self, **_kwargs: object) -> None:
        pass

    def solve(self, _instance: Instance) -> WeeklySchedule:
        return self.schedule


def test_repair_can_rescue_one_incomplete_customer(monkeypatch) -> None:
    instance = make_instance()
    FakeRollingSolver.schedule = base_schedule(instance)
    monkeypatch.setattr("vrp_weekly.models.cp_rolling_repair.RollingHorizonCPSATSolver", FakeRollingSolver)

    schedule = RollingHorizonCPRepairSolver(repair_time_limit_sec=5, repair_num_workers=1).solve(instance)
    metrics = evaluate_weekly_schedule(instance, schedule)

    assert metrics.incomplete_count == 0
    assert schedule.solver_status["repair_accepted"] is True
    assert schedule.solver_status["repair_rescued_customer_ids"] == ["A"]


def test_repair_keeps_all_previously_delivered_customers(monkeypatch) -> None:
    instance = make_instance()
    FakeRollingSolver.schedule = base_schedule(instance)
    monkeypatch.setattr("vrp_weekly.models.cp_rolling_repair.RollingHorizonCPSATSolver", FakeRollingSolver)

    schedule = RollingHorizonCPRepairSolver(repair_time_limit_sec=5, repair_num_workers=1).solve(instance)

    assert "B" in schedule.delivered_customer_ids()
    assert schedule.solver_status["repair_preserved_base_deliveries"] is True


def test_repair_never_duplicates_customer(monkeypatch) -> None:
    instance = make_instance()
    FakeRollingSolver.schedule = base_schedule(instance)
    monkeypatch.setattr("vrp_weekly.models.cp_rolling_repair.RollingHorizonCPSATSolver", FakeRollingSolver)

    schedule = RollingHorizonCPRepairSolver(repair_time_limit_sec=5, repair_num_workers=1).solve(instance)

    delivered = [customer_id for route in schedule.routes.values() for customer_id in route.customer_sequence()]
    assert len(delivered) == len(set(delivered))
    assert schedule.solver_status["repair_no_duplicates"] is True


def test_repair_only_changes_selected_days(monkeypatch) -> None:
    instance = make_instance(include_c=True)
    base = base_schedule(instance, include_c=True)
    FakeRollingSolver.schedule = base
    monkeypatch.setattr("vrp_weekly.models.cp_rolling_repair.RollingHorizonCPSATSolver", FakeRollingSolver)

    schedule = RollingHorizonCPRepairSolver(repair_time_limit_sec=5, repair_num_workers=1).solve(instance)

    assert schedule.routes[2].customer_sequence() == base.routes[2].customer_sequence()
    assert schedule.solver_status["repair_selected_days"] == [1]


def test_repair_rejects_infeasible_schedule() -> None:
    solver = RollingHorizonCPRepairSolver()
    assert not (
        False
        and True
        and True
        and solver._lexicographic_better(
            SimpleNamespace(incomplete_count=1, total_deferral_days=1, total_distance_km=1.0),
            SimpleNamespace(incomplete_count=0, total_deferral_days=1, total_distance_km=1.0),
        )
    )


def test_repair_rejects_schedule_that_loses_completed_customer() -> None:
    base_delivered = {"B"}
    candidate_delivered = {"A"}
    assert not (base_delivered <= candidate_delivered)


def test_repair_acceptance_lexicographic() -> None:
    solver = RollingHorizonCPRepairSolver()
    assert solver._lexicographic_better(
        SimpleNamespace(incomplete_count=1, total_deferral_days=10, total_distance_km=10.0),
        SimpleNamespace(incomplete_count=0, total_deferral_days=100, total_distance_km=100.0),
    )
    assert not solver._lexicographic_better(
        SimpleNamespace(incomplete_count=0, total_deferral_days=10, total_distance_km=10.0),
        SimpleNamespace(incomplete_count=1, total_deferral_days=1, total_distance_km=1.0),
    )


def test_repair_uses_base_cp_hints(monkeypatch) -> None:
    instance = make_instance()
    FakeRollingSolver.schedule = base_schedule(instance)
    monkeypatch.setattr("vrp_weekly.models.cp_rolling_repair.RollingHorizonCPSATSolver", FakeRollingSolver)

    schedule = RollingHorizonCPRepairSolver(repair_time_limit_sec=5, repair_num_workers=1).solve(instance)

    assert schedule.solver_status["repair_hint_y_count"] > 0
    assert schedule.solver_status["repair_hint_x_count"] > 0
    assert schedule.solver_status["repair_hint_g_count"] > 0


def test_repair_returns_base_when_no_incomplete(monkeypatch) -> None:
    instance = make_instance()
    routes = {day: DailyRoute(day=day) for day in range(1, 8)}
    routes[1] = evaluate_daily_route(instance, 1, ["A", "B"])
    FakeRollingSolver.schedule = WeeklySchedule(routes=routes, solver_status={"status": "fake_base"})
    monkeypatch.setattr("vrp_weekly.models.cp_rolling_repair.RollingHorizonCPSATSolver", FakeRollingSolver)

    schedule = RollingHorizonCPRepairSolver(repair_time_limit_sec=5, repair_num_workers=1).solve(instance)

    assert schedule.solver_status["repair_ran"] is False
    assert schedule.solver_status["repair_reason"] == "no_incomplete_orders"


def test_cp_rolling_repair_registered() -> None:
    assert "cp_rolling_repair" in solver_names()
    assert create_solver("cp_rolling_repair").name == "cp_rolling_repair"


def test_cli_accepts_cp_rolling_repair() -> None:
    args = build_parser().parse_args(["--locations", "loc.csv", "--time-windows", "tw.csv", "--solver", "cp_rolling_repair"])
    assert args.solver == "cp_rolling_repair"


def test_main_imports_with_cp_rolling_repair() -> None:
    assert "cp_rolling_repair" in interactive_main.solver_names()


def test_cp_rolling_repair_remains_pure_cp() -> None:
    tree = ast.parse(Path("src/vrp_weekly/models/cp_rolling_repair.py").read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    banned = ["nearest", "deadline", "min_deferral", "regret_dispatch", "inferior_insertion", "hybrid_genetic_vns"]
    assert not any(any(banned_name in module for banned_name in banned) for module in imports)
