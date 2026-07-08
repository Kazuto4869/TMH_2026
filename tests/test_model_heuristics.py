"""Tests for greedy baselines and regret heuristic behavior."""

from __future__ import annotations

from vrp_weekly.core import Instance, Location, TimeWindow
from vrp_weekly.evaluator import evaluate_weekly_schedule
from vrp_weekly.models.deadline import EarliestDeadlineSolver
from vrp_weekly.models.nearest import NearestNeighborSolver
from vrp_weekly.models.regret import (
    RegretInsertionSolver,
    best_feasible_insertion,
    fill_extra_customers,
)


def test_nearest_tie_break_chooses_earlier_selected_window_end() -> None:
    instance = Instance(
        locations={
            "DEPOT": Location("DEPOT", "Depot", 0, 0, 0, 0, True),
            "C_LATE": Location("C_LATE", "Late", 1, 0, 1, 5),
            "C_EARLY": Location("C_EARLY", "Early", -1, 0, 1, 5),
        },
        time_windows={
            "C_LATE": {1: [TimeWindow("C_LATE", 1, 100, 300)]},
            "C_EARLY": {1: [TimeWindow("C_EARLY", 1, 100, 200)]},
        },
        depot_id="DEPOT",
    )

    schedule = NearestNeighborSolver().solve(instance)

    assert schedule.routes[1].customer_sequence()[0] == "C_EARLY"


def test_deadline_tie_break_chooses_earlier_selected_window_start() -> None:
    instance = Instance(
        locations={
            "DEPOT": Location("DEPOT", "Depot", 0, 0, 0, 0, True),
            "C_LATE_START": Location("C_LATE_START", "Late Start", 1, 0, 1, 5),
            "C_EARLY_START": Location("C_EARLY_START", "Early Start", 10, 0, 1, 5),
        },
        time_windows={
            "C_LATE_START": {1: [TimeWindow("C_LATE_START", 1, 100, 300)]},
            "C_EARLY_START": {1: [TimeWindow("C_EARLY_START", 1, 50, 300)]},
        },
        depot_id="DEPOT",
    )

    schedule = EarliestDeadlineSolver().solve(instance)

    assert schedule.routes[1].customer_sequence()[0] == "C_EARLY_START"


def test_regret_single_feasible_insertion_has_positive_regret() -> None:
    instance = Instance(
        locations={
            "DEPOT": Location("DEPOT", "Depot", 0, 0, 0, 0, True),
            "A": Location("A", "A", 0, 0, 1, 5),
            "B": Location("B", "B", 0, 0, 1, 5),
        },
        time_windows={
            "A": {1: [TimeWindow("A", 1, 100, 200)]},
            "B": {1: [TimeWindow("B", 1, 0, 10)]},
        },
        depot_id="DEPOT",
    )

    candidate = best_feasible_insertion(instance, 1, ["A"], "B", single_insertion_regret_bonus=123.0)

    assert candidate is not None
    assert candidate.regret == 123.0


def test_regret_fill_extra_customers_can_add_one_more_customer() -> None:
    instance = Instance(
        locations={
            "DEPOT": Location("DEPOT", "Depot", 0, 0, 0, 0, True),
            "A": Location("A", "A", 1, 0, 1, 5),
            "B": Location("B", "B", 2, 0, 1, 5),
        },
        time_windows={
            "A": {1: [TimeWindow("A", 1, 0, 1440)]},
            "B": {1: [TimeWindow("B", 1, 0, 1440)]},
        },
        depot_id="DEPOT",
    )

    sequence, inserted = fill_extra_customers(instance, 1, ["A"], {"B"})

    assert inserted == ["B"]
    assert set(sequence) == {"A", "B"}


def test_greedy_and_regret_solvers_return_hard_feasible_schedules() -> None:
    instance = Instance(
        locations={
            "DEPOT": Location("DEPOT", "Depot", 0, 0, 0, 0, True),
            "A": Location("A", "A", 1, 0, 1, 5),
            "B": Location("B", "B", 2, 0, 1, 5),
            "C": Location("C", "C", 3, 0, 1, 5),
        },
        time_windows={
            "A": {1: [TimeWindow("A", 1, 0, 1440)]},
            "B": {1: [TimeWindow("B", 1, 0, 1440)]},
            "C": {2: [TimeWindow("C", 2, 0, 1440)]},
        },
        depot_id="DEPOT",
    )

    for solver in [NearestNeighborSolver(), EarliestDeadlineSolver(), RegretInsertionSolver()]:
        schedule = solver.solve(instance)
        metrics = evaluate_weekly_schedule(instance, schedule)
        assert metrics.hard_feasible is True
