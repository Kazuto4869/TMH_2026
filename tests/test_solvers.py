"""Tests for baseline and heuristic solvers."""

from __future__ import annotations

from vrp_weekly.models import Instance, Location, TimeWindow
from vrp_weekly.solvers.earliest_deadline import EarliestDeadlineSolver
from vrp_weekly.solvers.nearest_neighbor import NearestNeighborSolver
from vrp_weekly.solvers.regret_insertion import best_feasible_insertion, customer_urgency, improve_daily_sequence


def test_nearest_neighbor_known_first_choice() -> None:
    """Nearest neighbor should choose the closest feasible first customer."""
    instance = _solver_instance()

    schedule = NearestNeighborSolver().solve(instance)

    assert schedule.routes[1].customer_sequence()[0] == "C_NEAR"


def test_deadline_solver_prioritizes_farther_earlier_deadline() -> None:
    """Deadline baseline should choose the earlier-ending feasible window before distance."""
    instance = _solver_instance()

    schedule = EarliestDeadlineSolver().solve(instance)

    assert schedule.routes[1].customer_sequence()[0] == "C_FAR_EARLY"


def test_best_feasible_insertion_can_add_customer_in_middle() -> None:
    """Best insertion should be able to insert into a non-end position."""
    instance = Instance(
        locations={
            "DEPOT": Location("DEPOT", "Kho", 0, 0, 0, 0, True),
            "A": Location("A", "A", 0, 10, 1, 5),
            "B": Location("B", "B", 10, 10, 1, 5),
            "C": Location("C", "C", 5, 10, 1, 5),
        },
        time_windows={
            "A": {1: [TimeWindow("A", 1, 0, 1440)]},
            "B": {1: [TimeWindow("B", 1, 0, 1440)]},
            "C": {1: [TimeWindow("C", 1, 0, 1440)]},
        },
        depot_id="DEPOT",
    )

    candidate = best_feasible_insertion(instance, 1, ["A", "B"], "C")

    assert candidate is not None
    assert candidate.position == 1


def test_urgency_prioritizes_last_available_day() -> None:
    """Urgency should be larger when today is the last available day."""
    instance = _solver_instance()

    assert customer_urgency(instance, 1, "C_LAST") > customer_urgency(instance, 1, "C_NEAR")


def test_local_search_improves_distance() -> None:
    """Local search should improve a crossing or poor route order."""
    instance = Instance(
        locations={
            "DEPOT": Location("DEPOT", "Kho", 0, 0, 0, 0, True),
            "A": Location("A", "A", 0, 10, 1, 5),
            "B": Location("B", "B", 10, 10, 1, 5),
            "C": Location("C", "C", 10, 0, 1, 5),
        },
        time_windows={
            "A": {1: [TimeWindow("A", 1, 0, 1440)]},
            "B": {1: [TimeWindow("B", 1, 0, 1440)]},
            "C": {1: [TimeWindow("C", 1, 0, 1440)]},
        },
        depot_id="DEPOT",
    )

    improved = improve_daily_sequence(instance, 1, ["A", "C", "B"])

    assert improved != ["A", "C", "B"]


def _solver_instance() -> Instance:
    """Build a small solver test instance."""
    return Instance(
        locations={
            "DEPOT": Location("DEPOT", "Kho", 0, 0, 0, 0, True),
            "C_NEAR": Location("C_NEAR", "Near", 1, 0, 1, 5),
            "C_FAR_EARLY": Location("C_FAR_EARLY", "Far Early", 10, 0, 1, 5),
            "C_LAST": Location("C_LAST", "Last", 2, 0, 1, 5),
        },
        time_windows={
            "C_NEAR": {1: [TimeWindow("C_NEAR", 1, 0, 1440)], 2: [TimeWindow("C_NEAR", 2, 0, 1440)]},
            "C_FAR_EARLY": {1: [TimeWindow("C_FAR_EARLY", 1, 0, 30)]},
            "C_LAST": {1: [TimeWindow("C_LAST", 1, 0, 1440)]},
        },
        depot_id="DEPOT",
    )
