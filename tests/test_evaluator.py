"""Tests for schedule simulation and validation."""

from __future__ import annotations

from vrp_weekly.evaluator import evaluate_daily_route, evaluate_weekly_schedule, validate_schedule
from vrp_weekly.models import Instance, Location, TimeWindow, WeeklySchedule


def test_evaluate_daily_route_feasible_one_day() -> None:
    """A simple one-customer route should satisfy its time window."""
    instance = _simple_instance()

    route = evaluate_daily_route(instance, 1, ["C001"])

    assert route.hard_feasible is True
    assert route.stops[0].service_start_time == 60
    assert route.return_to_depot_time <= 1440


def test_evaluate_daily_route_infeasible_time_window() -> None:
    """A customer with an already-missed window should be marked infeasible."""
    instance = _simple_instance(window_end=5)

    route = evaluate_daily_route(instance, 1, ["C001"])

    assert route.hard_feasible is False
    assert "No feasible time window" in route.violations[0]


def test_validate_schedule_detects_duplicate_customer() -> None:
    """A customer cannot be delivered twice in the same week."""
    instance = _simple_instance()
    route1 = evaluate_daily_route(instance, 1, ["C001"])
    route2 = evaluate_daily_route(instance, 2, ["C001"])
    schedule = WeeklySchedule(routes={1: route1, 2: route2})

    violations = validate_schedule(instance, schedule)

    assert any("delivered more than once" in violation for violation in violations)


def test_evaluate_weekly_schedule_counts_incomplete_customers() -> None:
    """Undelivered customers should contribute to incomplete count."""
    instance = _simple_instance(include_second=True)
    route = evaluate_daily_route(instance, 1, ["C001"])

    metrics = evaluate_weekly_schedule(instance, WeeklySchedule(routes={1: route}))

    assert metrics.delivered_count == 1
    assert metrics.incomplete_count == 1


def _simple_instance(window_end: int = 120, include_second: bool = False) -> Instance:
    """Build a small synthetic instance."""
    locations = {
        "DEPOT": Location("DEPOT", "Kho", 0.0, 0.0, 0.0, 0, True),
        "C001": Location("C001", "Customer 1", 10.0, 0.0, 1.0, 5),
    }
    time_windows = {
        "C001": {
            1: [TimeWindow("C001", 1, 60, window_end)],
            2: [TimeWindow("C001", 2, 60, 120)],
        }
    }
    if include_second:
        locations["C002"] = Location("C002", "Customer 2", 20.0, 0.0, 1.0, 5)
        time_windows["C002"] = {1: [TimeWindow("C002", 1, 60, 120)]}
    return Instance(locations=locations, time_windows=time_windows, depot_id="DEPOT")
