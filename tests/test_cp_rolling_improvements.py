from __future__ import annotations

import inspect
from pathlib import Path
import pytest
from ortools.sat.python import cp_model

from vrp_weekly.cli import build_parser
from vrp_weekly.core import Instance, Location, TimeWindow
from vrp_weekly.evaluator import evaluate_weekly_schedule
from vrp_weekly.models.cp_full_week import FullWeekCPSATSolver
from vrp_weekly.models.cp_full_week import _build_daily_schedule_from_solution as _build_full_week_daily_schedule
from vrp_weekly.models.cp_rolling_horizon import (
    RollingHorizonCPSATSolver,
    _build_daily_schedule_from_solution,
    _can_follow,
    _can_return_to_depot,
    _can_start_from_depot,
    _candidate_priority,
    _fix_impossible_arcs,
    _get_service_time,
    _travel_time_minutes,
)


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


def test_can_follow_incompatible_windows() -> None:
    instance = make_instance(
        {"A": (0.0, 0.0), "B": (100.0, 0.0)},
        {"A": [(1, 480, 490)], "B": [(1, 491, 500)]},
    )

    assert _can_follow(instance, 1, "A", "B") is False


def test_can_follow_compatible_windows() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 480, 700)]},
    )

    assert _can_follow(instance, 1, "A", "B") is True


def test_service_not_fitting_window_is_infeasible_for_arc_check() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0)},
        {"A": [(1, 480, 484)]},
        service_time=5,
    )

    assert _can_start_from_depot(instance, 1, "A") is False


def test_can_start_from_depot_impossible() -> None:
    instance = make_instance(
        {"A": (1000.0, 0.0)},
        {"A": [(1, 480, 500)]},
    )

    assert _can_start_from_depot(instance, 1, "A") is False


def test_can_return_to_depot_impossible() -> None:
    instance = make_instance(
        {"A": (1000.0, 0.0)},
        {"A": [(1, 1430, 1440)]},
    )

    assert _can_return_to_depot(instance, 1, "A") is False


def test_fix_impossible_arcs_returns_positive_count() -> None:
    instance = make_instance(
        {"A": (0.0, 0.0), "B": (100.0, 0.0)},
        {"A": [(1, 480, 490)], "B": [(1, 491, 500)]},
    )
    model = cp_model.CpModel()
    nodes = ["DEPOT", "A", "B"]
    x = {(i, j): model.NewBoolVar(f"x[{i},{j}]") for i in nodes for j in nodes}

    stats = _fix_impossible_arcs(model, x, instance, 1, nodes, "DEPOT")

    assert stats["fixed_impossible_arcs"] > 0
    assert stats["total_nonself_arcs"] == 6


def test_candidate_filter_keeps_last_available_day() -> None:
    instance = make_instance(
        {"LAST": (10.0, 0.0), "LATER": (1.0, 0.0)},
        {"LAST": [(1, 480, 700)], "LATER": [(1, 480, 700), (2, 480, 700)]},
    )
    solver = RollingHorizonCPSATSolver(max_candidates_per_day=1)

    selected = solver._limit_candidates(instance, 1, ["LAST", "LATER"])

    assert selected == ["LAST"]


def test_hybrid_candidate_strategy_keeps_mandatory() -> None:
    instance = make_instance(
        {"LAST": (10.0, 0.0), "LATER": (1.0, 0.0), "EASY": (0.5, 0.0)},
        {
            "LAST": [(1, 480, 700)],
            "LATER": [(1, 480, 700), (2, 480, 700)],
            "EASY": [(1, 480, 1000), (2, 480, 1000)],
        },
    )
    solver = RollingHorizonCPSATSolver(max_candidates_per_day=1, candidate_strategy="hybrid")

    selected = solver._limit_candidates(instance, 1, ["LAST", "LATER", "EASY"])

    assert selected == ["LAST"]


def test_hybrid_candidate_strategy_includes_easy_candidates() -> None:
    instance = make_instance(
        {"URGENT": (20.0, 0.0), "EASY": (0.5, 0.0), "EARLY": (2.0, 0.0), "FAR": (30.0, 0.0)},
        {
            "URGENT": [(1, 480, 540), (2, 480, 540)],
            "EASY": [(1, 480, 1200), (2, 480, 1200), (3, 480, 1200)],
            "EARLY": [(1, 480, 600), (2, 480, 600), (3, 480, 600)],
            "FAR": [(1, 480, 1200), (2, 480, 1200), (3, 480, 1200)],
        },
    )
    solver = RollingHorizonCPSATSolver(max_candidates_per_day=2, candidate_strategy="hybrid")

    selected = solver._limit_candidates(instance, 1, ["URGENT", "EASY", "EARLY", "FAR"])

    assert "EASY" in selected


def test_urgent_strategy_matches_old_ordering() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0), "C": (3.0, 0.0)},
        {
            "A": [(1, 480, 700), (2, 480, 700)],
            "B": [(1, 480, 700), (2, 480, 700)],
            "C": [(1, 480, 700), (2, 480, 700)],
        },
    )
    solver = RollingHorizonCPSATSolver(max_candidates_per_day=2, candidate_strategy="urgent")

    selected = solver._limit_candidates(instance, 1, ["C", "A", "B"])
    expected = sorted(["C", "A", "B"], key=lambda customer_id: _candidate_priority(instance, 1, customer_id))[:2]

    assert selected == expected


def test_candidate_filter_is_deterministic() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0), "C": (3.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 480, 700)], "C": [(1, 480, 700)]},
    )
    solver = RollingHorizonCPSATSolver(max_candidates_per_day=2)

    first = solver._limit_candidates(instance, 1, ["C", "A", "B"])
    second = solver._limit_candidates(instance, 1, ["B", "C", "A"])

    assert first == second


def test_candidate_priority_fewer_days_first() -> None:
    instance = make_instance(
        {"FEW": (1.0, 0.0), "MANY": (2.0, 0.0)},
        {"FEW": [(1, 480, 700), (2, 480, 700)], "MANY": [(1, 480, 700), (2, 480, 700), (3, 480, 700)]},
    )

    assert _candidate_priority(instance, 1, "FEW") < _candidate_priority(instance, 1, "MANY")


def test_candidate_priority_narrower_window_first() -> None:
    instance = make_instance(
        {"NARROW": (1.0, 0.0), "WIDE": (2.0, 0.0)},
        {"NARROW": [(1, 480, 540), (2, 480, 540)], "WIDE": [(1, 480, 900), (2, 480, 900)]},
    )

    assert _candidate_priority(instance, 1, "NARROW") < _candidate_priority(instance, 1, "WIDE")


def test_urgency_last_day_greater() -> None:
    instance = make_instance(
        {"LAST": (1.0, 0.0), "LATER": (2.0, 0.0)},
        {"LAST": [(1, 480, 700)], "LATER": [(1, 480, 700), (2, 480, 700)]},
    )
    solver = RollingHorizonCPSATSolver()

    assert solver._urgency(instance, 1, "LAST") > solver._urgency(instance, 1, "LATER")


def test_urgency_earlier_deadline_greater() -> None:
    instance = make_instance(
        {"EARLY": (1.0, 0.0), "LATE": (2.0, 0.0)},
        {"EARLY": [(1, 480, 600), (2, 480, 600)], "LATE": [(1, 480, 900), (2, 480, 900)]},
    )
    solver = RollingHorizonCPSATSolver()

    assert solver._urgency(instance, 1, "EARLY") > solver._urgency(instance, 1, "LATE")


def test_urgency_narrower_window_greater() -> None:
    instance = make_instance(
        {"NARROW": (1.0, 0.0), "WIDE": (2.0, 0.0)},
        {"NARROW": [(1, 540, 600), (2, 540, 600)], "WIDE": [(1, 480, 600), (2, 480, 600)]},
    )
    solver = RollingHorizonCPSATSolver()

    assert solver._urgency(instance, 1, "NARROW") > solver._urgency(instance, 1, "WIDE")


def test_cp_rolling_returns_feasible_schedule() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)
    metrics = evaluate_weekly_schedule(instance, schedule)

    assert metrics.hard_feasible


def test_cp_rolling_still_hard_feasible() -> None:
    instance = make_instance(
        {
            "A": (1.0, 0.0),
            "B": (2.0, 0.0),
            "C": (3.0, 0.0),
        },
        {
            "A": [(1, 480, 700), (2, 480, 700)],
            "B": [(1, 500, 800), (3, 500, 800)],
            "C": [(2, 480, 760), (4, 480, 760)],
        },
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)
    metrics = evaluate_weekly_schedule(instance, schedule)

    assert metrics.hard_feasible
    assert metrics.incomplete_count == 0


def test_small_weekly_schedule_hard_feasible() -> None:
    instance = make_instance(
        {
            "A": (1.0, 0.0),
            "B": (2.0, 0.0),
            "C": (3.0, 0.0),
        },
        {
            "A": [(1, 480, 700), (2, 480, 700)],
            "B": [(1, 500, 800), (3, 500, 800)],
            "C": [(2, 480, 760), (4, 480, 760)],
        },
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)

    assert evaluate_weekly_schedule(instance, schedule).hard_feasible


def test_cp_rolling_has_no_fallback_constructor_args() -> None:
    signature = inspect.signature(RollingHorizonCPSATSolver)

    assert "fallback_to_min_deferral" not in signature.parameters
    assert "use_min_deferral_hint" not in signature.parameters


def test_no_fallback_fields_in_solver_status() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0)},
        {"A": [(1, 480, 700)]},
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)

    assert "used_fallback_days" not in schedule.solver_status
    assert all("used_fallback" not in status for status in schedule.solver_status["day_statuses"].values())


def test_degree_linking_does_not_break_small_feasible_instance() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)

    assert evaluate_weekly_schedule(instance, schedule).hard_feasible
    assert schedule.solver_status["day_statuses"][1]["degree_linking_constraints_count"] == 6


def test_unselected_customer_has_zero_service_start_time() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0)},
        {"A": [(1, 480, 700)]},
    )
    solver = RollingHorizonCPSATSolver(num_workers=1)
    data = solver._build_day_model(instance, 1, ["A"])
    data.model.Add(data.y["A"] == 0)
    cp_solver = cp_model.CpSolver()

    status = cp_solver.Solve(data.model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert cp_solver.Value(data.t["A"]) == 0


def test_service_duration_lower_bound_does_not_break_feasible_instance() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )
    solver = RollingHorizonCPSATSolver(num_workers=1)
    data = solver._build_day_model(instance, 1, ["A", "B"])
    data.model.Add(sum(data.y.values()) == 2)
    cp_solver = cp_model.CpSolver()

    status = cp_solver.Solve(data.model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_service_no_overlap_enabled_small_instance() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )
    data = RollingHorizonCPSATSolver(num_workers=1, use_service_no_overlap=True)._build_day_model(
        instance, 1, ["A", "B"]
    )

    assert data.tightening_stats["service_interval_count"] == 2


def test_route_intervals_created_for_selected_customers() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0)},
        {"A": [(1, 480, 700)]},
    )
    data = RollingHorizonCPSATSolver(num_workers=1)._build_day_model(instance, 1, ["A"])
    data.model.Add(data.y["A"] == 1)
    cp_solver = cp_model.CpSolver()

    status = cp_solver.Solve(data.model)

    assert data.tightening_stats["route_interval_count"] == 1
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert cp_solver.BooleanValue(data.y["A"])
    assert cp_solver.Value(data.interval_end["A"]) == (
        cp_solver.Value(data.t["A"])
        + _get_service_time(instance, "A")
        + cp_solver.Value(data.next_travel["A"])
    )


def test_no_overlap_route_intervals_enabled() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )
    data = RollingHorizonCPSATSolver(num_workers=1)._build_day_model(instance, 1, ["A", "B"])
    data.model.Add(sum(data.y.values()) == 2)
    cp_solver = cp_model.CpSolver()

    status = cp_solver.Solve(data.model)

    assert data.tightening_stats["no_overlap_route_intervals_enabled"] is True
    assert data.tightening_stats["depot_interval_enabled"] is True
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_depot_first_customer_equality() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )
    data = RollingHorizonCPSATSolver(num_workers=1)._build_day_model(instance, 1, ["A", "B"])
    data.model.Add(data.x["DEPOT", "A"] == 1)
    cp_solver = cp_model.CpSolver()

    status = cp_solver.Solve(data.model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert cp_solver.Value(data.depot_interval_end) == cp_solver.Value(data.t["A"])
    assert cp_solver.Value(data.t["A"]) == (
        cp_solver.Value(data.departure) + _travel_time_minutes(instance, "DEPOT", "A")
    )


def test_return_time_equals_last_interval_end() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )
    data = RollingHorizonCPSATSolver(num_workers=1)._build_day_model(instance, 1, ["A", "B"])
    data.model.Add(data.x["A", "DEPOT"] == 1)
    cp_solver = cp_model.CpSolver()

    status = cp_solver.Solve(data.model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert cp_solver.Value(data.return_time) == cp_solver.Value(data.interval_end["A"])
    assert cp_solver.Value(data.interval_end["A"]) == (
        cp_solver.Value(data.t["A"])
        + _get_service_time(instance, "A")
        + _travel_time_minutes(instance, "A", "DEPOT")
    )


def test_service_end_still_inside_window() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0)},
        {"A": [(1, 480, 486), (1, 600, 700)]},
        service_time=5,
    )
    data = RollingHorizonCPSATSolver(num_workers=1)._build_day_model(instance, 1, ["A"])
    data.model.Add(data.y["A"] == 1)
    cp_solver = cp_model.CpSolver()

    status = cp_solver.Solve(data.model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    selected_window = None
    for window_idx, window in enumerate(instance.windows_for_customer_day("A", 1)):
        if cp_solver.BooleanValue(data.g["A", window_idx]):
            selected_window = window
            break
    assert selected_window is not None
    assert cp_solver.Value(data.t["A"]) + _get_service_time(instance, "A") <= selected_window.end_minute


def test_service_no_overlap_blocks_overlapping_selected_services() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 480, 700)]},
    )
    data = RollingHorizonCPSATSolver(num_workers=1, use_service_no_overlap=True)._build_day_model(
        instance, 1, ["A", "B"]
    )
    data.model.Add(data.y["A"] == 1)
    data.model.Add(data.y["B"] == 1)
    data.model.Add(data.t["A"] == 500)
    data.model.Add(data.t["B"] == 500)
    cp_solver = cp_model.CpSolver()

    status = cp_solver.Solve(data.model)

    assert status == cp_model.INFEASIBLE


def test_roundtrip_duration_lb_added() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 480, 700)]},
    )
    data = RollingHorizonCPSATSolver(num_workers=1)._build_day_model(instance, 1, ["A", "B"])

    assert data.tightening_stats["roundtrip_duration_lb_count"] == 2


def test_impossible_customer_y_fixed_to_zero() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0)},
        {"A": [(1, 480, 484)]},
        service_time=5,
    )
    data = RollingHorizonCPSATSolver(num_workers=1)._build_day_model(instance, 1, ["A"])
    cp_solver = cp_model.CpSolver()
    status = cp_solver.Solve(data.model)

    assert data.tightening_stats["fixed_impossible_customers_count"] == 1
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert cp_solver.Value(data.y["A"]) == 0


def test_unreachable_customer_y_fixed_to_zero() -> None:
    instance = make_instance(
        {"A": (1000.0, 0.0)},
        {"A": [(1, 480, 700)]},
    )
    data = RollingHorizonCPSATSolver(num_workers=1)._build_day_model(instance, 1, ["A"])
    cp_solver = cp_model.CpSolver()
    status = cp_solver.Solve(data.model)

    assert data.tightening_stats["fixed_impossible_customers_count"] == 1
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert cp_solver.Value(data.y["A"]) == 0


def test_window_pair_incompatible_combination_is_cut() -> None:
    instance = make_instance(
        {"A": (0.0, 0.0), "B": (100.0, 0.0)},
        {"A": [(1, 480, 490)], "B": [(1, 491, 500)]},
    )
    data = RollingHorizonCPSATSolver(num_workers=1)._build_day_model(instance, 1, ["A", "B"])

    assert data.tightening_stats["window_pair_cuts_count"] > 0


def test_pair_conflict_cut_added_when_orders_cannot_coexist() -> None:
    instance = make_instance(
        {"A": (0.0, 0.0), "B": (100.0, 0.0)},
        {"A": [(1, 480, 490)], "B": [(1, 491, 500)]},
    )
    data = RollingHorizonCPSATSolver(num_workers=1)._build_day_model(instance, 1, ["A", "B"])

    assert data.tightening_stats["pair_conflict_cuts_count"] > 0


def test_depot_window_infeasible_pair_is_cut() -> None:
    instance = make_instance(
        {"A": (1000.0, 0.0)},
        {"A": [(1, 480, 700)]},
    )
    data = RollingHorizonCPSATSolver(num_workers=1)._build_day_model(instance, 1, ["A"])

    assert data.tightening_stats["depot_window_cuts_count"] > 0


def test_dominated_contained_window_is_fixed_to_zero() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0)},
        {"A": [(1, 480, 700), (1, 500, 600)]},
    )
    data = RollingHorizonCPSATSolver(num_workers=1)._build_day_model(instance, 1, ["A"])
    cp_solver = cp_model.CpSolver()
    status = cp_solver.Solve(data.model)

    assert data.tightening_stats["dominated_window_cuts_count"] == 1
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert cp_solver.Value(data.g["A", 1]) == 0


def test_precedence_cut_preserves_feasibility_on_euclidean_data() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 500)], "B": [(1, 520, 700)]},
    )
    data = RollingHorizonCPSATSolver(num_workers=1)._build_day_model(instance, 1, ["A", "B"])
    data.model.Add(sum(data.y.values()) == 2)
    cp_solver = cp_model.CpSolver()

    status = cp_solver.Solve(data.model)

    assert data.tightening_stats["precedence_cuts_count"] >= 1
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_two_phase_runs_without_heuristic() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)

    assert schedule.solver_status["day_statuses"][1]["objective_mode"] == "adaptive_three_stage"
    assert evaluate_weekly_schedule(instance, schedule).hard_feasible


def test_fixed_phase_split_still_available() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )

    schedule = RollingHorizonCPSATSolver(
        time_limit_per_day_sec=2,
        num_workers=1,
        adaptive_daily_deadline=False,
    ).solve(instance)

    assert schedule.solver_status["day_statuses"][1]["objective_mode"] == "two_phase"
    assert evaluate_weekly_schedule(instance, schedule).hard_feasible


def test_phase1_maximizes_delivered_count() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)
    day_status = schedule.solver_status["day_statuses"][1]

    assert day_status["phase1_delivered_count"] == 2


def test_phase1_prioritizes_mandatory_over_flexible() -> None:
    instance = make_instance(
        {"MANDATORY": (0.0, 0.0), "FLEX": (0.0, 0.0)},
        {
            "MANDATORY": [(1, 480, 485)],
            "FLEX": [(1, 480, 485), (2, 480, 700)],
        },
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1, solve_phase2=False).solve(instance)
    delivered_day1 = schedule.routes[1].delivered_customer_ids()
    day_status = schedule.solver_status["day_statuses"][1]

    assert "MANDATORY" in delivered_day1
    assert "FLEX" not in delivered_day1
    assert day_status["phase1_mandatory_delivered_count"] == 1
    assert day_status["phase1_total_delivered_count"] == 1


def test_mandatory_multiplier_is_lexicographically_safe() -> None:
    candidates = ["M", "F1", "F2", "F3"]
    multiplier = len(candidates) + 1

    one_more_mandatory = multiplier * 1 + 1
    all_flexible_difference = multiplier * 0 + len(candidates)

    assert one_more_mandatory > all_flexible_difference


def test_phase2_fixes_mandatory_count() -> None:
    instance = make_instance(
        {"MANDATORY": (0.0, 0.0), "FLEX": (0.0, 0.0)},
        {
            "MANDATORY": [(1, 480, 485)],
            "FLEX": [(1, 480, 485), (2, 480, 700)],
        },
    )
    solver = RollingHorizonCPSATSolver(num_workers=1)
    precomputed = solver._precompute_day(instance, 1, ["MANDATORY", "FLEX"])
    data = solver._build_day_model(instance, 1, ["MANDATORY", "FLEX"], precomputed=precomputed)
    solver._add_phase2_count_constraints(data, mandatory_delivered_count=1, total_delivered_count=1)
    data.model.Add(data.y["MANDATORY"] == 0)
    cp_solver = cp_model.CpSolver()

    status = cp_solver.Solve(data.model)

    assert status == cp_model.INFEASIBLE


def test_phase2_fixes_total_delivered_count() -> None:
    instance = make_instance(
        {"A": (0.0, 0.0), "B": (0.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 480, 700)]},
    )
    solver = RollingHorizonCPSATSolver(num_workers=1)
    precomputed = solver._precompute_day(instance, 1, ["A", "B"])
    data = solver._build_day_model(instance, 1, ["A", "B"], precomputed=precomputed)
    solver._add_phase2_count_constraints(data, mandatory_delivered_count=0, total_delivered_count=1)
    data.model.Add(data.y["A"] == 1)
    data.model.Add(data.y["B"] == 1)
    cp_solver = cp_model.CpSolver()

    status = cp_solver.Solve(data.model)

    assert status == cp_model.INFEASIBLE


def test_phase2_receives_phase1_cp_hint() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)
    day_status = schedule.solver_status["day_statuses"][1]

    assert day_status["phase2_hint_enabled"] is True
    assert day_status["phase2_hint_y_count"] == 2
    assert day_status["phase2_hint_x_count"] == 9
    assert day_status["phase2_hint_g_count"] == 2


def test_phase2_urgency_penalty_nonnegative() -> None:
    source = Path("src/vrp_weekly/models/cp_rolling_horizon.py").read_text(encoding="utf-8")

    assert "data.precomputed.urgency[customer_id] * (1 - data.y[customer_id])" in source
    assert "-self.urgency_weight *" not in source


def test_daily_compatibility_precomputed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )
    solver = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1)
    original = solver._precompute_day
    call_count = 0

    def wrapped_precompute(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_precompute_day", wrapped_precompute)

    solver.solve(instance)

    assert call_count == 1


def test_mandatory_first_decision_strategy() -> None:
    instance = make_instance(
        {"FLEX": (0.0, 0.0), "MANDATORY": (0.0, 0.0)},
        {
            "FLEX": [(1, 480, 700), (2, 480, 700)],
            "MANDATORY": [(1, 480, 700)],
        },
    )
    solver = RollingHorizonCPSATSolver(num_workers=1)
    data = solver._build_day_model(instance, 1, ["FLEX", "MANDATORY"])

    assert data.decision_strategy_customer_order[:1] == ["MANDATORY"]


def test_status_reports_mandatory_counts() -> None:
    instance = make_instance(
        {"MANDATORY": (0.0, 0.0), "FLEX": (0.0, 0.0)},
        {
            "MANDATORY": [(1, 480, 700)],
            "FLEX": [(1, 480, 700), (2, 480, 700)],
        },
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)
    day_status = schedule.solver_status["day_statuses"][1]

    assert day_status["mandatory_candidate_count"] == 1
    assert day_status["mandatory_delivered_count"] == 1
    assert day_status["mandatory_unserved_count"] == 0
    assert day_status["all_mandatory_served"] is True


def test_adaptive_stage_statuses_present_and_certifies_all_mandatory_served() -> None:
    instance = make_instance(
        {"MANDATORY": (0.0, 0.0), "FLEX": (0.0, 0.0)},
        {
            "MANDATORY": [(1, 480, 700)],
            "FLEX": [(1, 480, 700), (2, 480, 700)],
        },
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)
    day_status = schedule.solver_status["day_statuses"][1]

    assert day_status["adaptive_daily_deadline"] is True
    assert day_status["stage1a_status"] in {"OPTIMAL", "FEASIBLE"}
    assert day_status["stage1b_status"] in {"OPTIMAL", "FEASIBLE"}
    assert "stage2_status" in day_status
    assert day_status["mandatory_candidate_count"] == 1
    assert day_status["mandatory_delivered_count"] == 1
    assert day_status["all_mandatory_served"] is True
    assert day_status["mandatory_count_certified"] is True
    assert day_status["total_delivered_count"] >= 1
    assert "unused_daily_budget_sec" in day_status


def test_timing_diagnostics_present() -> None:
    instance = make_instance({"A": (1.0, 0.0)}, {"A": [(1, 480, 700)]})

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)
    day_status = schedule.solver_status["day_statuses"][1]

    for key in (
        "daily_precompute_time_sec",
        "phase1_model_build_time_sec",
        "phase1_solve_time_sec",
        "phase2_model_build_time_sec",
        "phase2_solve_time_sec",
        "stage1a_solve_time_sec",
        "stage1b_solve_time_sec",
        "stage2_solve_time_sec",
        "unused_daily_budget_sec",
        "daily_total_runtime_sec",
    ):
        assert key in day_status
        assert day_status[key] != ""


def test_cp_rolling_remains_pure_cp() -> None:
    source = Path("src/vrp_weekly/models/cp_rolling_horizon.py").read_text(encoding="utf-8")

    assert "min_deferral" not in source
    assert "regret" not in source
    assert "nearest" not in source
    assert "fallback" not in source
    assert "AddCircuit" in source


def test_phase2_fixes_delivered_count() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)

    assert len(schedule.routes[1].stops) == schedule.solver_status["day_statuses"][1]["phase1_delivered_count"]


def test_phase1_certifies_delivered_count_when_optimal() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0)},
        {"A": [(1, 480, 700)]},
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)
    day_status = schedule.solver_status["day_statuses"][1]

    assert day_status["phase1_status"] == "OPTIMAL"
    assert day_status["phase1_delivered_count_certified"] is True


def test_single_phase_mode_still_runs() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0)},
        {"A": [(1, 480, 700)]},
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1, use_two_phase_objective=False).solve(instance)

    assert schedule.solver_status["day_statuses"][1]["objective_mode"] == "single_phase"
    assert evaluate_weekly_schedule(instance, schedule).hard_feasible


def test_phase1_only_runs() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1, solve_phase2=False).solve(instance)
    day_status = schedule.solver_status["day_statuses"][1]

    assert day_status["phase1_only"] is True
    assert day_status["solve_phase2"] is False
    assert day_status["optimization_mode"] == "service_phases_only"
    assert day_status["stage1b_ran"] is True
    assert day_status["phase2_status"] == "SKIPPED"
    assert day_status["stage2_skipped_reason"] == "optimization_mode_service_phases_only"
    assert evaluate_weekly_schedule(instance, schedule).hard_feasible


def test_mandatory_stage_only_skips_service_and_route_stages() -> None:
    instance = make_instance(
        {"MANDATORY": (1.0, 0.0), "FLEX": (2.0, 0.0)},
        {
            "MANDATORY": [(1, 480, 700)],
            "FLEX": [(1, 500, 800), (2, 500, 800)],
        },
    )

    schedule = RollingHorizonCPSATSolver(
        time_limit_per_day_sec=2,
        num_workers=1,
        optimization_mode="mandatory_stage_only",
    ).solve(instance)
    day_status = schedule.solver_status["day_statuses"][1]

    assert day_status["optimization_mode"] == "mandatory_stage_only"
    assert day_status["stage1a_ran"] is True
    assert day_status["stage1b_ran"] is False
    assert day_status["stage2_ran"] is False
    assert day_status["stage1b_skipped_reason"] == "optimization_mode_mandatory_stage_only"
    assert day_status["stage2_skipped_reason"] == "optimization_mode_mandatory_stage_only"


def test_stage2_time_cap_status_present() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700)], "B": [(1, 500, 800)]},
    )

    schedule = RollingHorizonCPSATSolver(
        time_limit_per_day_sec=2,
        num_workers=1,
        stage2_max_time_fraction=0.10,
    ).solve(instance)

    assert schedule.solver_status["day_statuses"][1]["stage2_max_time_fraction"] == 0.10


def test_cli_adaptive_optimization_mode_flags() -> None:
    parser = build_parser()
    base_args = ["--locations", "data/locations.csv", "--time-windows", "data/time_windows.csv"]

    assert parser.parse_args([*base_args, "--cp-three-stage"]).cp_optimization_mode == "full_three_stage"
    assert parser.parse_args([*base_args, "--cp-service-phases-only"]).cp_optimization_mode == "service_phases_only"
    assert parser.parse_args([*base_args, "--cp-mandatory-stage-only"]).cp_optimization_mode == "mandatory_stage_only"
    assert parser.parse_args([*base_args, "--cp-phase1-only"]).cp_optimization_mode == "service_phases_only"


def test_decision_strategy_can_be_disabled() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0)},
        {"A": [(1, 480, 700)]},
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1, use_decision_strategy=False).solve(instance)

    assert schedule.solver_status["day_statuses"][1]["decision_strategy_enabled"] is False
    assert evaluate_weekly_schedule(instance, schedule).hard_feasible


def test_no_fallback_symbols_reintroduced() -> None:
    source = Path("src/vrp_weekly/models/cp_rolling_horizon.py").read_text(encoding="utf-8")

    assert "min_deferral" not in source
    assert "fallback_to_min_deferral" not in source
    assert "use_min_deferral_hint" not in source
    assert "_fallback_insertion_route" not in source
    assert "_add_min_deferral_hint" not in source


def test_global_optimality_claim_false() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0)},
        {"A": [(1, 480, 700)]},
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)

    assert schedule.solver_status["global_optimality_claim"] is False


def test_daily_optimality_claim_requires_all_days_optimal() -> None:
    solver = RollingHorizonCPSATSolver()
    status = solver._weekly_status(
        {
            1: {"status": "OPTIMAL", "runtime_sec": 0, "fixed_impossible_arcs": 0, "fixed_arc_ratio": 0},
            2: {"status": "FEASIBLE", "runtime_sec": 0, "fixed_impossible_arcs": 0, "fixed_arc_ratio": 0},
        },
        remaining_after_week=0,
    )

    assert status["daily_optimality_claim"] is False


def test_candidate_limit_scope_selected_candidates() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0)},
        {"A": [(1, 480, 700)]},
    )

    schedule = RollingHorizonCPSATSolver(max_candidates_per_day=1, time_limit_per_day_sec=2, num_workers=1).solve(instance)

    assert schedule.solver_status["daily_optimality_scope"] == "selected_candidates"


def test_no_duplicate_delivery_across_days() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0), "B": (2.0, 0.0)},
        {"A": [(1, 480, 700), (2, 480, 700)], "B": [(2, 480, 700)]},
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=2, num_workers=1).solve(instance)

    assert sum("A" in route.delivered_customer_ids() for route in schedule.routes.values()) <= 1


def test_incomplete_after_sunday() -> None:
    instance = make_instance(
        {"IMPOSSIBLE": (1000.0, 0.0)},
        {"IMPOSSIBLE": [(7, 480, 500)]},
    )

    schedule = RollingHorizonCPSATSolver(time_limit_per_day_sec=1, num_workers=1).solve(instance)
    metrics = evaluate_weekly_schedule(instance, schedule)

    assert metrics.incomplete_count == 1


def test_build_daily_schedule_uses_actual_return_time() -> None:
    instance = make_instance(
        {"A": (10.0, 0.0)},
        {"A": [(1, 480, 700)]},
    )
    window = instance.windows_for_customer_day("A", 1)[0]

    route = _build_daily_schedule_from_solution(
        instance=instance,
        day=1,
        route_sequence=["A"],
        service_start_times={"A": 600},
        selected_windows={"A": window},
        depot_departure_time=588,
    )

    assert route.return_to_depot_time == 617
    assert route.hard_feasible


def test_cp_full_week_small_instance_solves() -> None:
    instance = make_instance(
        {"A": (1.0, 0.0)},
        {"A": [(1, 480, 700)]},
    )

    schedule = FullWeekCPSATSolver(time_limit_sec=2, max_customers=1, num_workers=1).solve(instance)
    metrics = evaluate_weekly_schedule(instance, schedule)

    assert metrics.hard_feasible


def test_cp_full_week_build_daily_schedule_uses_actual_return_time() -> None:
    instance = make_instance(
        {"A": (10.0, 0.0)},
        {"A": [(1, 480, 700)]},
    )
    window = instance.windows_for_customer_day("A", 1)[0]

    route = _build_full_week_daily_schedule(
        instance=instance,
        day=1,
        route_sequence=["A"],
        service_start_times={"A": 600},
        selected_windows={"A": window},
        depot_departure_time=588,
    )

    assert route.return_to_depot_time == 617
    assert route.route_duration_min == 29
    assert route.hard_feasible


def test_cp_full_week_unlimited_large_instance_raises() -> None:
    coords = {f"C{i:03d}": (float(i), 0.0) for i in range(81)}
    windows = {customer_id: [(1, 480, 700)] for customer_id in coords}
    instance = make_instance(coords, windows)

    with pytest.raises(ValueError, match="cp_full_week has O\\(7\\*n\\^2\\) arc variables"):
        FullWeekCPSATSolver(max_customers=None).solve(instance)
