from __future__ import annotations

from vrp_weekly.core import DailyRoute, Instance, Location, TimeWindow
from vrp_weekly.models.cp_rolling_horizon import RollingHorizonCPSATSolver


def make_instance() -> Instance:
    return Instance(
        locations={
            "DEPOT": Location("DEPOT", "Depot", 0.0, 0.0, service_time=0, is_depot=True),
            "A": Location("A", "A", 1.0, 0.0, service_time=5),
            "B": Location("B", "B", 2.0, 0.0, service_time=5),
        },
        time_windows={
            "A": {1: [TimeWindow("A", 1, 480, 900)]},
            "B": {1: [TimeWindow("B", 1, 480, 900)], 2: [TimeWindow("B", 2, 480, 900)]},
        },
    )


def test_incomplete_diagnostic_reports_available_days() -> None:
    solver = RollingHorizonCPSATSolver(time_limit_per_day_sec=1, num_workers=1)
    rows = solver._build_incomplete_customer_diagnostics(make_instance(), {}, {"A"})
    assert rows[0]["available_days"] == [1]


def test_incomplete_diagnostic_reports_last_available_day() -> None:
    solver = RollingHorizonCPSATSolver(time_limit_per_day_sec=1, num_workers=1)
    rows = solver._build_incomplete_customer_diagnostics(make_instance(), {}, {"B"})
    assert rows[0]["last_available_day"] == 2


def test_filtered_customer_reason_candidate_limit() -> None:
    solver = RollingHorizonCPSATSolver()
    day_status = {
        "filtered_candidate_ids": ["A"],
        "mandatory_candidate_ids": ["A"],
        "stage1a_selected_ids": [],
        "stage1b_selected_ids": [],
        "stage2_selected_ids": [],
        "mandatory_count_certified": False,
        "stage1b_ran": True,
    }
    assert solver._diagnose_customer_day_reason("A", day_status, set()) == "filtered_by_candidate_limit"


def test_selected_but_unserved_reason() -> None:
    solver = RollingHorizonCPSATSolver()
    day_status = {
        "filtered_candidate_ids": [],
        "mandatory_candidate_ids": ["A"],
        "stage1a_selected_ids": [],
        "stage1b_selected_ids": [],
        "stage2_selected_ids": [],
        "mandatory_count_certified": False,
        "stage1b_ran": True,
    }
    assert solver._diagnose_customer_day_reason("A", day_status, set()) == "stage1a_timeout_or_conflict"


def test_extraction_mismatch_detected() -> None:
    instance = make_instance()
    solver = RollingHorizonCPSATSolver()
    status = {
        "stage1a_selected_ids": ["A"],
        "stage1b_selected_ids": [],
        "stage2_selected_ids": [],
        "mandatory_candidate_ids": ["A"],
        "mandatory_candidate_count": 1,
        "mandatory_delivered_count": 1,
        "mandatory_count_certified": True,
        "stage1a_status": "OPTIMAL",
        "stage1b_ran": False,
        "stage2_ran": False,
    }
    result = solver._check_extraction_consistency(instance, 1, ["A"], DailyRoute(day=1), status, set())
    assert result["extraction_consistency_error"] is True
    assert "omitted" in result["extraction_error_message"]


def test_customer_pipeline_sets_consistent() -> None:
    instance = make_instance()
    solver = RollingHorizonCPSATSolver()
    status = {
        "mandatory_candidate_ids": ["A"],
        "stage1a_selected_ids": [],
        "stage1b_selected_ids": [],
        "stage2_selected_ids": [],
        "stage1b_ran": False,
    }
    rows = solver._build_customer_day_diagnostics(
        instance,
        1,
        raw_candidates=["A", "B"],
        candidates=["A"],
        filtered_candidates=["B"],
        candidate_ranks={"A": 1},
        day_status=status,
        route=DailyRoute(day=1),
        delivered_so_far=set(),
        undelivered_after_day={"A", "B"},
    )
    by_id = {row["customer_id"]: row for row in rows}
    assert by_id["A"]["in_selected_candidates"] is True
    assert by_id["B"]["filtered_by_candidate_limit"] is True


def test_last_day_diagnostic_does_not_modify_schedule() -> None:
    instance = make_instance()
    solver = RollingHorizonCPSATSolver(time_limit_per_day_sec=1, num_workers=1)
    day_statuses = {1: {"extracted_route_ids": []}}
    before = dict(day_statuses)
    result = solver.diagnose_incomplete_customers_on_last_days(instance, day_statuses, [], time_limit_sec=1)
    assert result == []
    assert day_statuses == before
