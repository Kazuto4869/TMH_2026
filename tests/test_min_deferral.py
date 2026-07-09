from __future__ import annotations

from collections import Counter

from vrp_weekly.core import Instance, Location, TimeWindow
from vrp_weekly.evaluator import evaluate_daily_route, evaluate_weekly_schedule
from vrp_weekly.models.min_deferral import (
    MinDeferralSolver,
    best_feasible_insertion_for_customer,
    deferral_priority_score,
    fill_extra_customers,
    improve_daily_sequence,
    route_objective,
)
from vrp_weekly.models.nearest import NearestNeighborSolver


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


def delivered_sequence(schedule, day: int) -> list[str]:
    return [stop.customer_id for stop in schedule.routes[day].stops]


def delivered_counts(schedule) -> Counter[str]:
    return Counter(stop.customer_id for route in schedule.routes.values() for stop in route.stops)


def insertion_instance() -> Instance:
    return make_instance(
        {"A": (10.0, 0.0), "B": (20.0, 0.0), "C": (30.0, 0.0)},
        {
            "A": [(1, 480, 540)],
            "B": [(1, 540, 600), (2, 540, 600)],
            "C": [(1, 600, 660)],
        },
    )


def test_last_available_day_customer_is_chosen_first() -> None:
    instance = make_instance(
        {"LAST": (10.0, 0.0), "FUTURE": (5.0, 0.0)},
        {"LAST": [(1, 480, 540)], "FUTURE": [(1, 600, 700), (2, 600, 700)]},
    )

    schedule = MinDeferralSolver(enable_local_search=False).solve(instance)

    assert delivered_sequence(schedule, 1)[0] == "LAST"


def test_fewer_remaining_days_beats_more_remaining_days() -> None:
    instance = make_instance(
        {"TWO_DAYS": (10.0, 0.0), "THREE_DAYS": (5.0, 0.0)},
        {
            "TWO_DAYS": [(1, 480, 540), (2, 480, 540)],
            "THREE_DAYS": [(1, 600, 700), (2, 600, 700), (3, 600, 700)],
        },
    )

    assert deferral_priority_score(instance, 1, "TWO_DAYS") > deferral_priority_score(instance, 1, "THREE_DAYS")
    schedule = MinDeferralSolver(enable_local_search=False).solve(instance)

    assert delivered_sequence(schedule, 1)[0] == "TWO_DAYS"


def test_narrower_total_window_beats_wider_window_when_remaining_days_tie() -> None:
    instance = make_instance(
        {"NARROW": (10.0, 0.0), "WIDE": (5.0, 0.0)},
        {
            "NARROW": [(1, 640, 700), (2, 640, 700)],
            "WIDE": [(1, 400, 700), (2, 400, 700)],
        },
    )

    assert deferral_priority_score(instance, 1, "NARROW") > deferral_priority_score(instance, 1, "WIDE")
    schedule = MinDeferralSolver(enable_local_search=False).solve(instance)

    assert delivered_sequence(schedule, 1)[0] == "NARROW"


def test_best_insertion_can_insert_into_middle() -> None:
    instance = insertion_instance()

    insertion = best_feasible_insertion_for_customer(instance, 1, ["A", "C"], "B")

    assert insertion is not None
    assert insertion.position == 1
    assert [stop.customer_id for stop in insertion.trial_route.stops] == ["A", "B", "C"]


def test_min_deferral_delivers_at_least_as_many_customers_as_nearest() -> None:
    instance = insertion_instance()

    nearest_schedule = NearestNeighborSolver().solve(instance)
    min_deferral_schedule = MinDeferralSolver(enable_local_search=False).solve(instance)

    assert len(min_deferral_schedule.delivered_customer_ids()) >= len(nearest_schedule.delivered_customer_ids())


def test_min_deferral_produces_hard_feasible_schedule() -> None:
    instance = insertion_instance()

    schedule = MinDeferralSolver().solve(instance)
    metrics = evaluate_weekly_schedule(instance, schedule)

    assert metrics.hard_feasible


def test_min_deferral_has_no_duplicate_delivery() -> None:
    instance = make_instance(
        {"A": (10.0, 0.0), "B": (20.0, 0.0)},
        {"A": [(1, 480, 700), (2, 480, 700)], "B": [(2, 480, 700)]},
    )

    schedule = MinDeferralSolver().solve(instance)

    assert delivered_counts(schedule)["A"] == 1


def test_impossible_customer_remains_incomplete_after_sunday() -> None:
    instance = make_instance(
        {"IMPOSSIBLE": (1000.0, 0.0)},
        {"IMPOSSIBLE": [(7, 480, 500)]},
    )

    schedule = MinDeferralSolver().solve(instance)
    metrics = evaluate_weekly_schedule(instance, schedule)

    assert metrics.incomplete_count == 1
    assert "IMPOSSIBLE" not in schedule.delivered_customer_ids()


def test_local_search_never_removes_customers_and_preserves_feasibility() -> None:
    instance = make_instance(
        {"A": (10.0, 0.0), "B": (20.0, 0.0), "C": (30.0, 0.0)},
        {"A": [(1, 480, 900)], "B": [(1, 480, 900)], "C": [(1, 480, 900)]},
    )
    sequence = ["C", "B", "A"]
    before = evaluate_daily_route(instance, 1, sequence)

    improved = improve_daily_sequence(instance, 1, sequence)
    after = evaluate_daily_route(instance, 1, improved)

    assert set(improved) == set(sequence)
    assert len(improved) == len(sequence)
    assert after.hard_feasible
    assert route_objective(after) <= route_objective(before) + 1e-9


def test_post_fill_inserts_extra_feasible_customer() -> None:
    instance = insertion_instance()

    sequence, inserted = fill_extra_customers(instance, 1, ["A", "C"], {"B"})

    assert inserted == ["B"]
    assert sequence == ["A", "B", "C"]


def test_post_fill_does_not_insert_infeasible_customer() -> None:
    instance = make_instance(
        {"IMPOSSIBLE": (1000.0, 0.0)},
        {"IMPOSSIBLE": [(1, 480, 500)]},
    )

    sequence, inserted = fill_extra_customers(instance, 1, [], {"IMPOSSIBLE"})

    assert sequence == []
    assert inserted == []


def test_post_fill_has_no_duplicate_customers() -> None:
    instance = insertion_instance()

    sequence, inserted = fill_extra_customers(instance, 1, ["A"], {"A", "B"})

    assert inserted == ["B"]
    assert Counter(sequence)["A"] == 1
    assert Counter(sequence)["B"] == 1
