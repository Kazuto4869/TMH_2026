"""Shared route evaluation and insertion utilities for heuristic models."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vrp_weekly.config import DEFAULT_SERVICE_TIME_MIN, MAX_SPEED_KMPH, MINUTES_PER_HOUR
from vrp_weekly.core import DailyRoute, Instance, TimeWindow, WeeklySchedule
from vrp_weekly.distance import euclidean_distance_km
from vrp_weekly.evaluator import evaluate_daily_route, evaluate_weekly_schedule, validate_schedule


@dataclass(frozen=True)
class InsertionOption:
    """A feasible insertion of one customer into one daily route."""

    customer_id: str
    position: int
    sequence: list[str]
    route: DailyRoute
    incremental_cost: float
    selected_window_start: int | None = None
    selected_window_end: int | None = None


@dataclass(frozen=True)
class HeuristicWeights:
    """Secondary route-cost weights for constructive heuristics."""

    distance_weight: float = 10.0
    waiting_weight: float = 1.0
    duration_weight: float = 1.0


def windows_for(instance: Instance, customer_id: str, day: int) -> list[TimeWindow]:
    """Return sorted windows for one customer and day."""
    return sorted(instance.windows_for_customer_day(customer_id, day), key=lambda window: (window.start_minute, window.end_minute))


def get_service_time(instance: Instance, customer_id: str) -> int:
    """Return service time for a customer, with zero service at depots."""
    location = instance.locations[customer_id]
    if location.is_depot or customer_id == instance.depot_id:
        return 0
    return location.service_time if location.service_time > 0 else DEFAULT_SERVICE_TIME_MIN


def distance_km(instance: Instance, i: str, j: str) -> float:
    """Return Euclidean distance between two instance locations in km."""
    return euclidean_distance_km(instance.locations[i], instance.locations[j])


def travel_time_minutes(instance: Instance, i: str, j: str) -> int:
    """Return ceiling travel time in minutes at the configured max speed."""
    return int(math.ceil(MINUTES_PER_HOUR * distance_km(instance, i, j) / MAX_SPEED_KMPH))


def route_customer_ids(route: DailyRoute) -> list[str]:
    """Return customer ids from route stops, excluding depot ids if present."""
    return [stop.customer_id for stop in route.stops if stop.customer_id]


def validate_no_duplicates(schedule: WeeklySchedule) -> bool:
    """Return true iff no customer appears more than once in the weekly schedule."""
    seen: set[str] = set()
    for route in schedule.routes.values():
        for customer_id in route_customer_ids(route):
            if customer_id in seen:
                return False
            seen.add(customer_id)
    return True


def route_secondary_cost(route: DailyRoute, weights: HeuristicWeights | None = None) -> float:
    """Return weighted secondary route cost."""
    weights = HeuristicWeights() if weights is None else weights
    return (
        weights.distance_weight * route.route_distance_km
        + weights.waiting_weight * route.route_waiting_time_min
        + weights.duration_weight * route.route_duration_min
    )


def weekly_score(instance: Instance, schedule: WeeklySchedule) -> float:
    """Return lexicographic-priority numeric score for a weekly schedule."""
    metrics = evaluate_weekly_schedule(instance, schedule)
    return (
        1_000_000 * metrics.incomplete_count
        + 10_000 * metrics.total_deferral_days
        + 10 * metrics.total_distance_km
        + metrics.total_waiting_time_min
        + metrics.total_route_duration_min
    )


def best_feasible_insertion(
    instance: Instance,
    day: int,
    sequence: list[str],
    customer_id: str,
    base_route: DailyRoute | None = None,
    weights: HeuristicWeights | None = None,
) -> InsertionOption | None:
    """Return the cheapest feasible insertion for one customer."""
    options = all_feasible_insertions(instance, day, sequence, customer_id, base_route=base_route, weights=weights)
    return options[0] if options else None


def all_feasible_insertions(
    instance: Instance,
    day: int,
    sequence: list[str],
    customer_id: str,
    base_route: DailyRoute | None = None,
    weights: HeuristicWeights | None = None,
) -> list[InsertionOption]:
    """Return all feasible insertions for one customer sorted by secondary cost."""
    if customer_id in sequence:
        return []
    weights = HeuristicWeights() if weights is None else weights
    base_route = evaluate_daily_route(instance, day, sequence) if base_route is None else base_route
    base_cost = route_secondary_cost(base_route, weights)
    options: list[InsertionOption] = []

    for position in range(len(sequence) + 1):
        trial_sequence = sequence[:position] + [customer_id] + sequence[position:]
        trial_route = evaluate_daily_route(instance, day, trial_sequence)
        if not trial_route.hard_feasible:
            continue
        inserted_stop = next((stop for stop in trial_route.stops if stop.customer_id == customer_id), None)
        if inserted_stop is None or inserted_stop.selected_time_window is None:
            continue
        window = inserted_stop.selected_time_window
        options.append(
            InsertionOption(
                customer_id=customer_id,
                position=position,
                sequence=trial_sequence,
                route=trial_route,
                incremental_cost=route_secondary_cost(trial_route, weights) - base_cost,
                selected_window_start=window.start_minute,
                selected_window_end=window.end_minute,
            )
        )

    return sorted(
        options,
        key=lambda option: (
            option.incremental_cost,
            option.selected_window_end if option.selected_window_end is not None else 10**9,
            option.selected_window_start if option.selected_window_start is not None else 10**9,
            option.customer_id,
        ),
    )


def schedule_hard_feasible(instance: Instance, schedule: WeeklySchedule) -> bool:
    """Return true iff central schedule validation has no violations."""
    return not validate_schedule(instance, schedule)

