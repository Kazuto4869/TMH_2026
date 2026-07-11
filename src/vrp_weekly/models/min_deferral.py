"""Minimum-deferral baseline solver for the weekly routing problem."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from vrp_weekly.config import ENABLE_LOCAL_SEARCH, MAX_LOCAL_SEARCH_ITERATIONS, MONDAY, SUNDAY, WAITING_WEIGHT
from vrp_weekly.core import DailyRoute, Instance, WeeklySchedule
from vrp_weekly.evaluator import evaluate_daily_route, evaluate_weekly_schedule, official_objective_status

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InsertionCandidate:
    """Best feasible insertion result for one customer."""

    customer_id: str
    position: int
    trial_route: DailyRoute
    incremental_cost: float
    selected_window_end: int
    selected_window_start: int


class MinDeferralSolver:
    """Baseline that prioritizes avoiding customer deferral before route cost.

    This baseline tries to deliver as many same-day customers as possible before
    considering distance. It builds one route per day by repeatedly inserting an
    undelivered same-day customer into any feasible position, prioritizing
    last-chance and low-flexibility customers first. Local search is secondary:
    it may reorder the same customers to reduce route cost, but it never removes
    deliveries.
    """

    name = "min_deferral"
    display_name = "Min Deferral"

    def __init__(
        self,
        distance_weight: float = 10.0,
        waiting_weight: float = WAITING_WEIGHT,
        duration_weight: float = 0.0,
        enable_local_search: bool = ENABLE_LOCAL_SEARCH,
        max_local_search_iterations: int = MAX_LOCAL_SEARCH_ITERATIONS,
    ) -> None:
        """Initialize secondary route-cost weights."""
        self.distance_weight = distance_weight
        self.waiting_weight = waiting_weight
        self.duration_weight = duration_weight
        self.enable_local_search = enable_local_search
        self.max_local_search_iterations = max_local_search_iterations

    def solve(self, instance: Instance) -> WeeklySchedule:
        """Construct weekly routes by minimizing avoidable deferral first."""
        undelivered = set(instance.customer_ids())
        routes: dict[int, DailyRoute] = {}

        for day in range(MONDAY, SUNDAY + 1):
            LOGGER.info("min_deferral day=%s start undelivered=%s", day, len(undelivered))
            sequence: list[str] = []
            remaining_today = {customer_id for customer_id in undelivered if instance.windows_for_customer_day(customer_id, day)}

            sequence, inserted_customers = fill_extra_customers(
                instance,
                day,
                sequence,
                remaining_today,
                distance_weight=self.distance_weight,
                waiting_weight=self.waiting_weight,
                duration_weight=self.duration_weight,
            )
            undelivered -= set(inserted_customers)

            if self.enable_local_search:
                sequence = improve_daily_sequence(
                    instance,
                    day,
                    sequence,
                    distance_weight=self.distance_weight,
                    waiting_weight=self.waiting_weight,
                    duration_weight=self.duration_weight,
                    max_iterations=self.max_local_search_iterations,
                )

            remaining_today = {customer_id for customer_id in undelivered if instance.windows_for_customer_day(customer_id, day)}
            sequence, inserted_customers = fill_extra_customers(
                instance,
                day,
                sequence,
                remaining_today,
                distance_weight=self.distance_weight,
                waiting_weight=self.waiting_weight,
                duration_weight=self.duration_weight,
            )
            undelivered -= set(inserted_customers)

            if self.enable_local_search and inserted_customers:
                sequence = improve_daily_sequence(
                    instance,
                    day,
                    sequence,
                    distance_weight=self.distance_weight,
                    waiting_weight=self.waiting_weight,
                    duration_weight=self.duration_weight,
                    max_iterations=self.max_local_search_iterations,
                )

            routes[day] = evaluate_daily_route(instance, day, sequence)
            LOGGER.info(
                "min_deferral day=%s done stops=%s return=%s distance=%.2f remaining=%s",
                day,
                len(routes[day].stops),
                routes[day].return_to_depot_time,
                routes[day].route_distance_km,
                len(undelivered),
            )

        schedule = WeeklySchedule(routes=routes)
        metrics = evaluate_weekly_schedule(instance, schedule)
        status = {
            "solver": self.name,
            "status": "HEURISTIC_FEASIBLE" if metrics.hard_feasible else "HEURISTIC_INFEASIBLE",
            "gap_percent": "",
            **official_objective_status(metrics),
        }
        return WeeklySchedule(routes=routes, solver_status=status)


def remaining_available_days(instance: Instance, customer_id: str, day: int) -> list[int]:
    """Return days from today onward where the customer has a time window."""
    return sorted(available_day for available_day in instance.available_days(customer_id) if available_day >= day)


def is_last_available_day(instance: Instance, customer_id: str, day: int) -> bool:
    """Return true if today is the customer's last remaining delivery day."""
    return remaining_available_days(instance, customer_id, day) == [day]


def total_window_width_today(instance: Instance, customer_id: str, day: int) -> int:
    """Return the total width in minutes of all of today's windows."""
    return sum(window.end_minute - window.start_minute for window in instance.windows_for_customer_day(customer_id, day))


def earliest_window_end_today(instance: Instance, customer_id: str, day: int) -> int:
    """Return the earliest window end for the customer today."""
    windows = instance.windows_for_customer_day(customer_id, day)
    return min((window.end_minute for window in windows), default=24 * 60)


def deferral_priority(instance: Instance, day: int, customer_id: str) -> tuple[int, int, int, int, str]:
    """Return a deterministic lexicographic deferral priority tuple."""
    remaining_days = remaining_available_days(instance, customer_id, day)
    return (
        0 if is_last_available_day(instance, customer_id, day) else 1,
        len(remaining_days),
        total_window_width_today(instance, customer_id, day),
        earliest_window_end_today(instance, customer_id, day),
        customer_id,
    )


def deferral_priority_score(instance: Instance, day: int, customer_id: str) -> int:
    """Return a numeric score where higher means more important to deliver today."""
    remaining_days_count = max(1, len(remaining_available_days(instance, customer_id, day)))
    if is_last_available_day(instance, customer_id, day):
        base = 1_000_000
    else:
        base = 100_000 // remaining_days_count

    narrow_bonus = 10_000 // max(1, total_window_width_today(instance, customer_id, day))
    deadline_bonus = max(0, 24 * 60 - earliest_window_end_today(instance, customer_id, day))
    return base + narrow_bonus + deadline_bonus


def best_feasible_insertion_for_customer(
    instance: Instance,
    day: int,
    sequence: list[str],
    customer: str,
    base_route: DailyRoute | None = None,
    distance_weight: float = 10.0,
    waiting_weight: float = WAITING_WEIGHT,
    duration_weight: float = 0.0,
) -> InsertionCandidate | None:
    """Return the cheapest feasible insertion of one customer into a route."""
    if base_route is None:
        base_route = evaluate_daily_route(instance, day, sequence)
    base_cost = route_objective(base_route, distance_weight, waiting_weight, duration_weight)

    best: InsertionCandidate | None = None
    for position in range(len(sequence) + 1):
        trial_sequence = sequence[:position] + [customer] + sequence[position:]
        trial_route = evaluate_daily_route(instance, day, trial_sequence)
        if not trial_route.hard_feasible:
            continue

        inserted_stop = next((stop for stop in trial_route.stops if stop.customer_id == customer), None)
        if inserted_stop is None or inserted_stop.selected_time_window is None:
            continue

        incremental_cost = route_objective(trial_route, distance_weight, waiting_weight, duration_weight) - base_cost
        candidate = InsertionCandidate(
            customer_id=customer,
            position=position,
            trial_route=trial_route,
            incremental_cost=incremental_cost,
            selected_window_end=inserted_stop.selected_time_window.end_minute,
            selected_window_start=inserted_stop.selected_time_window.start_minute,
        )
        if best is None or _insertion_cost_key(candidate) < _insertion_cost_key(best):
            best = candidate

    return best


def fill_extra_customers(
    instance: Instance,
    day: int,
    sequence: list[str],
    customers: Iterable[str],
    distance_weight: float = 10.0,
    waiting_weight: float = WAITING_WEIGHT,
    duration_weight: float = 0.0,
) -> tuple[list[str], list[str]]:
    """Insert feasible same-day customers using deferral priority first."""
    updated_sequence = list(sequence)
    remaining = set(customers) - set(updated_sequence)
    inserted_customers: list[str] = []

    while True:
        base_route = evaluate_daily_route(instance, day, updated_sequence)
        options: list[tuple[tuple[float, float, int, str], InsertionCandidate]] = []
        for customer_id in sorted(remaining):
            insertion = best_feasible_insertion_for_customer(
                instance,
                day,
                updated_sequence,
                customer_id,
                base_route=base_route,
                distance_weight=distance_weight,
                waiting_weight=waiting_weight,
                duration_weight=duration_weight,
            )
            if insertion is None:
                continue
            options.append((_selection_key(instance, day, insertion), insertion))

        if not options:
            break

        _, chosen = min(options, key=lambda item: item[0])
        updated_sequence.insert(chosen.position, chosen.customer_id)
        inserted_customers.append(chosen.customer_id)
        remaining.remove(chosen.customer_id)

    return updated_sequence, inserted_customers


def improve_daily_sequence(
    instance: Instance,
    day: int,
    sequence: list[str],
    distance_weight: float = 10.0,
    waiting_weight: float = WAITING_WEIGHT,
    duration_weight: float = 0.0,
    max_passes: int | None = None,
    max_iterations: int | None = MAX_LOCAL_SEARCH_ITERATIONS,
) -> list[str]:
    """Reduce route cost with relocate, swap, and 2-opt without removing customers."""
    best_sequence = list(sequence)
    best_route = evaluate_daily_route(instance, day, best_sequence)
    if not best_route.hard_feasible:
        return best_sequence
    best_score = route_objective(best_route, distance_weight, waiting_weight, duration_weight)

    improved = True
    passes = 0
    iterations = 0
    while improved:
        if max_passes is not None and passes >= max_passes:
            break
        passes += 1
        improved = False
        for candidate_sequence in _neighbor_sequences(best_sequence):
            if max_iterations is not None and iterations >= max_iterations:
                return best_sequence
            iterations += 1
            if set(candidate_sequence) != set(best_sequence) or len(candidate_sequence) != len(best_sequence):
                continue
            route = evaluate_daily_route(instance, day, candidate_sequence)
            if not route.hard_feasible:
                continue
            score = route_objective(route, distance_weight, waiting_weight, duration_weight)
            if score + 1e-9 < best_score:
                best_sequence = candidate_sequence
                best_score = score
                improved = True
                break

    return best_sequence


def route_objective(
    route: DailyRoute,
    distance_weight: float = 10.0,
    waiting_weight: float = WAITING_WEIGHT,
    duration_weight: float = 0.0,
) -> float:
    """Return secondary weighted route cost."""
    del distance_weight, waiting_weight, duration_weight
    return 10.0 * route.route_distance_km + route.route_waiting_time_min


def _selection_key(instance: Instance, day: int, insertion: InsertionCandidate) -> tuple[float, float, int, str]:
    """Return a min-sort key matching the solver's deterministic priorities."""
    return (
        -deferral_priority_score(instance, day, insertion.customer_id),
        insertion.incremental_cost,
        insertion.selected_window_end,
        insertion.customer_id,
    )


def _insertion_cost_key(insertion: InsertionCandidate) -> tuple[float, int, int, int, str]:
    """Return a deterministic key for best insertion of one customer."""
    return (
        insertion.incremental_cost,
        insertion.selected_window_end,
        insertion.selected_window_start,
        insertion.position,
        insertion.customer_id,
    )


def _neighbor_sequences(sequence: list[str]) -> Iterable[list[str]]:
    """Generate relocate, swap, and 2-opt neighbors in deterministic order."""
    n = len(sequence)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            candidate = list(sequence)
            customer = candidate.pop(i)
            candidate.insert(j, customer)
            yield candidate

    for i in range(n):
        for j in range(i + 1, n):
            candidate = list(sequence)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            yield candidate

    for i in range(n):
        for j in range(i + 2, n + 1):
            yield sequence[:i] + list(reversed(sequence[i:j])) + sequence[j:]
