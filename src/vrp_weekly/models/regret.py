"""Rolling-horizon regret insertion solver with simple local search."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from vrp_weekly.config import (
    ENABLE_LOCAL_SEARCH,
    INSERTION_WEIGHT,
    MAX_LOCAL_SEARCH_ITERATIONS,
    MONDAY,
    REGRET_WEIGHT,
    SUNDAY,
    URGENCY_WEIGHT,
    WAITING_WEIGHT,
)
from vrp_weekly.evaluator import evaluate_daily_route
from vrp_weekly.core import DailyRoute, Instance, WeeklySchedule

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InsertionCandidate:
    """Best insertion result for one customer."""

    customer_id: str
    position: int
    best_cost: float
    second_best_cost: float
    route: DailyRoute

    @property
    def regret(self) -> float:
        """Return regret-2 value."""
        return self.second_best_cost - self.best_cost


class RegretInsertionSolver:
    """Rolling-horizon regret insertion plus local improvement."""

    name = "regret"

    def __init__(
        self,
        regret_weight: float = REGRET_WEIGHT,
        insertion_weight: float = INSERTION_WEIGHT,
        urgency_weight: float = URGENCY_WEIGHT,
        distance_weight: float = 10.0,
        waiting_weight: float = WAITING_WEIGHT,
        duration_weight: float = 0.0,
        single_insertion_regret_bonus: float = 100.0,
        seed: int | None = None,
        candidate_scan_limit: int | None = 80,
        enable_local_search: bool = ENABLE_LOCAL_SEARCH,
        max_local_search_iterations: int = MAX_LOCAL_SEARCH_ITERATIONS,
    ) -> None:
        """Initialize heuristic weights."""
        self.regret_weight = regret_weight
        self.insertion_weight = insertion_weight
        self.urgency_weight = urgency_weight
        self.distance_weight = distance_weight
        self.waiting_weight = waiting_weight
        self.duration_weight = duration_weight
        self.single_insertion_regret_bonus = single_insertion_regret_bonus
        self.seed = seed
        self.candidate_scan_limit = candidate_scan_limit
        self.enable_local_search = enable_local_search
        self.max_local_search_iterations = max_local_search_iterations

    def solve(self, instance: Instance) -> WeeklySchedule:
        """Construct weekly routes with rolling-horizon regret insertion."""
        undelivered = set(instance.customer_ids())
        routes: dict[int, DailyRoute] = {}

        for day in range(MONDAY, SUNDAY + 1):
            LOGGER.info("regret day=%s start undelivered=%s", day, len(undelivered))
            sequence: list[str] = []
            while True:
                day_candidates = _rank_day_candidates(instance, day, undelivered)
                scan_candidates = _select_scan_candidates(instance, day, day_candidates, self.candidate_scan_limit)
                insertion_options = self._find_insertion_options(instance, day, sequence, scan_candidates)

                if not insertion_options:
                    break

                _, chosen_customer, chosen_candidate = max(insertion_options, key=lambda item: (item[0], item[1]))
                sequence.insert(chosen_candidate.position, chosen_customer)
                undelivered.remove(chosen_customer)

            LOGGER.info("regret day=%s constructed=%s remaining=%s", day, len(sequence), len(undelivered))
            improved_sequence = sequence
            if self.enable_local_search:
                improved_sequence = improve_daily_sequence(
                    instance,
                    day,
                    sequence,
                    distance_weight=self.distance_weight,
                    waiting_weight=self.waiting_weight,
                    duration_weight=self.duration_weight,
                    max_iterations=self.max_local_search_iterations,
                )
                LOGGER.info("regret day=%s local_search_done stops=%s", day, len(improved_sequence))
            extra_sequence, inserted_customers = fill_extra_customers(
                instance,
                day,
                improved_sequence,
                undelivered,
                distance_weight=self.distance_weight,
                waiting_weight=self.waiting_weight,
                duration_weight=self.duration_weight,
                single_insertion_regret_bonus=self.single_insertion_regret_bonus,
            )
            for customer_id in inserted_customers:
                undelivered.remove(customer_id)
            if inserted_customers:
                LOGGER.info("regret day=%s fill_extra_inserted=%s", day, len(inserted_customers))
            if self.enable_local_search and inserted_customers:
                extra_sequence = improve_daily_sequence(
                    instance,
                    day,
                    extra_sequence,
                    distance_weight=self.distance_weight,
                    waiting_weight=self.waiting_weight,
                    duration_weight=self.duration_weight,
                    max_iterations=self.max_local_search_iterations,
                )
                LOGGER.info("regret day=%s final_local_search_done stops=%s", day, len(extra_sequence))
            improved_sequence = extra_sequence
            routes[day] = evaluate_daily_route(instance, day, improved_sequence)
            LOGGER.info(
                "regret day=%s done stops=%s return=%s distance=%.2f remaining=%s",
                day,
                len(routes[day].stops),
                routes[day].return_to_depot_time,
                routes[day].route_distance_km,
                len(undelivered),
            )

        return WeeklySchedule(routes=routes)

    def _find_insertion_options(
        self,
        instance: Instance,
        day: int,
        sequence: list[str],
        customers: Iterable[str],
    ) -> list[tuple[float, str, InsertionCandidate]]:
        """Return scored feasible insertions for a batch of customers."""
        insertion_options: list[tuple[float, str, InsertionCandidate]] = []
        base_route = evaluate_daily_route(instance, day, sequence)
        for customer_id in customers:
            candidate = best_feasible_insertion(
                instance,
                day,
                sequence,
                customer_id,
                distance_weight=self.distance_weight,
                waiting_weight=self.waiting_weight,
                duration_weight=self.duration_weight,
                single_insertion_regret_bonus=self.single_insertion_regret_bonus,
                base_route=base_route,
            )
            if candidate is None:
                continue
            urgency = customer_urgency(instance, day, customer_id)
            score = (
                self.regret_weight * candidate.regret
                - self.insertion_weight * candidate.best_cost
                + self.urgency_weight * urgency
            )
            insertion_options.append((score, customer_id, candidate))
        return insertion_options


def best_feasible_insertion(
    instance: Instance,
    day: int,
    current_sequence: list[str],
    customer: str,
    distance_weight: float = 10.0,
    waiting_weight: float = 0.2,
    duration_weight: float = 0.0,
    single_insertion_regret_bonus: float = 100.0,
    base_route: DailyRoute | None = None,
) -> InsertionCandidate | None:
    """Return the best feasible insertion for a customer into a daily sequence."""
    if base_route is None:
        base_route = evaluate_daily_route(instance, day, current_sequence)
    options: list[tuple[float, int, DailyRoute]] = []
    for position in range(len(current_sequence) + 1):
        trial_sequence = current_sequence[:position] + [customer] + current_sequence[position:]
        trial_route = evaluate_daily_route(instance, day, trial_sequence)
        if not trial_route.hard_feasible:
            continue
        cost = _route_objective(trial_route, distance_weight, waiting_weight, duration_weight) - _route_objective(
            base_route,
            distance_weight,
            waiting_weight,
            duration_weight,
        )
        options.append((cost, position, trial_route))

    if not options:
        return None

    options.sort(key=lambda item: (item[0], item[1]))
    best_cost, best_position, best_route = options[0]
    second_best_cost = options[1][0] if len(options) > 1 else best_cost + single_insertion_regret_bonus
    return InsertionCandidate(
        customer_id=customer,
        position=best_position,
        best_cost=best_cost,
        second_best_cost=second_best_cost,
        route=best_route,
    )


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
    """Apply first-improvement relocate, swap, and 2-opt moves within a daily route."""
    best_sequence = list(sequence)
    best_route = evaluate_daily_route(instance, day, best_sequence)
    if not best_route.hard_feasible:
        return best_sequence
    best_score = _route_objective(best_route, distance_weight, waiting_weight, duration_weight)

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
            route = evaluate_daily_route(instance, day, candidate_sequence)
            if not route.hard_feasible:
                continue
            score = _route_objective(route, distance_weight, waiting_weight, duration_weight)
            if score + 1e-9 < best_score:
                best_sequence = candidate_sequence
                best_score = score
                improved = True
                break

    return best_sequence


def fill_extra_customers(
    instance: Instance,
    day: int,
    sequence: list[str],
    undelivered: set[str],
    distance_weight: float = 10.0,
    waiting_weight: float = WAITING_WEIGHT,
    duration_weight: float = 0.0,
    single_insertion_regret_bonus: float = 100.0,
) -> tuple[list[str], list[str]]:
    """Greedily insert additional feasible same-day customers after improvement."""
    updated_sequence = list(sequence)
    inserted_customers: list[str] = []
    remaining = {customer_id for customer_id in undelivered if instance.windows_for_customer_day(customer_id, day)}

    while True:
        options: list[tuple[float, str, InsertionCandidate]] = []
        base_route = evaluate_daily_route(instance, day, updated_sequence)
        for customer_id in sorted(remaining):
            candidate = best_feasible_insertion(
                instance,
                day,
                updated_sequence,
                customer_id,
                distance_weight=distance_weight,
                waiting_weight=waiting_weight,
                duration_weight=duration_weight,
                single_insertion_regret_bonus=single_insertion_regret_bonus,
                base_route=base_route,
            )
            if candidate is not None:
                options.append((candidate.best_cost, customer_id, candidate))

        if not options:
            break

        _, chosen_customer, chosen_candidate = min(options, key=lambda item: (item[0], item[1]))
        updated_sequence.insert(chosen_candidate.position, chosen_customer)
        inserted_customers.append(chosen_customer)
        remaining.remove(chosen_customer)

    return updated_sequence, inserted_customers


def _rank_day_candidates(instance: Instance, day: int, customer_ids: set[str]) -> list[str]:
    """Rank available customers by urgency, deadline, then id for deterministic scanning."""
    ranked: list[tuple[float, int, str]] = []
    for customer_id in customer_ids:
        windows = instance.windows_for_customer_day(customer_id, day)
        if not windows:
            continue
        earliest_end = min(window.end_minute for window in windows)
        ranked.append((-customer_urgency(instance, day, customer_id), earliest_end, customer_id))
    return [customer_id for _, _, customer_id in sorted(ranked)]


def _select_scan_candidates(
    instance: Instance,
    day: int,
    day_candidates: list[str],
    candidate_scan_limit: int | None,
) -> list[str]:
    """Return candidate scan list, always including last-available-day customers."""
    if candidate_scan_limit is None:
        return list(day_candidates)
    selected = list(day_candidates[:candidate_scan_limit])
    selected_set = set(selected)
    for customer_id in day_candidates:
        remaining_days = [available_day for available_day in instance.available_days(customer_id) if available_day >= day]
        if len(remaining_days) == 1 and customer_id not in selected_set:
            selected.append(customer_id)
            selected_set.add(customer_id)
    return selected


def customer_urgency(instance: Instance, day: int, customer_id: str) -> float:
    """Return urgency score based on remaining available days."""
    remaining_days = [available_day for available_day in instance.available_days(customer_id) if available_day >= day]
    if not remaining_days:
        return 0.0
    day_pressure = 10.0 if len(remaining_days) == 1 else 1.0 / len(remaining_days)
    windows_today = instance.windows_for_customer_day(customer_id, day)
    if not windows_today:
        return day_pressure
    earliest_window_end_today = min(window.end_minute for window in windows_today)
    deadline_pressure = (1440 - earliest_window_end_today) / 1440
    return day_pressure + deadline_pressure


def _route_objective(
    route: DailyRoute,
    distance_weight: float,
    waiting_weight: float,
    duration_weight: float,
) -> float:
    """Return weighted route objective for insertion and local search."""
    return (
        distance_weight * route.route_distance_km
        + waiting_weight * route.route_waiting_time_min
        + duration_weight * route.route_duration_min
    )


def _neighbor_sequences(sequence: list[str]) -> Iterable[list[str]]:
    """Generate local-search neighbors in deterministic order."""
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
            candidate = sequence[:i] + list(reversed(sequence[i:j])) + sequence[j:]
            yield candidate

