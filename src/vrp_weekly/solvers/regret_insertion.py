"""Rolling-horizon regret insertion solver with simple local search."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from vrp_weekly.config import MONDAY, SUNDAY
from vrp_weekly.evaluator import evaluate_daily_route
from vrp_weekly.models import DailyRoute, Instance, WeeklySchedule
from vrp_weekly.solvers.base import Solver


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


class RegretInsertionSolver(Solver):
    """Rolling-horizon regret insertion plus local improvement."""

    name = "regret"

    def __init__(
        self,
        regret_weight: float = 1.0,
        insertion_weight: float = 1.0,
        urgency_weight: float = 100.0,
        waiting_weight: float = 0.2,
        seed: int | None = None,
        candidate_scan_limit: int = 80,
        local_search_passes: int = 3,
    ) -> None:
        """Initialize heuristic weights."""
        self.regret_weight = regret_weight
        self.insertion_weight = insertion_weight
        self.urgency_weight = urgency_weight
        self.waiting_weight = waiting_weight
        self.seed = seed
        self.candidate_scan_limit = candidate_scan_limit
        self.local_search_passes = local_search_passes

    def solve(self, instance: Instance) -> WeeklySchedule:
        """Construct weekly routes with rolling-horizon regret insertion."""
        undelivered = set(instance.customer_ids())
        routes: dict[int, DailyRoute] = {}

        for day in range(MONDAY, SUNDAY + 1):
            sequence: list[str] = []
            while True:
                day_candidates = _rank_day_candidates(instance, day, undelivered)
                scan_candidates = day_candidates[: self.candidate_scan_limit]
                insertion_options = self._find_insertion_options(instance, day, sequence, scan_candidates)

                if not insertion_options and len(day_candidates) > len(scan_candidates):
                    insertion_options = self._find_insertion_options(
                        instance,
                        day,
                        sequence,
                        day_candidates[self.candidate_scan_limit :],
                    )

                if not insertion_options:
                    break

                _, chosen_customer, chosen_candidate = max(insertion_options, key=lambda item: (item[0], item[1]))
                sequence.insert(chosen_candidate.position, chosen_customer)
                undelivered.remove(chosen_customer)

            improved_sequence = improve_daily_sequence(
                instance,
                day,
                sequence,
                waiting_weight=self.waiting_weight,
                max_passes=self.local_search_passes,
            )
            routes[day] = evaluate_daily_route(instance, day, improved_sequence)

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
                waiting_weight=self.waiting_weight,
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
    waiting_weight: float = 0.2,
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
        cost = _route_objective(trial_route, waiting_weight) - _route_objective(base_route, waiting_weight)
        options.append((cost, position, trial_route))

    if not options:
        return None

    options.sort(key=lambda item: (item[0], item[1]))
    best_cost, best_position, best_route = options[0]
    second_best_cost = options[1][0] if len(options) > 1 else best_cost
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
    waiting_weight: float = 0.2,
    max_passes: int | None = None,
) -> list[str]:
    """Apply first-improvement relocate, swap, and 2-opt moves within a daily route."""
    best_sequence = list(sequence)
    best_route = evaluate_daily_route(instance, day, best_sequence)
    if not best_route.hard_feasible:
        return best_sequence
    best_score = _route_objective(best_route, waiting_weight)

    improved = True
    passes = 0
    while improved:
        if max_passes is not None and passes >= max_passes:
            break
        passes += 1
        improved = False
        for candidate_sequence in _neighbor_sequences(best_sequence):
            route = evaluate_daily_route(instance, day, candidate_sequence)
            if not route.hard_feasible:
                continue
            score = _route_objective(route, waiting_weight)
            if score + 1e-9 < best_score:
                best_sequence = candidate_sequence
                best_score = score
                improved = True
                break

    return best_sequence


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


def customer_urgency(instance: Instance, day: int, customer_id: str) -> float:
    """Return urgency score based on remaining available days."""
    remaining_days = [available_day for available_day in instance.available_days(customer_id) if available_day >= day]
    if not remaining_days:
        return 0.0
    if len(remaining_days) == 1:
        return 2.0
    return 1.0 / len(remaining_days)


def _route_objective(route: DailyRoute, waiting_weight: float) -> float:
    """Return distance-plus-waiting objective for a route."""
    return route.route_distance_km + waiting_weight * route.route_waiting_time_min


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
