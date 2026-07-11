"""Shared local-search improvement routines for heuristic routes."""

from __future__ import annotations

import time
from dataclasses import dataclass

from vrp_weekly.config import MONDAY, SUNDAY
from vrp_weekly.core import DailyRoute, Instance, WeeklySchedule
from vrp_weekly.evaluator import evaluate_daily_route, evaluate_weekly_schedule
from vrp_weekly.heuristics.route_eval import (
    HeuristicWeights,
    best_feasible_insertion,
    route_customer_ids,
    route_secondary_cost,
    validate_no_duplicates,
    windows_for,
)


@dataclass
class LocalSearchParams:
    """Configuration for intra-day and optional inter-day local search."""

    enable_relocate: bool = True
    enable_swap: bool = True
    enable_two_opt: bool = True
    enable_remove_reinsert: bool = True
    enable_post_fill: bool = True
    enable_inter_day_move: bool = False
    enable_inter_day_swap: bool = False
    max_iterations: int = 100
    time_limit_sec: int = 10
    distance_weight: float = 10.0
    waiting_weight: float = 1.0
    duration_weight: float = 0.0


def improve_daily_route(
    instance: Instance,
    day: int,
    route: DailyRoute,
    undelivered_today: list[str] | None = None,
    params: LocalSearchParams | None = None,
) -> DailyRoute:
    """Improve one daily route with feasible VNS-style moves."""
    params = LocalSearchParams() if params is None else params
    weights = HeuristicWeights(params.distance_weight, params.waiting_weight, params.duration_weight)
    deadline = time.perf_counter() + max(0.0, params.time_limit_sec)
    current = route if route.hard_feasible else evaluate_daily_route(instance, day, route_customer_ids(route))
    iterations = 0

    while iterations < params.max_iterations and time.perf_counter() <= deadline:
        iterations += 1
        current_sequence = route_customer_ids(current)
        accepted = False
        for candidate_sequence in _neighbor_sequences(current_sequence, params):
            if time.perf_counter() > deadline:
                break
            if len(candidate_sequence) != len(current_sequence) or set(candidate_sequence) != set(current_sequence):
                continue
            candidate_route = evaluate_daily_route(instance, day, candidate_sequence)
            if _route_improves(candidate_route, current, weights, allow_more_delivered=False):
                current = candidate_route
                accepted = True
                break
        if not accepted:
            break

    if params.enable_post_fill and undelivered_today:
        remaining = [customer for customer in sorted(set(undelivered_today)) if customer not in set(route_customer_ids(current))]
        while time.perf_counter() <= deadline:
            base_sequence = route_customer_ids(current)
            options = [
                insertion
                for customer in remaining
                if windows_for(instance, customer, day)
                for insertion in [best_feasible_insertion(instance, day, base_sequence, customer, base_route=current, weights=weights)]
                if insertion is not None
            ]
            if not options:
                break
            best = min(
                options,
                key=lambda option: (
                    option.incremental_cost,
                    option.selected_window_end if option.selected_window_end is not None else 10**9,
                    option.customer_id,
                ),
            )
            candidate_route = best.route
            if not _route_improves(candidate_route, current, weights, allow_more_delivered=True):
                break
            current = candidate_route
            remaining = [customer for customer in remaining if customer != best.customer_id]

    return current


def improve_weekly_schedule(
    instance: Instance,
    schedule: WeeklySchedule,
    undelivered: set[str] | None = None,
    params: LocalSearchParams | None = None,
) -> WeeklySchedule:
    """Improve a weekly schedule by improving each daily route independently."""
    params = LocalSearchParams() if params is None else params
    delivered = schedule.delivered_customer_ids()
    remaining = set(instance.customer_ids()) - delivered if undelivered is None else set(undelivered)
    routes = dict(schedule.routes)

    for day in range(MONDAY, SUNDAY + 1):
        route = routes.get(day, DailyRoute(day=day))
        today_candidates = [customer for customer in remaining if windows_for(instance, customer, day)]
        improved = improve_daily_route(instance, day, route, undelivered_today=today_candidates, params=params)
        old_delivered = set(route_customer_ids(route))
        new_delivered = set(route_customer_ids(improved))
        remaining |= old_delivered - new_delivered
        remaining -= new_delivered
        routes[day] = improved

    improved_schedule = WeeklySchedule(
        routes=routes,
        solver_status={
            **schedule.solver_status,
            "local_search_enabled": True,
            "local_search_iterations": params.max_iterations,
            "local_search_time_limit_sec": params.time_limit_sec,
        },
    )
    if params.enable_inter_day_move or params.enable_inter_day_swap:
        improved_schedule = _improve_inter_day(instance, improved_schedule, params)
    return improved_schedule


def _neighbor_sequences(sequence: list[str], params: LocalSearchParams) -> list[list[str]]:
    """Generate deterministic intra-day neighbors."""
    neighbors: list[list[str]] = []
    n = len(sequence)
    if params.enable_relocate:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                candidate = list(sequence)
                customer = candidate.pop(i)
                candidate.insert(j, customer)
                neighbors.append(candidate)
    if params.enable_swap:
        for i in range(n):
            for j in range(i + 1, n):
                candidate = list(sequence)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                neighbors.append(candidate)
    if params.enable_two_opt:
        for i in range(n):
            for j in range(i + 2, n + 1):
                neighbors.append(sequence[:i] + list(reversed(sequence[i:j])) + sequence[j:])
    if params.enable_remove_reinsert:
        for i, customer in enumerate(sequence):
            shorter = sequence[:i] + sequence[i + 1 :]
            for j in range(len(shorter) + 1):
                neighbors.append(shorter[:j] + [customer] + shorter[j:])
    return neighbors


def _route_improves(candidate: DailyRoute, current: DailyRoute, weights: HeuristicWeights, allow_more_delivered: bool) -> bool:
    """Return true if candidate improves delivered count or official route cost."""
    if not candidate.hard_feasible:
        return False
    candidate_ids = route_customer_ids(candidate)
    current_ids = route_customer_ids(current)
    if len(candidate_ids) != len(set(candidate_ids)):
        return False
    if not allow_more_delivered and set(candidate_ids) != set(current_ids):
        return False
    if len(candidate_ids) > len(current_ids):
        return True
    if len(candidate_ids) < len(current_ids):
        return False
    return (
        route_secondary_cost(candidate, weights),
        candidate.route_waiting_time_min,
    ) < (
        route_secondary_cost(current, weights),
        current.route_waiting_time_min,
    )


def _improve_inter_day(instance: Instance, schedule: WeeklySchedule, params: LocalSearchParams) -> WeeklySchedule:
    """Optionally attempt simple inter-day moves/swaps when enabled."""
    best = schedule
    best_metrics = evaluate_weekly_schedule(instance, best)
    deadline = time.perf_counter() + max(0.0, params.time_limit_sec)
    routes = dict(best.routes)

    if params.enable_inter_day_move:
        for from_day in range(MONDAY, SUNDAY + 1):
            for to_day in range(MONDAY, SUNDAY + 1):
                if from_day == to_day or time.perf_counter() > deadline:
                    continue
                from_seq = route_customer_ids(routes.get(from_day, DailyRoute(day=from_day)))
                to_seq = route_customer_ids(routes.get(to_day, DailyRoute(day=to_day)))
                for customer in list(from_seq):
                    if not windows_for(instance, customer, to_day):
                        continue
                    new_from = [item for item in from_seq if item != customer]
                    insertion = best_feasible_insertion(instance, to_day, to_seq, customer)
                    if insertion is None:
                        continue
                    candidate_routes = dict(routes)
                    candidate_routes[from_day] = evaluate_daily_route(instance, from_day, new_from)
                    candidate_routes[to_day] = insertion.route
                    candidate = WeeklySchedule(routes=candidate_routes, solver_status=best.solver_status)
                    candidate_metrics = evaluate_weekly_schedule(instance, candidate)
                    candidate_key = (
                        candidate_metrics.objective_value,
                        candidate_metrics.incomplete_count,
                        candidate_metrics.total_deferral_days,
                        candidate_metrics.total_distance_km,
                        candidate_metrics.total_waiting_time_min,
                    )
                    best_key = (
                        best_metrics.objective_value,
                        best_metrics.incomplete_count,
                        best_metrics.total_deferral_days,
                        best_metrics.total_distance_km,
                        best_metrics.total_waiting_time_min,
                    )
                    if validate_no_duplicates(candidate) and candidate_key < best_key:
                        best = candidate
                        best_metrics = candidate_metrics
                        routes = dict(best.routes)
    return best
