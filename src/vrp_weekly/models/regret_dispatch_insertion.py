"""Dispatch/defer regret feasible insertion heuristic."""

from __future__ import annotations

import math
import time

from vrp_weekly.config import DAY_END_MIN, MONDAY, SUNDAY
from vrp_weekly.core import DailyRoute, Instance, WeeklySchedule
from vrp_weekly.evaluator import evaluate_daily_route, evaluate_weekly_schedule
from vrp_weekly.heuristics.local_search import LocalSearchParams, improve_daily_route
from vrp_weekly.heuristics.route_eval import (
    HeuristicWeights,
    InsertionOption,
    all_feasible_insertions,
    validate_no_duplicates,
    windows_for,
)
from vrp_weekly.heuristics.scoring import (
    deadline_pressure,
    is_last_available_day,
    remaining_available_days,
    total_window_width_today,
    window_width_loss_to_future,
)


class RegretDispatchInsertionSolver:
    """Construct routes by balancing dispatch urgency and insertion regret."""

    name = "regret_dispatch"

    def __init__(
        self,
        use_local_search: bool = False,
        local_search_time_limit_sec: int = 10,
        local_search_max_iterations: int = 100,
        max_candidates_per_day: int | None = None,
        regret_weight: float = 1.0,
        defer_risk_weight: float = 1.0,
        insertion_cost_weight: float = 1.0,
        distance_weight: float = 10.0,
        waiting_weight: float = 1.0,
        duration_weight: float = 1.0,
        random_seed: int = 1,
    ) -> None:
        """Initialize heuristic parameters."""
        self.use_local_search = use_local_search
        self.local_search_time_limit_sec = local_search_time_limit_sec
        self.local_search_max_iterations = local_search_max_iterations
        self.max_candidates_per_day = max_candidates_per_day
        self.regret_weight = regret_weight
        self.defer_risk_weight = defer_risk_weight
        self.insertion_cost_weight = insertion_cost_weight
        self.distance_weight = distance_weight
        self.waiting_weight = waiting_weight
        self.duration_weight = duration_weight
        self.random_seed = random_seed

    def solve(self, instance: Instance) -> WeeklySchedule:
        """Build a weekly schedule with regret dispatch insertion."""
        start = time.perf_counter()
        undelivered = set(instance.customer_ids())
        routes: dict[int, DailyRoute] = {}
        day_statuses: dict[int, dict[str, object]] = {}
        weights = HeuristicWeights(self.distance_weight, self.waiting_weight, self.duration_weight)

        for day in range(MONDAY, SUNDAY + 1):
            raw_candidates = sorted(customer for customer in undelivered if windows_for(instance, customer, day))
            selected = self._select_candidates(instance, day, raw_candidates)
            sequence: list[str] = []
            current_route = evaluate_daily_route(instance, day, sequence)
            inserted_today: list[str] = []

            while True:
                options = []
                for customer in selected:
                    if customer not in undelivered:
                        continue
                    insertions = all_feasible_insertions(instance, day, sequence, customer, base_route=current_route, weights=weights)
                    if not insertions:
                        continue
                    best = insertions[0]
                    best_cost = best.incremental_cost
                    ins_regret = insertion_regret_score(insertions)
                    defer_risk = defer_risk_score(instance, day, customer, selected)
                    priority = self.dispatch_priority_score(defer_risk, ins_regret, best_cost)
                    options.append(
                        (
                            priority,
                            defer_risk,
                            ins_regret,
                            -best_cost,
                            -(best.selected_window_end if best.selected_window_end is not None else DAY_END_MIN + 1),
                            customer,
                            best,
                        )
                    )
                if not options:
                    break
                chosen = max(options, key=lambda item: item[:6])
                customer = chosen[5]
                insertion = chosen[6]
                sequence = insertion.sequence
                current_route = insertion.route
                inserted_today.append(customer)
                undelivered.remove(customer)

            final_route = current_route
            if self.use_local_search:
                final_route = improve_daily_route(
                    instance,
                    day,
                    final_route,
                    undelivered_today=[customer for customer in undelivered if windows_for(instance, customer, day)],
                    params=LocalSearchParams(
                        max_iterations=self.local_search_max_iterations,
                        time_limit_sec=self.local_search_time_limit_sec,
                        distance_weight=self.distance_weight,
                        waiting_weight=self.waiting_weight,
                        duration_weight=self.duration_weight,
                    ),
                )
                undelivered -= set(final_route.delivered_customer_ids())

            routes[day] = final_route
            day_statuses[day] = {
                "day": day,
                "raw_candidate_count": len(raw_candidates),
                "selected_candidate_count": len(selected),
                "inserted_count": len(final_route.stops),
                "carried_over_count": len(undelivered),
                "use_local_search": self.use_local_search,
                "route_distance": final_route.route_distance_km,
                "route_waiting": final_route.route_waiting_time_min,
                "route_duration": final_route.route_duration_min,
            }

        schedule = WeeklySchedule(routes=routes)
        metrics = evaluate_weekly_schedule(instance, schedule)
        status = {
            "solver": self.name,
            "status": "HEURISTIC_FEASIBLE" if metrics.hard_feasible else "HEURISTIC_INFEASIBLE",
            "gap_percent": "",
            "use_local_search": self.use_local_search,
            "max_candidates_per_day": self.max_candidates_per_day,
            "regret_weight": self.regret_weight,
            "defer_risk_weight": self.defer_risk_weight,
            "insertion_cost_weight": self.insertion_cost_weight,
            "delivered_count": metrics.delivered_count,
            "incomplete_count": metrics.incomplete_count,
            "total_deferral_days": metrics.total_deferral_days,
            "total_distance_km": metrics.total_distance_km,
            "total_waiting_time_min": metrics.total_waiting_time_min,
            "total_route_duration_min": metrics.total_route_duration_min,
            "runtime_sec": time.perf_counter() - start,
            "day_statuses": day_statuses,
            "no_duplicate_delivery": validate_no_duplicates(schedule),
        }
        return WeeklySchedule(routes=routes, solver_status=status)

    def dispatch_priority_score(self, defer_risk: float, insertion_regret: float, best_cost: float) -> float:
        """Return combined dispatch priority."""
        return (
            self.defer_risk_weight * defer_risk
            + self.regret_weight * insertion_regret
            - self.insertion_cost_weight * best_cost
        )

    def _select_candidates(self, instance: Instance, day: int, candidates: list[str]) -> list[str]:
        """Apply daily candidate limit while keeping last-day customers."""
        if self.max_candidates_per_day is None or len(candidates) <= self.max_candidates_per_day:
            return list(candidates)
        mandatory = [customer for customer in candidates if is_last_available_day(instance, customer, day)]
        if len(mandatory) >= self.max_candidates_per_day:
            return sorted(mandatory)
        optional = [customer for customer in candidates if customer not in set(mandatory)]
        optional.sort(key=lambda customer: (-defer_risk_score(instance, day, customer, candidates), customer))
        return sorted(mandatory) + optional[: self.max_candidates_per_day - len(mandatory)]


def defer_risk_score(instance: Instance, day: int, customer_id: str, candidates: list[str] | set[str]) -> float:
    """Return higher score when postponing a customer is risky."""
    del candidates
    remaining_days = remaining_available_days(instance, customer_id, day)
    remaining_count = len(remaining_days)
    width = total_window_width_today(instance, customer_id, day)
    future_loss = window_width_loss_to_future(instance, customer_id, day)
    return (
        1000 * int(is_last_available_day(instance, customer_id, day))
        + 300 / max(1, remaining_count)
        + 200 * future_loss
        + 100 * deadline_pressure(instance, customer_id, day)
        + 500 / max(1, width)
    )


def insertion_regret_score(insertions: list[InsertionOption]) -> float:
    """Return regret score from sorted feasible insertions."""
    if not insertions:
        return -math.inf
    if len(insertions) == 1:
        return 100.0
    return insertions[1].incremental_cost - insertions[0].incremental_cost

