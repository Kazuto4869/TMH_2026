"""Inferior-first feasible insertion heuristic."""

from __future__ import annotations

import time

from vrp_weekly.config import DAY_END_MIN, MONDAY, SUNDAY
from vrp_weekly.core import DailyRoute, Instance, WeeklySchedule
from vrp_weekly.evaluator import evaluate_daily_route, evaluate_weekly_schedule
from vrp_weekly.heuristics.local_search import LocalSearchParams, improve_daily_route
from vrp_weekly.heuristics.route_eval import HeuristicWeights, best_feasible_insertion, validate_no_duplicates, windows_for
from vrp_weekly.heuristics.scoring import (
    deadline_pressure,
    is_last_available_day,
    num_windows_today,
    remaining_available_days,
    spatial_isolation,
    total_window_width_today,
)


class InferiorInsertionSolver:
    """Construct routes by inserting hard-to-serve customers first."""

    name = "inferior_insertion"

    def __init__(
        self,
        use_local_search: bool = False,
        local_search_time_limit_sec: int = 10,
        local_search_max_iterations: int = 100,
        max_candidates_per_day: int | None = None,
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
        self.distance_weight = distance_weight
        self.waiting_weight = waiting_weight
        self.duration_weight = duration_weight
        self.random_seed = random_seed

    def solve(self, instance: Instance) -> WeeklySchedule:
        """Build a weekly schedule with inferior-first insertion."""
        start = time.perf_counter()
        undelivered = set(instance.customer_ids())
        routes: dict[int, DailyRoute] = {}
        day_statuses: dict[int, dict[str, object]] = {}
        weights = HeuristicWeights(self.distance_weight, self.waiting_weight, self.duration_weight)

        for day in range(MONDAY, SUNDAY + 1):
            raw_candidates = sorted(customer for customer in undelivered if windows_for(instance, customer, day))
            selected = self._select_candidates(instance, day, raw_candidates)
            selected = sorted(
                selected,
                key=lambda customer: (-inferiority_score(instance, day, customer, selected), customer),
            )
            sequence: list[str] = []
            current_route = evaluate_daily_route(instance, day, sequence)
            inserted_today: list[str] = []

            while True:
                options = []
                for customer in selected:
                    if customer in inserted_today or customer not in undelivered:
                        continue
                    insertion = best_feasible_insertion(instance, day, sequence, customer, base_route=current_route, weights=weights)
                    if insertion is None:
                        continue
                    score = inferiority_score(instance, day, customer, selected)
                    options.append(
                        (
                            score,
                            -insertion.incremental_cost,
                            -(insertion.selected_window_end if insertion.selected_window_end is not None else DAY_END_MIN + 1),
                            customer,
                            insertion,
                        )
                    )
                if not options:
                    break
                _, _, _, customer, insertion = max(options, key=lambda item: item[:4])
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

    def _select_candidates(self, instance: Instance, day: int, candidates: list[str]) -> list[str]:
        """Apply daily candidate limit while keeping last-day customers."""
        if self.max_candidates_per_day is None or len(candidates) <= self.max_candidates_per_day:
            return list(candidates)
        mandatory = [customer for customer in candidates if is_last_available_day(instance, customer, day)]
        if len(mandatory) >= self.max_candidates_per_day:
            return sorted(mandatory)
        optional = [customer for customer in candidates if customer not in set(mandatory)]
        optional.sort(key=lambda customer: (-inferiority_score(instance, day, customer, candidates), customer))
        return sorted(mandatory) + optional[: self.max_candidates_per_day - len(mandatory)]


def inferiority_score(instance: Instance, day: int, customer_id: str, candidates: list[str] | set[str]) -> float:
    """Return higher score for customers that are harder to serve later."""
    remaining_days = remaining_available_days(instance, customer_id, day)
    remaining_count = len(remaining_days)
    width = total_window_width_today(instance, customer_id, day)
    nwindows = num_windows_today(instance, customer_id, day)
    return (
        1000 * int(is_last_available_day(instance, customer_id, day))
        + 200 / max(1, remaining_count)
        + 500 / max(1, width)
        + 50 / max(1, nwindows)
        + 100 * deadline_pressure(instance, customer_id, day)
        + 20 * spatial_isolation(instance, customer_id, candidates)
    )

