"""Pure rolling-horizon daily CP-SAT routing model."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from vrp_weekly.config import (
    DAY_END_MIN,
    DEFAULT_SERVICE_TIME_MIN,
    DROP_PENALTY_BY_DAY,
    MAX_SPEED_KMPH,
    MINUTES_PER_HOUR,
    MONDAY,
    SUNDAY,
)
from vrp_weekly.core import DailyRoute, Instance, Stop, TimeWindow, WeeklySchedule
from vrp_weekly.distance import euclidean_distance_km

LOGGER = logging.getLogger(__name__)
DISTANCE_SCALE = 1000


@dataclass
class DayModelData:
    """Variables and metadata for one daily CP-SAT model."""

    model: cp_model.CpModel
    x: dict[tuple[str, str], cp_model.IntVar]
    y: dict[str, cp_model.IntVar]
    t: dict[str, cp_model.IntVar]
    g: dict[tuple[str, int], cp_model.IntVar]
    z: cp_model.IntVar
    departure: cp_model.IntVar
    return_time: cp_model.IntVar
    next_travel: dict[str, cp_model.IntVar]
    interval_end: dict[str, cp_model.IntVar]
    depot_first_travel: cp_model.IntVar
    depot_interval_end: cp_model.IntVar
    window_lookup: dict[tuple[str, int], TimeWindow]
    nodes: list[str]
    node_index: dict[str, int]
    arc_stats: dict[str, int]
    tightening_stats: dict[str, int]


def _gap_percent(objective: float, best_bound: float) -> float:
    """Return minimization/maximization optimality gap in percent."""
    return 100.0 * abs(objective - best_bound) / max(1.0, abs(objective))


def _travel_time_minutes(instance: Instance, i: str, j: str) -> int:
    """Return ceil travel time in minutes at the configured max speed."""
    distance_km = euclidean_distance_km(instance.locations[i], instance.locations[j])
    return int(math.ceil(MINUTES_PER_HOUR * distance_km / MAX_SPEED_KMPH))


def _distance_scaled(instance: Instance, i: str, j: str) -> int:
    """Return meter-like scaled distance for tie-breaking/diagnostics only."""
    distance_km = euclidean_distance_km(instance.locations[i], instance.locations[j])
    return int(round(distance_km * DISTANCE_SCALE))


def _distance_objective_cost(instance: Instance, i: str, j: str, distance_weight: int | float) -> int:
    """Return CP objective distance cost in kilometer scale, not meter scale."""
    distance_km = euclidean_distance_km(instance.locations[i], instance.locations[j])
    return int(round(distance_weight * distance_km))


def _get_service_time(instance: Instance, i: str) -> int:
    """Return service time for a customer, with zero service at depots."""
    location = instance.locations[i]
    if location.is_depot or i == instance.depot_id:
        return 0
    return location.service_time if location.service_time > 0 else DEFAULT_SERVICE_TIME_MIN


def _windows_for(instance: Instance, customer_id: str, day: int) -> list[TimeWindow]:
    """Return sorted windows for one customer/day."""
    return instance.windows_for_customer_day(customer_id, day)


def _window_width(window: TimeWindow) -> int:
    """Return a time window width in minutes."""
    return window.end_minute - window.start_minute


def _total_window_width_today(instance: Instance, customer_id: str, day: int) -> int:
    """Return total width of all windows for a customer today."""
    return sum(_window_width(window) for window in _windows_for(instance, customer_id, day))


def _remaining_available_days(instance: Instance, customer_id: str, day: int) -> list[int]:
    """Return sorted available days from today onward."""
    return sorted(available_day for available_day in instance.available_days(customer_id) if available_day >= day)


def _is_last_available_day(instance: Instance, customer_id: str, day: int) -> bool:
    """Return true when today is the customer's last remaining available day."""
    return _remaining_available_days(instance, customer_id, day) == [day]


def _earliest_window_end_today(instance: Instance, customer_id: str, day: int) -> int:
    """Return earliest window end today, or a large value when none exists."""
    windows = _windows_for(instance, customer_id, day)
    return min((window.end_minute for window in windows), default=DAY_END_MIN + 1)


def _num_windows_today(instance: Instance, customer_id: str, day: int) -> int:
    """Return number of time windows for a customer today."""
    return len(_windows_for(instance, customer_id, day))


def _service_fits_window(service_time: int, window: TimeWindow) -> bool:
    """Return true if service can fit inside a window at its opening time."""
    return window.start_minute + service_time <= window.end_minute


def _can_start_from_depot(instance: Instance, day: int, customer_id: str) -> bool:
    """Return true if a flexible depot departure can reach a customer window."""
    travel = _travel_time_minutes(instance, instance.depot_id, customer_id)
    service = _get_service_time(instance, customer_id)
    return any(
        _service_fits_window(service, window) and travel <= window.end_minute - service
        for window in _windows_for(instance, customer_id, day)
    )


def _can_return_to_depot(instance: Instance, day: int, customer_id: str) -> bool:
    """Return true if a customer can be served and return to depot by day end."""
    travel_back = _travel_time_minutes(instance, customer_id, instance.depot_id)
    service = _get_service_time(instance, customer_id)
    return any(
        _service_fits_window(service, window) and window.start_minute + service + travel_back <= DAY_END_MIN
        for window in _windows_for(instance, customer_id, day)
    )


def _can_follow(instance: Instance, day: int, i: str, j: str) -> bool:
    """Return a conservative arc feasibility check between two route nodes."""
    if i == j:
        return True
    depot_id = instance.depot_id
    if i == depot_id and j != depot_id:
        return _can_start_from_depot(instance, day, j)
    if i != depot_id and j == depot_id:
        return _can_return_to_depot(instance, day, i)
    if i != depot_id and j != depot_id:
        service_i = _get_service_time(instance, i)
        service_j = _get_service_time(instance, j)
        travel = _travel_time_minutes(instance, i, j)
        for window_i in _windows_for(instance, i, day):
            if not _service_fits_window(service_i, window_i):
                continue
            for window_j in _windows_for(instance, j, day):
                if not _service_fits_window(service_j, window_j):
                    continue
                if window_i.start_minute + service_i + travel <= window_j.end_minute - service_j:
                    return True
        return False
    return True


def _window_pair_can_follow(
    instance: Instance,
    i: str,
    j: str,
    window_i: TimeWindow,
    window_j: TimeWindow,
) -> bool:
    """Return true if selected windows can support customer i before j."""
    service_i = _get_service_time(instance, i)
    service_j = _get_service_time(instance, j)
    travel = _travel_time_minutes(instance, i, j)
    return (
        _service_fits_window(service_i, window_i)
        and _service_fits_window(service_j, window_j)
        and window_i.start_minute + service_i + travel <= window_j.end_minute - service_j
    )


def _depot_can_reach_window(instance: Instance, customer_id: str, window: TimeWindow) -> bool:
    """Return true if depot can reach a specific selected window."""
    service = _get_service_time(instance, customer_id)
    travel = _travel_time_minutes(instance, instance.depot_id, customer_id)
    return _service_fits_window(service, window) and travel <= window.end_minute - service


def _window_can_return_to_depot(instance: Instance, customer_id: str, window: TimeWindow) -> bool:
    """Return true if a specific selected window can return to depot by day end."""
    service = _get_service_time(instance, customer_id)
    travel_back = _travel_time_minutes(instance, customer_id, instance.depot_id)
    return _service_fits_window(service, window) and window.start_minute + service + travel_back <= DAY_END_MIN


def _is_window_dominated(candidate: TimeWindow, other: TimeWindow, candidate_index: int, other_index: int) -> bool:
    """Return true if candidate is contained in another same-day window."""
    contains = other.start_minute <= candidate.start_minute and candidate.end_minute <= other.end_minute
    strictly_wider = other.start_minute < candidate.start_minute or candidate.end_minute < other.end_minute
    duplicate_tiebreak = (
        other.start_minute == candidate.start_minute
        and other.end_minute == candidate.end_minute
        and other_index < candidate_index
    )
    return contains and (strictly_wider or duplicate_tiebreak)


def _fix_impossible_arcs(
    model: cp_model.CpModel,
    x: dict[tuple[str, str], cp_model.IntVar],
    instance: Instance,
    day: int,
    nodes: list[str],
    depot_id: str,
) -> dict[str, int]:
    """Fix provably impossible non-self arcs to zero while keeping AddCircuit vars."""
    del depot_id
    fixed_count = 0
    total_nonself_arcs = 0
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            total_nonself_arcs += 1
            if not _can_follow(instance, day, i, j):
                model.Add(x[i, j] == 0)
                fixed_count += 1
    return {"fixed_impossible_arcs": fixed_count, "total_nonself_arcs": total_nonself_arcs}


def _candidate_priority(instance: Instance, day: int, customer_id: str) -> tuple[int, int, int, int, int, int, str]:
    """Return deterministic inferior-first candidate priority."""
    return (
        0 if _is_last_available_day(instance, customer_id, day) else 1,
        len(_remaining_available_days(instance, customer_id, day)),
        _total_window_width_today(instance, customer_id, day),
        _num_windows_today(instance, customer_id, day),
        _earliest_window_end_today(instance, customer_id, day),
        -_distance_scaled(instance, instance.depot_id, customer_id),
        customer_id,
    )


def _easy_candidate_priority(instance: Instance, day: int, customer_id: str) -> tuple[int, int, int, int, str]:
    """Return deterministic priority for easier, central customers."""
    return (
        -_total_window_width_today(instance, customer_id, day),
        -len(_remaining_available_days(instance, customer_id, day)),
        _distance_scaled(instance, instance.depot_id, customer_id),
        _earliest_window_end_today(instance, customer_id, day),
        customer_id,
    )


def _deadline_candidate_priority(instance: Instance, day: int, customer_id: str) -> tuple[int, int, str]:
    """Return deterministic earliest-deadline candidate priority."""
    return (
        _earliest_window_end_today(instance, customer_id, day),
        _total_window_width_today(instance, customer_id, day),
        customer_id,
    )


def _isolated_candidate_priority(instance: Instance, day: int, customer_id: str) -> tuple[int, int, str]:
    """Return deterministic far-from-depot candidate priority."""
    return (
        -_distance_scaled(instance, instance.depot_id, customer_id),
        _earliest_window_end_today(instance, customer_id, day),
        customer_id,
    )


def _extract_route_from_arcs(
    selected_arcs: set[tuple[str, str]],
    depot_id: str,
    customers: list[str],
) -> list[str]:
    """Extract the route after the depot from selected directed arcs."""
    next_by_node = {i: j for i, j in selected_arcs if i != j}
    route: list[str] = []
    seen: set[str] = set()
    current = depot_id
    customer_set = set(customers)

    for _ in range(len(customers) + 1):
        next_node = next_by_node.get(current)
        if next_node is None or next_node == depot_id:
            break
        if next_node in seen:
            break
        seen.add(next_node)
        if next_node in customer_set:
            route.append(next_node)
        current = next_node

    return route


def _build_daily_schedule_from_solution(
    instance: Instance,
    day: int,
    route_sequence: list[str],
    service_start_times: dict[str, int],
    selected_windows: dict[str, TimeWindow],
    depot_departure_time: int = 0,
) -> DailyRoute:
    """Build a project-compatible daily route from CP solution values."""
    previous_id = instance.depot_id
    previous_departure = depot_departure_time
    stops: list[Stop] = []
    route_distance_km = 0.0
    route_travel_time_min = 0
    route_waiting_time_min = 0
    route_service_time_min = 0
    violations: list[str] = []

    for customer_id in route_sequence:
        travel_time = _travel_time_minutes(instance, previous_id, customer_id)
        distance_km = euclidean_distance_km(instance.locations[previous_id], instance.locations[customer_id])
        arrival_time = previous_departure + travel_time
        service_start = service_start_times[customer_id]
        service_time = _get_service_time(instance, customer_id)
        service_end = service_start + service_time
        selected_window = selected_windows.get(customer_id)
        waiting_min = max(0, service_start - arrival_time)
        hard_feasible = True
        violation = None

        if selected_window is None:
            hard_feasible = False
            violation = f"missing selected time window for {customer_id}"
        elif service_start < selected_window.start_minute or service_end > selected_window.end_minute:
            hard_feasible = False
            violation = f"service outside selected time window for {customer_id}"
        elif service_start < arrival_time:
            hard_feasible = False
            violation = f"service starts before arrival for {customer_id}"

        if violation:
            violations.append(violation)

        stops.append(
            Stop(
                customer_id=customer_id,
                arrival_time=arrival_time,
                service_start_time=service_start,
                service_end_time=service_end,
                selected_time_window=selected_window,
                travel_from_previous_min=travel_time,
                waiting_min=waiting_min,
                distance_from_previous_km=distance_km,
                hard_feasible=hard_feasible,
                violation=violation,
            )
        )
        route_distance_km += distance_km
        route_travel_time_min += travel_time
        route_waiting_time_min += waiting_min
        route_service_time_min += service_time
        previous_id = customer_id
        previous_departure = service_end

    return_travel_time = _travel_time_minutes(instance, previous_id, instance.depot_id)
    return_distance_km = euclidean_distance_km(instance.locations[previous_id], instance.depot())
    route_distance_km += return_distance_km
    route_travel_time_min += return_travel_time
    actual_return = previous_departure + return_travel_time if route_sequence else 0
    if actual_return > DAY_END_MIN:
        violations.append(f"Route on day {day} returns after 24:00")

    route_duration = actual_return - depot_departure_time if route_sequence else 0
    return DailyRoute(
        day=day,
        stops=stops,
        depot_departure_time=depot_departure_time if route_sequence else 0,
        return_to_depot_time=actual_return if route_sequence else 0,
        route_distance_km=route_distance_km if route_sequence else 0.0,
        route_travel_time_min=route_travel_time_min if route_sequence else 0,
        route_waiting_time_min=route_waiting_time_min if route_sequence else 0,
        route_service_time_min=route_service_time_min if route_sequence else 0,
        route_duration_min=route_duration,
        hard_feasible=not violations and all(stop.hard_feasible for stop in stops),
        violations=violations,
    )


class RollingHorizonCPSATSolver:
    """Pure CP-SAT rolling horizon, one independent daily model at a time."""

    name = "cp_rolling"

    def __init__(
        self,
        time_limit_per_day_sec: int = 10,
        max_candidates_per_day: int | None = None,
        drop_penalty_by_day: dict[int, int] | None = None,
        distance_weight: int = 10,
        route_duration_weight: int = 1,
        urgency_weight: int = 100,
        num_workers: int = 8,
        log_search_progress: bool = False,
        use_two_phase_objective: bool = True,
        phase1_time_limit_sec: int | None = None,
        phase2_time_limit_sec: int | None = None,
        random_seed: int = 1,
        use_decision_strategy: bool = True,
        use_service_no_overlap: bool = True,
        candidate_strategy: str = "hybrid",
        solve_phase2: bool = True,
    ) -> None:
        if candidate_strategy not in {"urgent", "hybrid"}:
            raise ValueError(f"candidate_strategy must be 'urgent' or 'hybrid', got {candidate_strategy!r}")
        self.time_limit_per_day_sec = time_limit_per_day_sec
        self.max_candidates_per_day = max_candidates_per_day
        self.drop_penalty_by_day = dict(DROP_PENALTY_BY_DAY if drop_penalty_by_day is None else drop_penalty_by_day)
        self.distance_weight = distance_weight
        self.route_duration_weight = route_duration_weight
        self.urgency_weight = urgency_weight
        self.num_workers = num_workers
        self.log_search_progress = log_search_progress
        self.use_two_phase_objective = use_two_phase_objective
        self.phase1_time_limit_sec = phase1_time_limit_sec
        self.phase2_time_limit_sec = phase2_time_limit_sec
        self.random_seed = random_seed
        self.use_decision_strategy = use_decision_strategy
        self.use_service_no_overlap = use_service_no_overlap
        self.candidate_strategy = candidate_strategy
        self.solve_phase2 = solve_phase2
        self._last_candidate_selection_stats: dict[str, object] = {}

    def solve(self, instance: Instance) -> WeeklySchedule:
        """Build a weekly schedule by solving pure daily CP-SAT models."""
        undelivered = set(instance.customer_ids())
        routes: dict[int, DailyRoute] = {}
        day_statuses: dict[int, dict[str, object]] = {}

        for day in range(MONDAY, SUNDAY + 1):
            LOGGER.info("cp_rolling day=%s start undelivered=%s", day, len(undelivered))
            day_start = time.perf_counter()
            raw_candidates = sorted(customer for customer in undelivered if _windows_for(instance, customer, day))
            raw_candidate_count = len(raw_candidates)
            mandatory_last_day_count = sum(1 for customer in raw_candidates if _is_last_available_day(instance, customer, day))
            candidates = self._limit_candidates(instance, day, raw_candidates)
            candidate_selection_stats = dict(self._last_candidate_selection_stats)
            selected_candidate_count = len(candidates)
            if not candidates:
                routes[day] = DailyRoute(day=day)
                day_statuses[day] = self._empty_day_status(
                    day_start=day_start,
                    raw_candidate_count=raw_candidate_count,
                    selected_candidate_count=selected_candidate_count,
                    mandatory_last_day_count=mandatory_last_day_count,
                    remaining_after_day=len(undelivered),
                    candidate_selection_stats=candidate_selection_stats,
                )
                LOGGER.info("cp_rolling day=%s no candidates", day)
                continue

            LOGGER.info("cp_rolling day=%s solve candidates=%s", day, len(candidates))
            route, day_status = self._solve_day(instance, day, candidates)
            delivered_ids = route.delivered_customer_ids()
            undelivered -= delivered_ids
            delivered_today = len(delivered_ids)
            day_status.update(
                {
                    "runtime_sec": time.perf_counter() - day_start,
                    "raw_candidate_count": raw_candidate_count,
                    "selected_candidate_count": selected_candidate_count,
                    "mandatory_last_day_count": mandatory_last_day_count,
                    "complete_count": delivered_today,
                    "delivered_today": delivered_today,
                    "selected_carried_over_count": max(0, selected_candidate_count - delivered_today),
                    "raw_carried_over_count": max(0, raw_candidate_count - delivered_today),
                    "carried_over_count": max(0, selected_candidate_count - delivered_today),
                    "remaining_after_day": len(undelivered),
                    **candidate_selection_stats,
                }
            )
            routes[day] = route
            day_statuses[day] = day_status
            LOGGER.info(
                "cp_rolling day=%s done status=%s stops=%s remaining=%s runtime=%.3fs",
                day,
                day_status.get("status", ""),
                len(route.stops),
                len(undelivered),
                float(day_status["runtime_sec"]),
            )

        return WeeklySchedule(routes=routes, solver_status=self._weekly_status(day_statuses, len(undelivered)))

    def _empty_day_status(
        self,
        day_start: float,
        raw_candidate_count: int,
        selected_candidate_count: int,
        mandatory_last_day_count: int,
        remaining_after_day: int,
        candidate_selection_stats: dict[str, object],
    ) -> dict[str, object]:
        return {
            "objective_mode": "two_phase" if self.use_two_phase_objective else "single_phase",
            "status": "NO_CANDIDATES",
            "objective": "",
            "best_bound": "",
            "runtime_sec": time.perf_counter() - day_start,
            "gap_percent": "",
            "raw_candidate_count": raw_candidate_count,
            "selected_candidate_count": selected_candidate_count,
            "mandatory_last_day_count": mandatory_last_day_count,
            "fixed_impossible_arcs": 0,
            "total_nonself_arcs": 0,
            "fixed_arc_ratio": 0.0,
            "degree_linking_constraints_count": 0,
            "arc_linking_constraints_count": 0,
            "window_pair_cuts_count": 0,
            "pair_conflict_cuts_count": 0,
            "depot_window_cuts_count": 0,
            "dominated_window_cuts_count": 0,
            "precedence_cuts_count": 0,
            "drop_penalty": self.drop_penalty_by_day.get(MONDAY, DROP_PENALTY_BY_DAY[SUNDAY]),
            "distance_weight": self.distance_weight,
            "route_duration_weight": self.route_duration_weight,
            "urgency_weight": self.urgency_weight,
            "distance_objective_scale": "km",
            "complete_count": 0,
            "delivered_today": 0,
            "selected_carried_over_count": 0,
            "raw_carried_over_count": raw_candidate_count,
            "carried_over_count": 0,
            "remaining_after_day": remaining_after_day,
            "solver_return_time": "",
            "daily_optimal_for": "no_candidates",
            "service_no_overlap_enabled": self.use_service_no_overlap,
            "service_interval_count": 0,
            "route_interval_count": 0,
            "depot_interval_enabled": False,
            "no_overlap_route_intervals_enabled": False,
            "roundtrip_duration_lb_count": 0,
            "fixed_impossible_customers_count": 0,
            "phase1_only": not self.solve_phase2,
            "solve_phase2": self.solve_phase2,
            "decision_strategy_enabled": self.use_decision_strategy,
            **candidate_selection_stats,
        }

    def _weekly_status(self, day_statuses: dict[int, dict[str, object]], remaining_after_week: int) -> dict[str, object]:
        solved_statuses = [status for status in day_statuses.values() if status.get("status") != "NO_CANDIDATES"]
        if not solved_statuses:
            overall_status = "NO_CANDIDATES"
        elif any(status.get("status") not in {"OPTIMAL", "FEASIBLE"} for status in solved_statuses):
            overall_status = "PARTIAL_OR_FAILED"
        elif all(status.get("status") == "OPTIMAL" for status in solved_statuses):
            overall_status = "ALL_DAYS_OPTIMAL"
        else:
            overall_status = "FEASIBLE"

        gaps = [float(status["gap_percent"]) for status in solved_statuses if status.get("gap_percent") not in ("", None)]
        fixed_counts = [int(status.get("fixed_impossible_arcs", 0)) for status in day_statuses.values()]
        fixed_ratios = [float(status.get("fixed_arc_ratio", 0.0)) for status in day_statuses.values()]
        delivered_count_certified_days = sum(1 for status in solved_statuses if status.get("phase1_status") == "OPTIMAL")
        route_cost_certified_days = sum(
            1
            for status in solved_statuses
            if status.get("phase2_status") == "OPTIMAL"
            or (status.get("objective_mode") == "single_phase" and status.get("status") == "OPTIMAL")
        )
        all_days_delivered_count_optimal = bool(solved_statuses) and all(
            status.get("phase1_status") == "OPTIMAL"
            or (status.get("objective_mode") == "single_phase" and status.get("status") == "OPTIMAL")
            for status in solved_statuses
        )
        all_days_route_cost_optimal = bool(solved_statuses) and all(
            status.get("phase2_status") == "OPTIMAL"
            or (status.get("objective_mode") == "single_phase" and status.get("status") == "OPTIMAL")
            for status in solved_statuses
        )
        daily_optimality_claim = bool(solved_statuses) and all(status.get("status") == "OPTIMAL" for status in solved_statuses)

        return {
            "status": overall_status,
            "gap_percent": max(gaps) if gaps else "",
            "max_day_gap_percent": max(gaps) if gaps else "",
            "global_optimality_claim": False,
            "daily_optimality_claim": daily_optimality_claim,
            "daily_optimality_scope": "selected_candidates"
            if self.max_candidates_per_day is not None
            else "all_available_daily_candidates",
            "total_runtime_sec": sum(float(status.get("runtime_sec", 0.0)) for status in day_statuses.values()),
            "total_fixed_impossible_arcs": sum(fixed_counts),
            "average_fixed_arc_ratio": sum(fixed_ratios) / len(fixed_ratios) if fixed_ratios else 0.0,
            "total_remaining_after_week": remaining_after_week,
            "total_route_interval_count": sum(int(status.get("route_interval_count", 0)) for status in day_statuses.values()),
            "route_no_overlap_days": sum(
                1 for status in day_statuses.values() if status.get("no_overlap_route_intervals_enabled") is True
            ),
            "day_statuses": day_statuses,
            "delivered_count_certified_days": delivered_count_certified_days,
            "route_cost_certified_days": route_cost_certified_days,
            "all_days_delivered_count_optimal": all_days_delivered_count_optimal,
            "all_days_route_cost_optimal": all_days_route_cost_optimal,
        }

    def _limit_candidates(self, instance: Instance, day: int, candidates: list[str]) -> list[str]:
        """Apply deterministic candidate screening for large real-data days."""
        ordered = sorted(candidates, key=lambda customer_id: _candidate_priority(instance, day, customer_id))
        self._last_candidate_selection_stats = {
            "candidate_strategy": self.candidate_strategy,
            "urgent_bucket_count": 0,
            "easy_bucket_count": 0,
            "deadline_bucket_count": 0,
            "isolated_bucket_count": 0,
        }
        if self.max_candidates_per_day is None or len(ordered) <= self.max_candidates_per_day:
            return ordered

        if self.candidate_strategy == "urgent":
            selected = self._limit_candidates_urgent(instance, day, ordered)
            self._last_candidate_selection_stats["urgent_bucket_count"] = len(selected)
            return selected

        return self._limit_candidates_hybrid(instance, day, ordered)

    def _limit_candidates_urgent(self, instance: Instance, day: int, ordered: list[str]) -> list[str]:
        """Apply the original urgent-first candidate filtering."""
        mandatory = [customer_id for customer_id in ordered if _is_last_available_day(instance, customer_id, day)]
        if len(mandatory) > self.max_candidates_per_day:
            LOGGER.warning("candidate limit exceeded by mandatory last-available-day customers")
            return mandatory

        selected = list(mandatory)
        selected_set = set(selected)
        for customer_id in ordered:
            if len(selected) >= self.max_candidates_per_day:
                break
            if customer_id not in selected_set:
                selected.append(customer_id)
                selected_set.add(customer_id)
        return selected

    def _limit_candidates_hybrid(self, instance: Instance, day: int, ordered: list[str]) -> list[str]:
        """Select candidates from urgent, easy, deadline, and isolated buckets."""
        assert self.max_candidates_per_day is not None
        mandatory = [customer_id for customer_id in ordered if _is_last_available_day(instance, customer_id, day)]
        if len(mandatory) > self.max_candidates_per_day:
            LOGGER.warning("candidate limit exceeded by mandatory last-available-day customers")
            self._last_candidate_selection_stats.update(
                {
                    "urgent_bucket_count": len(mandatory),
                    "easy_bucket_count": 0,
                    "deadline_bucket_count": 0,
                    "isolated_bucket_count": 0,
                }
            )
            return mandatory

        selected = list(mandatory)
        selected_set = set(selected)
        remaining_slots = self.max_candidates_per_day - len(selected)
        if remaining_slots <= 0:
            return selected

        nonmandatory = [customer_id for customer_id in ordered if customer_id not in selected_set]
        buckets = {
            "urgent_bucket_count": list(nonmandatory),
            "easy_bucket_count": sorted(nonmandatory, key=lambda customer_id: _easy_candidate_priority(instance, day, customer_id)),
            "deadline_bucket_count": sorted(nonmandatory, key=lambda customer_id: _deadline_candidate_priority(instance, day, customer_id)),
            "isolated_bucket_count": sorted(nonmandatory, key=lambda customer_id: _isolated_candidate_priority(instance, day, customer_id)),
        }
        quotas = self._bucket_quotas(remaining_slots)
        positions = {name: 0 for name in buckets}
        contributed = {name: 0 for name in buckets}

        for name in ("urgent_bucket_count", "easy_bucket_count", "deadline_bucket_count", "isolated_bucket_count"):
            while contributed[name] < quotas[name] and len(selected) < self.max_candidates_per_day:
                customer_id, positions[name] = self._next_bucket_customer(buckets[name], positions[name], selected_set)
                if customer_id is None:
                    break
                selected.append(customer_id)
                selected_set.add(customer_id)
                contributed[name] += 1

        while len(selected) < self.max_candidates_per_day:
            added = False
            for name in ("urgent_bucket_count", "easy_bucket_count", "deadline_bucket_count", "isolated_bucket_count"):
                customer_id, positions[name] = self._next_bucket_customer(buckets[name], positions[name], selected_set)
                if customer_id is None:
                    continue
                selected.append(customer_id)
                selected_set.add(customer_id)
                contributed[name] += 1
                added = True
                if len(selected) >= self.max_candidates_per_day:
                    break
            if not added:
                break

        self._last_candidate_selection_stats.update(contributed)
        return selected

    @staticmethod
    def _bucket_quotas(remaining_slots: int) -> dict[str, int]:
        """Return deterministic bucket quotas for hybrid candidate selection."""
        weights = [
            ("urgent_bucket_count", 0.4),
            ("easy_bucket_count", 0.3),
            ("deadline_bucket_count", 0.2),
            ("isolated_bucket_count", 0.1),
        ]
        raw = [(name, remaining_slots * weight) for name, weight in weights]
        quotas = {name: int(value) for name, value in raw}
        remainder = remaining_slots - sum(quotas.values())
        for name, _value in sorted(raw, key=lambda item: (-(item[1] - int(item[1])), item[0])):
            if remainder <= 0:
                break
            quotas[name] += 1
            remainder -= 1
        return quotas

    @staticmethod
    def _next_bucket_customer(bucket: list[str], start_index: int, selected_set: set[str]) -> tuple[str | None, int]:
        """Return the next not-yet-selected customer from a bucket."""
        index = start_index
        while index < len(bucket):
            customer_id = bucket[index]
            index += 1
            if customer_id not in selected_set:
                return customer_id, index
        return None, index

    def _build_day_model(
        self,
        instance: Instance,
        day: int,
        candidates: list[str],
        use_decision_strategy: bool | None = None,
    ) -> DayModelData:
        """Build one pure daily CP-SAT model without an objective."""
        if use_decision_strategy is None:
            use_decision_strategy = self.use_decision_strategy
        model = cp_model.CpModel()
        depot_id = instance.depot_id
        nodes = [depot_id] + candidates
        node_index = {node_id: index for index, node_id in enumerate(nodes)}

        y = {customer_id: model.NewBoolVar(f"y[{customer_id}]") for customer_id in candidates}
        t = {customer_id: model.NewIntVar(0, DAY_END_MIN, f"T[{customer_id}]") for customer_id in candidates}
        g: dict[tuple[str, int], cp_model.IntVar] = {}
        x = {(i, j): model.NewBoolVar(f"x[{i},{j}]") for i in nodes for j in nodes}
        z = model.NewBoolVar("z")
        departure = model.NewIntVar(0, DAY_END_MIN, "L")
        return_time = model.NewIntVar(0, DAY_END_MIN, "R")
        max_travel = max(
            (_travel_time_minutes(instance, i, j) for i in nodes for j in nodes if i != j),
            default=0,
        )
        next_travel = {
            customer_id: model.NewIntVar(0, max_travel, f"next_travel[{customer_id}]")
            for customer_id in candidates
        }
        interval_end = {
            customer_id: model.NewIntVar(0, DAY_END_MIN, f"interval_end[{customer_id}]")
            for customer_id in candidates
        }
        depot_first_travel = model.NewIntVar(0, max_travel, "first_travel")
        depot_interval_end = model.NewIntVar(0, DAY_END_MIN, "depot_interval_end")
        window_lookup: dict[tuple[str, int], TimeWindow] = {}
        tightening_stats = {
            "degree_linking_constraints_count": 0,
            "arc_linking_constraints_count": 0,
            "window_pair_cuts_count": 0,
            "pair_conflict_cuts_count": 0,
            "depot_window_cuts_count": 0,
            "dominated_window_cuts_count": 0,
            "precedence_cuts_count": 0,
            "service_interval_count": 0,
            "route_interval_count": 0,
            "depot_interval_enabled": True,
            "no_overlap_route_intervals_enabled": True,
            "roundtrip_duration_lb_count": 0,
            "fixed_impossible_customers_count": 0,
        }
        service_intervals: list[cp_model.IntervalVar] = []
        route_intervals: list[cp_model.IntervalVar] = []

        for customer_id in candidates:
            window_vars: list[cp_model.IntVar] = []
            service_time = _get_service_time(instance, customer_id)
            windows_today = _windows_for(instance, customer_id, day)
            for window_idx, window in enumerate(_windows_for(instance, customer_id, day)):
                var = model.NewBoolVar(f"g[{customer_id},{window_idx}]")
                g[customer_id, window_idx] = var
                window_lookup[customer_id, window_idx] = window
                window_vars.append(var)
                model.Add(t[customer_id] >= window.start_minute).OnlyEnforceIf(var)
                model.Add(t[customer_id] + service_time <= window.end_minute).OnlyEnforceIf(var)
            for window_idx, window in enumerate(windows_today):
                if any(
                    other_idx != window_idx and _is_window_dominated(window, other, window_idx, other_idx)
                    for other_idx, other in enumerate(windows_today)
                ):
                    model.Add(g[customer_id, window_idx] == 0)
                    tightening_stats["dominated_window_cuts_count"] += 1
            model.Add(sum(window_vars) == y[customer_id])
            model.Add(x[customer_id, customer_id] + y[customer_id] == 1)
            model.Add(y[customer_id] <= z)
            model.Add(t[customer_id] == 0).OnlyEnforceIf(y[customer_id].Not())
            model.Add(
                next_travel[customer_id]
                == sum(
                    _travel_time_minutes(instance, customer_id, j) * x[customer_id, j]
                    for j in nodes
                    if j != customer_id
                )
            )
            model.Add(interval_end[customer_id] == t[customer_id] + service_time + next_travel[customer_id]).OnlyEnforceIf(
                y[customer_id]
            )
            model.Add(interval_end[customer_id] == 0).OnlyEnforceIf(y[customer_id].Not())
            model.Add(next_travel[customer_id] == 0).OnlyEnforceIf(y[customer_id].Not())
            route_intervals.append(
                model.NewOptionalIntervalVar(
                    t[customer_id],
                    service_time + next_travel[customer_id],
                    interval_end[customer_id],
                    y[customer_id],
                    f"route_interval[{customer_id}]",
                )
            )
            tightening_stats["route_interval_count"] += 1
            feasible_latest_starts = [
                window.end_minute - service_time for window in windows_today if _service_fits_window(service_time, window)
            ]
            if feasible_latest_starts:
                model.Add(t[customer_id] >= min(window.start_minute for window in windows_today)).OnlyEnforceIf(y[customer_id])
                model.Add(t[customer_id] <= max(feasible_latest_starts)).OnlyEnforceIf(y[customer_id])
            else:
                model.Add(y[customer_id] == 0)
                tightening_stats["fixed_impossible_customers_count"] += 1
            if feasible_latest_starts and (
                not _can_start_from_depot(instance, day, customer_id)
                or not _can_return_to_depot(instance, day, customer_id)
            ):
                model.Add(y[customer_id] == 0)
                tightening_stats["fixed_impossible_customers_count"] += 1
            if self.use_service_no_overlap and service_time > 0:
                service_end = model.NewIntVar(0, DAY_END_MIN + service_time, f"S_end[{customer_id}]")
                model.Add(service_end == t[customer_id] + service_time)
                service_intervals.append(
                    model.NewOptionalIntervalVar(
                        t[customer_id],
                        service_time,
                        service_end,
                        y[customer_id],
                        f"S_interval[{customer_id}]",
                    )
                )
                tightening_stats["service_interval_count"] += 1

        if self.use_service_no_overlap and service_intervals:
            model.AddNoOverlap(service_intervals)

        model.Add(
            depot_first_travel
            == sum(_travel_time_minutes(instance, depot_id, j) * x[depot_id, j] for j in candidates)
        )
        model.Add(depot_interval_end == departure + depot_first_travel).OnlyEnforceIf(z)
        model.Add(depot_first_travel == 0).OnlyEnforceIf(z.Not())
        model.Add(depot_interval_end == 0).OnlyEnforceIf(z.Not())
        depot_interval = model.NewOptionalIntervalVar(
            departure,
            depot_first_travel,
            depot_interval_end,
            z,
            "route_interval[depot]",
        )
        model.AddNoOverlap([depot_interval] + route_intervals)

        model.Add(z <= sum(y.values()))
        model.Add(x[depot_id, depot_id] + z == 1)
        for customer_id in candidates:
            model.Add(sum(x[customer_id, j] for j in nodes if j != customer_id) == y[customer_id])
            model.Add(sum(x[j, customer_id] for j in nodes if j != customer_id) == y[customer_id])
            tightening_stats["degree_linking_constraints_count"] += 2
        model.Add(sum(x[depot_id, j] for j in candidates) == z)
        model.Add(sum(x[i, depot_id] for i in candidates) == z)
        tightening_stats["degree_linking_constraints_count"] += 2

        for i in candidates:
            for j in candidates:
                if i == j:
                    continue
                model.Add(x[i, j] <= y[i])
                model.Add(x[i, j] <= y[j])
                tightening_stats["arc_linking_constraints_count"] += 2
        for customer_id in candidates:
            model.Add(x[depot_id, customer_id] <= y[customer_id])
            model.Add(x[customer_id, depot_id] <= y[customer_id])
            tightening_stats["arc_linking_constraints_count"] += 2

        model.AddCircuit([(node_index[i], node_index[j], x[i, j]) for i in nodes for j in nodes])
        arc_stats = _fix_impossible_arcs(
            model=model,
            x=x,
            instance=instance,
            day=day,
            nodes=nodes,
            depot_id=depot_id,
        )
        LOGGER.info(
            "cp_rolling day=%s fixed impossible arcs %s/%s",
            day,
            arc_stats["fixed_impossible_arcs"],
            arc_stats["total_nonself_arcs"],
        )

        model.Add(departure <= DAY_END_MIN * z)
        model.Add(return_time <= DAY_END_MIN * z)
        model.Add(return_time >= departure).OnlyEnforceIf(z)
        model.Add(return_time <= DAY_END_MIN)
        model.Add(return_time - departure >= sum(_get_service_time(instance, customer_id) * y[customer_id] for customer_id in candidates))
        for customer_id in candidates:
            roundtrip_min = (
                _travel_time_minutes(instance, depot_id, customer_id)
                + _get_service_time(instance, customer_id)
                + _travel_time_minutes(instance, customer_id, depot_id)
            )
            model.Add(return_time - departure >= roundtrip_min * y[customer_id])
            tightening_stats["roundtrip_duration_lb_count"] += 1

        for i in candidates:
            for j in candidates:
                if i == j:
                    continue
                model.Add(interval_end[i] <= t[j]).OnlyEnforceIf(x[i, j])
                windows_i = _windows_for(instance, i, day)
                windows_j = _windows_for(instance, j, day)
                for window_i_idx, window_i in enumerate(windows_i):
                    for window_j_idx, window_j in enumerate(windows_j):
                        if not _window_pair_can_follow(instance, i, j, window_i, window_j):
                            model.Add(x[i, j] + g[i, window_i_idx] + g[j, window_j_idx] <= 2)
                            tightening_stats["window_pair_cuts_count"] += 1
                if _can_follow(instance, day, i, j) and not _can_follow(instance, day, j, i):
                    model.Add(interval_end[i] <= t[j]).OnlyEnforceIf([y[i], y[j]])
                    tightening_stats["precedence_cuts_count"] += 1

        for i_idx, i in enumerate(candidates):
            for j in candidates[i_idx + 1 :]:
                if not _can_follow(instance, day, i, j) and not _can_follow(instance, day, j, i):
                    model.Add(y[i] + y[j] <= 1)
                    tightening_stats["pair_conflict_cuts_count"] += 1

        for j in candidates:
            model.Add(depot_interval_end == t[j]).OnlyEnforceIf(x[depot_id, j])
            for window_idx, window in enumerate(_windows_for(instance, j, day)):
                if not _depot_can_reach_window(instance, j, window):
                    model.Add(x[depot_id, j] + g[j, window_idx] <= 1)
                    tightening_stats["depot_window_cuts_count"] += 1

        for i in candidates:
            model.Add(return_time == interval_end[i]).OnlyEnforceIf(x[i, depot_id])
            for window_idx, window in enumerate(_windows_for(instance, i, day)):
                if not _window_can_return_to_depot(instance, i, window):
                    model.Add(x[i, depot_id] + g[i, window_idx] <= 1)
                    tightening_stats["depot_window_cuts_count"] += 1

        if use_decision_strategy:
            model.AddDecisionStrategy(list(y.values()), cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE)

        return DayModelData(
            model=model,
            x=x,
            y=y,
            t=t,
            g=g,
            z=z,
            departure=departure,
            return_time=return_time,
            next_travel=next_travel,
            interval_end=interval_end,
            depot_first_travel=depot_first_travel,
            depot_interval_end=depot_interval_end,
            window_lookup=window_lookup,
            nodes=nodes,
            node_index=node_index,
            arc_stats=arc_stats,
            tightening_stats=tightening_stats,
        )

    def _solve_day(self, instance: Instance, day: int, candidates: list[str]) -> tuple[DailyRoute, dict[str, object]]:
        """Solve one daily CP-SAT subproblem."""
        if self.use_two_phase_objective:
            return self._solve_day_two_phase(instance, day, candidates)
        return self._solve_day_single_phase(instance, day, candidates)

    def _solve_day_two_phase(self, instance: Instance, day: int, candidates: list[str]) -> tuple[DailyRoute, dict[str, object]]:
        """Run pure two-phase CP: maximize delivered count, then route quality."""
        phase1_limit, phase2_limit = self._phase_time_limits()
        phase1_data = self._build_day_model(instance, day, candidates, use_decision_strategy=self.use_decision_strategy)
        phase1_data.model.Maximize(sum(phase1_data.y.values()))
        phase1_solver = self._new_solver(phase1_limit)
        phase1_status_code = phase1_solver.Solve(phase1_data.model)
        phase1_status = phase1_solver.StatusName(phase1_status_code)
        day_status = self._base_day_status("two_phase", day, phase1_data)
        day_status.update(self._solver_fields("phase1", phase1_status_code, phase1_solver))

        if phase1_status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            day_status.update(
                {
                    "status": phase1_status,
                    "objective": day_status["phase1_objective"],
                    "best_bound": day_status["phase1_best_bound"],
                    "gap_percent": day_status["phase1_gap_percent"],
                    "phase1_delivered_count": 0,
                    "phase1_delivered_count_certified": False,
                    "daily_optimal_for": "no_feasible_daily_solution",
                    "solver_return_time": "",
                }
            )
            return DailyRoute(day=day), day_status

        delivered_count = int(sum(phase1_solver.BooleanValue(var) for var in phase1_data.y.values()))
        phase1_route = self._extract_route_from_solution(instance, day, candidates, phase1_data, phase1_solver)
        day_status["phase1_delivered_count"] = delivered_count
        day_status["phase1_delivered_count_certified"] = phase1_status_code == cp_model.OPTIMAL

        if not self.solve_phase2:
            day_status.update(
                {
                    "status": "OPTIMAL" if phase1_status_code == cp_model.OPTIMAL else "FEASIBLE",
                    "objective": day_status["phase1_objective"],
                    "best_bound": day_status["phase1_best_bound"],
                    "gap_percent": day_status["phase1_gap_percent"],
                    "daily_optimal_for": "delivered_count_only"
                    if phase1_status_code == cp_model.OPTIMAL
                    else "not_certified",
                    "solver_return_time": phase1_solver.Value(phase1_data.return_time),
                    "phase2_status": "",
                    "phase2_objective": "",
                    "phase2_best_bound": "",
                    "phase2_gap_percent": "",
                }
            )
            return phase1_route, day_status

        phase2_data = self._build_day_model(instance, day, candidates, use_decision_strategy=False)
        phase2_data.model.Add(sum(phase2_data.y.values()) == delivered_count)
        self._set_route_cost_objective(phase2_data, instance, day, candidates)
        phase2_solver = self._new_solver(phase2_limit)
        phase2_status_code = phase2_solver.Solve(phase2_data.model)
        phase2_status = phase2_solver.StatusName(phase2_status_code)
        day_status.update(self._solver_fields("phase2", phase2_status_code, phase2_solver))

        if phase2_status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            route = self._extract_route_from_solution(instance, day, candidates, phase2_data, phase2_solver)
            if phase1_status_code == cp_model.OPTIMAL and phase2_status_code == cp_model.OPTIMAL:
                status = "OPTIMAL"
                daily_optimal_for = "delivered_count_and_route_cost"
            elif phase1_status_code == cp_model.OPTIMAL:
                status = "FEASIBLE"
                daily_optimal_for = "delivered_count_only"
            else:
                status = "FEASIBLE"
                daily_optimal_for = "not_certified"
            solver_return = phase2_solver.Value(phase2_data.return_time)
        else:
            route = phase1_route
            status = "FEASIBLE"
            daily_optimal_for = "delivered_count_incumbent_only"
            solver_return = phase1_solver.Value(phase1_data.return_time)

        day_status.update(
            {
                "status": status,
                "objective": day_status["phase2_objective"] if day_status["phase2_objective"] != "" else day_status["phase1_objective"],
                "best_bound": day_status["phase2_best_bound"] if day_status["phase2_best_bound"] != "" else day_status["phase1_best_bound"],
                "gap_percent": day_status["phase2_gap_percent"] if day_status["phase2_gap_percent"] != "" else day_status["phase1_gap_percent"],
                "daily_optimal_for": daily_optimal_for,
                "solver_return_time": solver_return,
            }
        )
        return route, day_status

    def _solve_day_single_phase(self, instance: Instance, day: int, candidates: list[str]) -> tuple[DailyRoute, dict[str, object]]:
        """Run the original pure single-phase CP objective."""
        data = self._build_day_model(instance, day, candidates)
        drop_penalty = self.drop_penalty_by_day.get(day, DROP_PENALTY_BY_DAY[SUNDAY])
        objective_terms: list[cp_model.LinearExpr] = []
        objective_terms.extend(drop_penalty * (1 - data.y[customer_id]) for customer_id in candidates)
        for i in data.nodes:
            for j in data.nodes:
                if i == j:
                    continue
                objective_terms.append(_distance_objective_cost(instance, i, j, self.distance_weight) * data.x[i, j])
        objective_terms.append(self.route_duration_weight * (data.return_time - data.departure))
        for customer_id in candidates:
            objective_terms.append(-self.urgency_weight * self._urgency(instance, day, customer_id) * data.y[customer_id])
        data.model.Minimize(sum(objective_terms))

        solver = self._new_solver(self.time_limit_per_day_sec)
        status_code = solver.Solve(data.model)
        status = solver.StatusName(status_code)
        day_status = self._base_day_status("single_phase", day, data)
        if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            route = self._extract_route_from_solution(instance, day, candidates, data, solver)
            day_status.update(
                {
                    "status": status,
                    "objective": solver.ObjectiveValue(),
                    "best_bound": solver.BestObjectiveBound(),
                    "gap_percent": _gap_percent(solver.ObjectiveValue(), solver.BestObjectiveBound()),
                    "daily_optimal_for": "single_phase_objective" if status_code == cp_model.OPTIMAL else "not_certified",
                    "solver_return_time": solver.Value(data.return_time),
                }
            )
            return route, day_status

        day_status.update(
            {
                "status": status,
                "objective": "",
                "best_bound": "",
                "gap_percent": "",
                "daily_optimal_for": "no_feasible_daily_solution",
                "solver_return_time": "",
            }
        )
        return DailyRoute(day=day), day_status

    def _phase_time_limits(self) -> tuple[float, float]:
        """Return phase time limits from explicit values or a 60/40 split."""
        phase1 = self.phase1_time_limit_sec
        phase2 = self.phase2_time_limit_sec
        if phase1 is None:
            phase1 = max(0.001, 0.6 * self.time_limit_per_day_sec)
        if phase2 is None:
            phase2 = max(0.001, 0.4 * self.time_limit_per_day_sec)
        return float(phase1), float(phase2)

    def _new_solver(self, time_limit_sec: float) -> cp_model.CpSolver:
        """Create a configured CP-SAT solver."""
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_sec
        solver.parameters.num_search_workers = self.num_workers
        solver.parameters.log_search_progress = self.log_search_progress
        solver.parameters.random_seed = self.random_seed
        return solver

    def _set_route_cost_objective(
        self,
        data: DayModelData,
        instance: Instance,
        day: int,
        candidates: list[str],
    ) -> None:
        """Set phase-2 route quality objective with delivered count fixed."""
        objective_terms: list[cp_model.LinearExpr] = []
        for i in data.nodes:
            for j in data.nodes:
                if i == j:
                    continue
                objective_terms.append(_distance_objective_cost(instance, i, j, self.distance_weight) * data.x[i, j])
        objective_terms.append(self.route_duration_weight * (data.return_time - data.departure))
        for customer_id in candidates:
            objective_terms.append(-self.urgency_weight * self._urgency(instance, day, customer_id) * data.y[customer_id])
        data.model.Minimize(sum(objective_terms))

    def _base_day_status(self, objective_mode: str, day: int, data: DayModelData) -> dict[str, object]:
        total_nonself_arcs = data.arc_stats["total_nonself_arcs"]
        fixed_impossible_arcs = data.arc_stats["fixed_impossible_arcs"]
        return {
            "objective_mode": objective_mode,
            "status": "",
            "objective": "",
            "best_bound": "",
            "gap_percent": "",
            "runtime_sec": "",
            "raw_candidate_count": "",
            "selected_candidate_count": "",
            "mandatory_last_day_count": "",
            "fixed_impossible_arcs": fixed_impossible_arcs,
            "total_nonself_arcs": total_nonself_arcs,
            "fixed_arc_ratio": fixed_impossible_arcs / total_nonself_arcs if total_nonself_arcs else 0.0,
            **data.tightening_stats,
            "service_no_overlap_enabled": self.use_service_no_overlap,
            "use_service_no_overlap": self.use_service_no_overlap,
            "phase1_only": not self.solve_phase2,
            "solve_phase2": self.solve_phase2,
            "decision_strategy_enabled": self.use_decision_strategy,
            "drop_penalty": self.drop_penalty_by_day.get(day, DROP_PENALTY_BY_DAY[SUNDAY]),
            "distance_weight": self.distance_weight,
            "route_duration_weight": self.route_duration_weight,
            "urgency_weight": self.urgency_weight,
            "distance_objective_scale": "km",
            "complete_count": "",
            "delivered_today": "",
            "selected_carried_over_count": "",
            "raw_carried_over_count": "",
            "carried_over_count": "",
            "remaining_after_day": "",
            "solver_return_time": "",
            "daily_optimal_for": "",
        }

    def _solver_fields(self, prefix: str, status_code: int, solver: cp_model.CpSolver) -> dict[str, object]:
        status = solver.StatusName(status_code)
        if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            objective = solver.ObjectiveValue()
            best_bound = solver.BestObjectiveBound()
            gap = _gap_percent(objective, best_bound)
        else:
            objective = ""
            best_bound = ""
            gap = ""
        return {
            f"{prefix}_status": status,
            f"{prefix}_objective": objective,
            f"{prefix}_best_bound": best_bound,
            f"{prefix}_gap_percent": gap,
        }

    def _extract_route_from_solution(
        self,
        instance: Instance,
        day: int,
        candidates: list[str],
        data: DayModelData,
        solver: cp_model.CpSolver,
    ) -> DailyRoute:
        """Extract a DailyRoute only from CP solution variables."""
        selected_arcs = {(i, j) for i in data.nodes for j in data.nodes if solver.BooleanValue(data.x[i, j])}
        sequence = _extract_route_from_arcs(selected_arcs, instance.depot_id, candidates)
        start_times = {customer_id: solver.Value(data.t[customer_id]) for customer_id in sequence}
        selected_windows: dict[str, TimeWindow] = {}
        for customer_id in sequence:
            for window_idx, window in enumerate(_windows_for(instance, customer_id, day)):
                var = data.g.get((customer_id, window_idx))
                if var is not None and solver.BooleanValue(var):
                    selected_windows[customer_id] = window
                    break
        return _build_daily_schedule_from_solution(
            instance=instance,
            day=day,
            route_sequence=sequence,
            service_start_times=start_times,
            selected_windows=selected_windows,
            depot_departure_time=solver.Value(data.departure),
        )

    def _urgency(self, instance: Instance, day: int, customer_id: str) -> int:
        """Return integer urgency score for the daily objective."""
        remaining_days = _remaining_available_days(instance, customer_id, day)
        if not remaining_days:
            return 0

        if len(remaining_days) == 1:
            day_pressure = 1000
        else:
            day_pressure = round(200 / len(remaining_days))

        windows_today = _windows_for(instance, customer_id, day)
        if not windows_today:
            return day_pressure

        earliest_end = min(window.end_minute for window in windows_today)
        total_width = sum(window.end_minute - window.start_minute for window in windows_today)
        num_windows = len(windows_today)

        deadline_pressure = round(100 * (DAY_END_MIN - earliest_end) / DAY_END_MIN)
        narrow_pressure = round(500 / max(1, total_width))
        few_window_pressure = round(50 / max(1, num_windows))
        return day_pressure + deadline_pressure + narrow_pressure + few_window_pressure
