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
from vrp_weekly.evaluator import evaluate_weekly_schedule, official_objective_status

LOGGER = logging.getLogger(__name__)
DISTANCE_SCALE = 1000


@dataclass
class DayPrecomputedData:
    """Reusable daily data shared by phase-1 and phase-2 CP models."""

    day: int
    candidates: list[str]
    nodes: list[str]
    travel_time: dict[tuple[str, str], int]
    distance_cost: dict[tuple[str, str], int]
    windows_by_customer: dict[str, list[TimeWindow]]
    service_time: dict[str, int]
    can_follow: dict[tuple[str, str], bool]
    incompatible_window_pairs: set[tuple[str, str, int, int]]
    depot_window_incompatibilities: set[tuple[str, int]]
    return_window_incompatibilities: set[tuple[str, int]]
    dominated_window_indices: set[tuple[str, int]]
    mandatory_customers: list[str]
    urgency: dict[str, int]
    candidate_priorities: dict[str, tuple[int, int, int, int, int, int, str]]


@dataclass
class DayModelData:
    """Variables and metadata for one daily CP-SAT model."""

    model: cp_model.CpModel
    precomputed: DayPrecomputedData
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
    decision_strategy_customer_order: list[str]


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
    can_follow: dict[tuple[str, str], bool] | None = None,
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
            follows = can_follow[i, j] if can_follow is not None else _can_follow(instance, day, i, j)
            if not follows:
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
        time_limit_per_day_sec: int = 60,
        max_candidates_per_day: int | None = 80,
        drop_penalty_by_day: dict[int, int] | None = None,
        distance_weight: int = 10,
        route_duration_weight: int = 0,
        urgency_weight: int = 0,
        num_workers: int = 4,
        log_search_progress: bool = False,
        use_two_phase_objective: bool = True,
        phase1_time_limit_sec: int | None = None,
        phase2_time_limit_sec: int | None = None,
        phase1_time_fraction: float = 0.85,
        phase2_time_fraction: float = 0.15,
        random_seed: int = 1,
        use_decision_strategy: bool = True,
        use_service_no_overlap: bool = True,
        use_route_interval_no_overlap: bool = True,
        use_window_pair_cuts: bool = True,
        use_precedence_cuts: bool = True,
        use_pair_conflict_cuts: bool = True,
        use_depot_window_cuts: bool = True,
        use_dominated_window_cuts: bool = True,
        candidate_strategy: str = "hybrid",
        solve_phase2: bool = True,
        adaptive_daily_deadline: bool = True,
        optimization_mode: str | None = "full_three_stage",
        stage2_max_time_fraction: float = 0.10,
        run_incomplete_diagnostics: bool = False,
        incomplete_diagnostic_time_limit_sec: int = 60,
    ) -> None:
        if candidate_strategy not in {"urgent", "hybrid"}:
            raise ValueError(f"candidate_strategy must be 'urgent' or 'hybrid', got {candidate_strategy!r}")
        valid_optimization_modes = {"full_three_stage", "service_phases_only", "mandatory_stage_only"}
        if optimization_mode is None:
            optimization_mode = "full_three_stage" if solve_phase2 else "service_phases_only"
        elif solve_phase2 is False and optimization_mode == "full_three_stage":
            optimization_mode = "service_phases_only"
        if optimization_mode not in valid_optimization_modes:
            raise ValueError(
                "optimization_mode must be one of "
                f"{sorted(valid_optimization_modes)}, got {optimization_mode!r}"
            )
        if not 0.0 <= stage2_max_time_fraction <= 1.0:
            raise ValueError("stage2_max_time_fraction must be between 0.0 and 1.0")
        if phase1_time_fraction <= 0:
            raise ValueError("phase1_time_fraction must be > 0")
        if phase2_time_fraction < 0:
            raise ValueError("phase2_time_fraction must be >= 0")
        if not (phase1_time_limit_sec is not None and phase2_time_limit_sec is not None):
            if phase1_time_fraction + phase2_time_fraction > 1.0:
                raise ValueError("phase time fractions must sum to <= 1.0 unless explicit phase limits are supplied")
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
        self.phase1_time_fraction = phase1_time_fraction
        self.phase2_time_fraction = phase2_time_fraction
        self.random_seed = random_seed
        self.use_decision_strategy = use_decision_strategy
        self.use_service_no_overlap = use_service_no_overlap
        self.use_route_interval_no_overlap = use_route_interval_no_overlap
        self.use_window_pair_cuts = use_window_pair_cuts
        self.use_precedence_cuts = use_precedence_cuts
        self.use_pair_conflict_cuts = use_pair_conflict_cuts
        self.use_depot_window_cuts = use_depot_window_cuts
        self.use_dominated_window_cuts = use_dominated_window_cuts
        self.candidate_strategy = candidate_strategy
        self.optimization_mode = optimization_mode
        self.solve_phase2 = solve_phase2 if not adaptive_daily_deadline else optimization_mode == "full_three_stage"
        self.adaptive_daily_deadline = adaptive_daily_deadline
        self.stage2_max_time_fraction = stage2_max_time_fraction
        self.run_incomplete_diagnostics = run_incomplete_diagnostics
        self.incomplete_diagnostic_time_limit_sec = incomplete_diagnostic_time_limit_sec
        self._last_candidate_selection_stats: dict[str, object] = {}

    def solve(self, instance: Instance) -> WeeklySchedule:
        """Build a weekly schedule by solving pure daily CP-SAT models."""
        undelivered = set(instance.customer_ids())
        routes: dict[int, DailyRoute] = {}
        day_statuses: dict[int, dict[str, object]] = {}
        delivered_so_far: set[str] = set()

        for day in range(MONDAY, SUNDAY + 1):
            LOGGER.info("cp_rolling day=%s start undelivered=%s", day, len(undelivered))
            day_start = time.perf_counter()
            raw_candidates = sorted(customer for customer in undelivered if _windows_for(instance, customer, day))
            raw_candidate_count = len(raw_candidates)
            mandatory_last_day_count = sum(1 for customer in raw_candidates if _is_last_available_day(instance, customer, day))
            candidates = self._limit_candidates(instance, day, raw_candidates)
            candidate_selection_stats = dict(self._last_candidate_selection_stats)
            selected_candidate_count = len(candidates)
            filtered_candidates = [customer for customer in raw_candidates if customer not in set(candidates)]
            candidate_ranks = {customer: rank for rank, customer in enumerate(candidates, start=1)}
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
                day_statuses[day].update(
                    {
                        "raw_candidate_ids": raw_candidates,
                        "selected_candidate_ids": [],
                        "filtered_candidate_ids": filtered_candidates,
                        "mandatory_candidate_ids": [],
                        "remaining_customer_ids_after_day": sorted(undelivered),
                        "customer_day_diagnostics": self._build_customer_day_diagnostics(
                            instance=instance,
                            day=day,
                            raw_candidates=raw_candidates,
                            candidates=[],
                            filtered_candidates=filtered_candidates,
                            candidate_ranks={},
                            day_status=day_statuses[day],
                            route=routes[day],
                            delivered_so_far=delivered_so_far,
                            undelivered_after_day=undelivered,
                        ),
                    }
                )
                LOGGER.info("cp_rolling day=%s no candidates", day)
                continue

            LOGGER.info("cp_rolling day=%s solve candidates=%s", day, len(candidates))
            route, day_status = self._solve_day(instance, day, candidates, day_start)
            day_status.update(
                self._check_extraction_consistency(
                    instance=instance,
                    day=day,
                    candidates=candidates,
                    route=route,
                    day_status=day_status,
                    delivered_so_far=delivered_so_far,
                )
            )
            delivered_ids = route.delivered_customer_ids()
            undelivered -= delivered_ids
            delivered_so_far |= delivered_ids
            delivered_today = len(delivered_ids)
            day_status.update(
                {
                    "runtime_sec": time.perf_counter() - day_start,
                    "daily_total_runtime_sec": time.perf_counter() - day_start,
                    "raw_candidate_count": raw_candidate_count,
                    "selected_candidate_count": selected_candidate_count,
                    "filtered_candidate_count": len(filtered_candidates),
                    "candidate_limit_active": self.max_candidates_per_day is not None and len(filtered_candidates) > 0,
                    "candidate_limit_value": self.max_candidates_per_day if self.max_candidates_per_day is not None else "",
                    "raw_candidate_ids": raw_candidates,
                    "selected_candidate_ids": candidates,
                    "filtered_candidate_ids": filtered_candidates,
                    "mandatory_candidate_ids": [
                        customer for customer in candidates if _is_last_available_day(instance, customer, day)
                    ],
                    "extracted_route_ids": route.customer_sequence(),
                    "extracted_route_customer_count": len(route.customer_sequence()),
                    "remaining_customer_ids_after_day": sorted(undelivered),
                    "mandatory_last_day_count": mandatory_last_day_count,
                    "mandatory_candidate_count": int(day_status.get("mandatory_candidate_count", 0) or 0),
                    "complete_count": delivered_today,
                    "delivered_today": delivered_today,
                    "selected_carried_over_count": max(0, selected_candidate_count - delivered_today),
                    "raw_carried_over_count": max(0, raw_candidate_count - delivered_today),
                    "carried_over_count": max(0, selected_candidate_count - delivered_today),
                    "remaining_after_day": len(undelivered),
                    **candidate_selection_stats,
                }
            )
            day_status["customer_day_diagnostics"] = self._build_customer_day_diagnostics(
                instance=instance,
                day=day,
                raw_candidates=raw_candidates,
                candidates=candidates,
                filtered_candidates=filtered_candidates,
                candidate_ranks=candidate_ranks,
                day_status=day_status,
                route=route,
                delivered_so_far=delivered_so_far,
                undelivered_after_day=undelivered,
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

        weekly_status = self._weekly_status(day_statuses, len(undelivered))
        incomplete_diagnostics = self._build_incomplete_customer_diagnostics(instance, day_statuses, undelivered)
        if self.run_incomplete_diagnostics and incomplete_diagnostics:
            last_day_diagnostics = self.diagnose_incomplete_customers_on_last_days(
                instance=instance,
                day_statuses=day_statuses,
                incomplete_customer_ids=sorted(undelivered),
                time_limit_sec=self.incomplete_diagnostic_time_limit_sec,
            )
            self._apply_last_day_replay_diagnoses(incomplete_diagnostics, last_day_diagnostics)
            weekly_status["incomplete_last_day_diagnostics"] = last_day_diagnostics
        else:
            weekly_status["incomplete_last_day_diagnostics"] = []
        weekly_status["incomplete_customer_diagnostics"] = incomplete_diagnostics
        weekly_status["stage2_waiting_objective_enabled"] = False
        schedule = WeeklySchedule(routes=routes, solver_status=weekly_status)
        weekly_status.update(official_objective_status(evaluate_weekly_schedule(instance, schedule)))
        return WeeklySchedule(routes=routes, solver_status=weekly_status)

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
            "optimization_mode": self.optimization_mode,
            "official_default_profile": "adaptive_full_three_stage",
            "status": "NO_CANDIDATES",
            "adaptive_daily_deadline": self.adaptive_daily_deadline,
            "objective": "",
            "best_bound": "",
            "runtime_sec": time.perf_counter() - day_start,
            "daily_precompute_time_sec": 0.0,
            "phase1_model_build_time_sec": 0.0,
            "phase1_solve_time_sec": 0.0,
            "phase2_model_build_time_sec": 0.0,
            "phase2_solve_time_sec": 0.0,
            "stage1a_solve_time_sec": 0.0,
            "stage1b_solve_time_sec": 0.0,
            "stage2_solve_time_sec": 0.0,
            "unused_daily_budget_sec": self.time_limit_per_day_sec,
            "daily_total_runtime_sec": time.perf_counter() - day_start,
            "gap_percent": "",
            "gap_scope": "",
            "raw_candidate_count": raw_candidate_count,
            "selected_candidate_count": selected_candidate_count,
            "filtered_candidate_count": max(0, raw_candidate_count - selected_candidate_count),
            "candidate_limit_active": self.max_candidates_per_day is not None and raw_candidate_count > selected_candidate_count,
            "candidate_limit_value": self.max_candidates_per_day if self.max_candidates_per_day is not None else "",
            "raw_candidate_ids": [],
            "selected_candidate_ids": [],
            "filtered_candidate_ids": [],
            "mandatory_candidate_ids": [],
            "stage1a_selected_ids": [],
            "stage1b_selected_ids": [],
            "stage2_selected_ids": [],
            "extracted_route_ids": [],
            "remaining_customer_ids_after_day": [],
            "customer_day_diagnostics": [],
            "mandatory_last_day_count": mandatory_last_day_count,
            "mandatory_candidate_count": 0,
            "mandatory_delivered_count": 0,
            "mandatory_unserved_count": 0,
            "all_mandatory_served": True,
            "mandatory_count_certified": True,
            "total_delivered_count": 0,
            "total_count_certified": True,
            "stage1a_delivered_count": 0,
            "stage1b_delivered_count": 0,
            "stage2_delivered_count": 0,
            "extracted_route_customer_count": 0,
            "stage1a_status": "NO_CANDIDATES",
            "stage1b_status": "",
            "stage2_status": "",
            "stage1a_ran": False,
            "stage1b_ran": False,
            "stage2_ran": False,
            "stage1b_skipped_reason": "no_candidates",
            "stage2_skipped_reason": "no_candidates",
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
            "window_pair_cuts_build_time_sec": 0.0,
            "pair_conflict_cuts_build_time_sec": 0.0,
            "depot_window_cuts_build_time_sec": 0.0,
            "dominated_window_cuts_build_time_sec": 0.0,
            "precedence_cuts_build_time_sec": 0.0,
            "service_no_overlap_build_time_sec": 0.0,
            "route_interval_no_overlap_build_time_sec": 0.0,
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
            "route_interval_no_overlap_enabled": self.use_route_interval_no_overlap,
            "service_interval_count": 0,
            "route_interval_count": 0,
            "depot_interval_enabled": False,
            "no_overlap_route_intervals_enabled": False,
            "roundtrip_duration_lb_count": 0,
            "fixed_impossible_customers_count": 0,
            "phase1_only": not self.solve_phase2,
            "solve_phase2": self.solve_phase2,
            "stage2_max_time_fraction": self.stage2_max_time_fraction,
            "decision_strategy_enabled": self.use_decision_strategy,
            "mandatory_first_decision_strategy": False,
            "phase2_hint_enabled": False,
            "phase2_hint_y_count": 0,
            "phase2_hint_x_count": 0,
            "phase2_hint_g_count": 0,
            "extraction_consistency_error": False,
            "extraction_error_message": "",
            **candidate_selection_stats,
        }

    def _check_extraction_consistency(
        self,
        instance: Instance,
        day: int,
        candidates: list[str],
        route: DailyRoute,
        day_status: dict[str, object],
        delivered_so_far: set[str],
    ) -> dict[str, object]:
        """Return extraction consistency diagnostics for one solved CP day."""
        errors: list[str] = []
        extracted = route.customer_sequence()
        extracted_set = set(extracted)
        if len(extracted) != len(extracted_set):
            errors.append("extracted route contains duplicate customers")
        duplicate_previous = sorted(extracted_set & delivered_so_far)
        if duplicate_previous:
            errors.append(f"customers already delivered on a previous day: {duplicate_previous}")

        final_selected = self._final_selected_ids_from_status(day_status)
        final_selected_set = set(final_selected)
        missing_from_y = sorted(extracted_set - final_selected_set)
        missing_from_route = sorted(final_selected_set - extracted_set)
        if missing_from_y:
            errors.append(f"route customers with final y=0 or missing: {missing_from_y}")
        if missing_from_route:
            errors.append(f"final y=1 customers omitted from route extraction: {missing_from_route}")

        for stop in route.stops:
            if not instance.windows_for_customer_day(stop.customer_id, day):
                errors.append(f"{stop.customer_id} has no window on day {day}")
            if stop.selected_time_window is None:
                errors.append(f"{stop.customer_id} has no selected window")
                continue
            if stop.service_end_time > stop.selected_time_window.end_minute:
                errors.append(f"{stop.customer_id} finishes after selected window")
            if stop.service_start_time < stop.selected_time_window.start_minute:
                errors.append(f"{stop.customer_id} starts before selected window")

        mandatory_count = int(day_status.get("mandatory_candidate_count", 0) or 0)
        mandatory_delivered = int(day_status.get("mandatory_delivered_count", 0) or 0)
        if day_status.get("stage1b_ran") is True:
            stage1b_mandatory = len(set(day_status.get("stage1b_selected_ids", [])) & set(day_status.get("mandatory_candidate_ids", [])))
            if stage1b_mandatory != mandatory_delivered:
                errors.append(
                    f"Stage 1B mandatory count {stage1b_mandatory} != fixed Stage 1A count {mandatory_delivered}"
                )
        if day_status.get("stage2_ran") is True:
            stage2_ids = set(day_status.get("stage2_selected_ids", []))
            stage2_mandatory = len(stage2_ids & set(day_status.get("mandatory_candidate_ids", [])))
            stage2_total = len(stage2_ids)
            total_count = int(day_status.get("total_delivered_count", 0) or 0)
            if stage2_mandatory != mandatory_delivered:
                errors.append(f"Stage 2 mandatory count {stage2_mandatory} != fixed count {mandatory_delivered}")
            if stage2_total != total_count:
                errors.append(f"Stage 2 total count {stage2_total} != fixed Stage 1B total {total_count}")
        if day_status.get("mandatory_count_certified") is True:
            if day_status.get("stage1a_status") != "OPTIMAL" and mandatory_delivered != mandatory_count:
                errors.append("mandatory_count_certified without OPTIMAL Stage 1A or all mandatory served")

        return {
            "extraction_consistency_error": bool(errors),
            "extraction_error_message": "; ".join(errors),
        }

    @staticmethod
    def _final_selected_ids_from_status(day_status: dict[str, object]) -> list[str]:
        """Return selected ids from the last feasible optimization stage in a day status."""
        for key in ("stage2_selected_ids", "stage1b_selected_ids", "stage1a_selected_ids"):
            ids = day_status.get(key)
            if isinstance(ids, list) and ids:
                return [str(customer_id) for customer_id in ids]
        return []

    def _build_customer_day_diagnostics(
        self,
        instance: Instance,
        day: int,
        raw_candidates: list[str],
        candidates: list[str],
        filtered_candidates: list[str],
        candidate_ranks: dict[str, int],
        day_status: dict[str, object],
        route: DailyRoute,
        delivered_so_far: set[str],
        undelivered_after_day: set[str],
    ) -> list[dict[str, object]]:
        """Build JSON-serializable customer/day pipeline diagnostics."""
        raw_set = set(raw_candidates)
        selected_set = set(candidates)
        filtered_set = set(filtered_candidates)
        mandatory_set = {
            customer_id
            for customer_id in raw_set
            if _is_last_available_day(instance, customer_id, day)
        }
        extracted_set = set(route.customer_sequence())
        stage1a_set = set(day_status.get("stage1a_selected_ids", []))
        stage1b_set = set(day_status.get("stage1b_selected_ids", []))
        stage2_set = set(day_status.get("stage2_selected_ids", []))
        diagnostics: list[dict[str, object]] = []
        relevant_customers = sorted(raw_set | selected_set | filtered_set | extracted_set)
        for customer_id in relevant_customers:
            available_days = sorted(instance.available_days(customer_id))
            selected_window = None
            selected_window_index = ""
            for stop in route.stops:
                if stop.customer_id == customer_id:
                    selected_window = stop.selected_time_window
                    if selected_window is not None:
                        for idx, window in enumerate(instance.windows_for_customer_day(customer_id, day)):
                            if window == selected_window:
                                selected_window_index = idx
                                break
                    break
            diagnostics.append(
                {
                    "customer_id": customer_id,
                    "day": day,
                    "available_days": available_days,
                    "last_available_day": max(available_days) if available_days else "",
                    "has_window_today": bool(instance.windows_for_customer_day(customer_id, day)),
                    "in_raw_candidates": customer_id in raw_set,
                    "in_selected_candidates": customer_id in selected_set,
                    "mandatory_today": customer_id in mandatory_set,
                    "candidate_rank": candidate_ranks.get(customer_id, ""),
                    "filtered_by_candidate_limit": customer_id in filtered_set,
                    "filter_reason": "candidate_limit" if customer_id in filtered_set else "",
                    "stage1a_value": 1 if customer_id in stage1a_set else 0 if customer_id in selected_set else "",
                    "stage1b_value": 1 if customer_id in stage1b_set else 0 if customer_id in selected_set and day_status.get("stage1b_ran") else "",
                    "stage2_value": 1 if customer_id in stage2_set else 0 if customer_id in selected_set and day_status.get("stage2_ran") else "",
                    "selected_window_index": selected_window_index,
                    "selected_window_start": selected_window.start_minute if selected_window else "",
                    "selected_window_end": selected_window.end_minute if selected_window else "",
                    "extracted_in_route": customer_id in extracted_set,
                    "delivered_after_day": customer_id in delivered_so_far,
                    "remaining_after_day": customer_id in undelivered_after_day,
                    "unserved_reason": self._diagnose_customer_day_reason(customer_id, day_status, extracted_set),
                }
            )
        return diagnostics

    def _diagnose_customer_day_reason(
        self,
        customer_id: str,
        day_status: dict[str, object],
        extracted_set: set[str],
    ) -> str:
        """Return a concise per-day unserved reason."""
        if customer_id in extracted_set:
            return ""
        if customer_id in set(day_status.get("filtered_candidate_ids", [])):
            return "filtered_by_candidate_limit"
        stage1a = customer_id in set(day_status.get("stage1a_selected_ids", []))
        stage1b = customer_id in set(day_status.get("stage1b_selected_ids", []))
        stage2 = customer_id in set(day_status.get("stage2_selected_ids", []))
        if stage1a or stage1b or stage2:
            return "extraction_mismatch"
        if customer_id in set(day_status.get("mandatory_candidate_ids", [])):
            return "mandatory_conflict_certified" if day_status.get("mandatory_count_certified") else "stage1a_timeout_or_conflict"
        if not day_status.get("stage1b_ran"):
            return "stage1b_not_run"
        return "stage1b_not_selected"

    def _build_incomplete_customer_diagnostics(
        self,
        instance: Instance,
        day_statuses: dict[int, dict[str, object]],
        incomplete_customer_ids: set[str],
    ) -> list[dict[str, object]]:
        """Build final weekly diagnostics for customers incomplete after Sunday."""
        diagnostics: list[dict[str, object]] = []
        by_customer_day: dict[tuple[str, int], dict[str, object]] = {}
        for day, day_status in day_statuses.items():
            for row in day_status.get("customer_day_diagnostics", []):
                if isinstance(row, dict):
                    by_customer_day[str(row["customer_id"]), int(day)] = row
        for customer_id in sorted(incomplete_customer_ids):
            available_days = sorted(instance.available_days(customer_id))
            if not available_days:
                reason = "no_available_day"
                last_day = ""
            else:
                last_day = max(available_days)
                last_row = by_customer_day.get((customer_id, last_day), {})
                reason = self._diagnose_incomplete_reason(customer_id, last_day, day_statuses, last_row)
            last_status = day_statuses.get(last_day, {}) if isinstance(last_day, int) else {}
            last_row = by_customer_day.get((customer_id, last_day), {}) if isinstance(last_day, int) else {}
            diagnostics.append(
                {
                    "customer_id": customer_id,
                    "available_days": available_days,
                    "earliest_available_day": min(available_days) if available_days else "",
                    "last_available_day": last_day,
                    "appeared_in_raw_candidates_by_day": [
                        day for day in available_days if by_customer_day.get((customer_id, day), {}).get("in_raw_candidates")
                    ],
                    "appeared_in_selected_candidates_by_day": [
                        day for day in available_days if by_customer_day.get((customer_id, day), {}).get("in_selected_candidates")
                    ],
                    "mandatory_on_days": [
                        day for day in available_days if by_customer_day.get((customer_id, day), {}).get("mandatory_today")
                    ],
                    "stage1a_values_by_day": {
                        str(day): by_customer_day.get((customer_id, day), {}).get("stage1a_value", "")
                        for day in available_days
                    },
                    "stage1b_values_by_day": {
                        str(day): by_customer_day.get((customer_id, day), {}).get("stage1b_value", "")
                        for day in available_days
                    },
                    "stage2_values_by_day": {
                        str(day): by_customer_day.get((customer_id, day), {}).get("stage2_value", "")
                        for day in available_days
                    },
                    "extracted_days": [
                        day for day in available_days if by_customer_day.get((customer_id, day), {}).get("extracted_in_route")
                    ],
                    "candidate_rank_on_last_available_day": last_row.get("candidate_rank", ""),
                    "selected_on_last_available_day": bool(last_row.get("in_selected_candidates", False)),
                    "mandatory_on_last_available_day": bool(last_row.get("mandatory_today", False)),
                    "last_day_stage1a_status": last_status.get("stage1a_status", ""),
                    "last_day_stage1b_status": last_status.get("stage1b_status", ""),
                    "last_day_stage2_status": last_status.get("stage2_status", ""),
                    "last_day_mandatory_count_certified": last_status.get("mandatory_count_certified", ""),
                    "last_day_total_count_certified": last_status.get("total_count_certified", ""),
                    "last_day_remaining_budget_sec": last_status.get("unused_daily_budget_sec", ""),
                    "diagnosis_reason": reason,
                }
            )
        return diagnostics

    def _diagnose_incomplete_reason(
        self,
        customer_id: str,
        last_day: int,
        day_statuses: dict[int, dict[str, object]],
        last_row: dict[str, object],
    ) -> str:
        """Classify an incomplete customer using the final last-available-day diagnostics."""
        if not last_row:
            return "no_valid_window"
        if not last_row.get("in_selected_candidates", False):
            return "filtered_by_candidate_limit"
        if last_row.get("extracted_in_route", False):
            return "unknown"
        if any(last_row.get(field) == 1 for field in ("stage1a_value", "stage1b_value", "stage2_value")):
            return "extraction_mismatch"
        day_status = day_statuses.get(last_day, {})
        if last_row.get("mandatory_today") and last_row.get("stage1a_value") == 0:
            return "mandatory_conflict_certified" if day_status.get("mandatory_count_certified") else "stage1a_timeout_or_conflict"
        if not day_status.get("stage1b_ran"):
            return "stage1b_not_run"
        if last_row.get("stage1b_value") == 0:
            return "stage1b_not_selected"
        if not day_status.get("stage2_ran"):
            return "stage2_not_run"
        return "unknown"

    def diagnose_incomplete_customers_on_last_days(
        self,
        instance: Instance,
        day_statuses: dict[int, dict[str, object]],
        incomplete_customer_ids: list[str],
        time_limit_sec: int = 60,
    ) -> list[dict[str, object]]:
        """Run no-cap mandatory-only diagnostic solves for affected last days."""
        affected_days = sorted(
            {
                max(instance.available_days(customer_id))
                for customer_id in incomplete_customer_ids
                if instance.available_days(customer_id)
            }
        )
        diagnostics: list[dict[str, object]] = []
        delivered_before: set[str] = set()
        extracted_by_day = {
            day: set(status.get("extracted_route_ids", []))
            for day, status in day_statuses.items()
            if isinstance(status, dict)
        }
        for diagnostic_day in affected_days:
            for day in range(MONDAY, diagnostic_day):
                delivered_before.update(extracted_by_day.get(day, set()))
            undelivered_before = set(instance.customer_ids()) - delivered_before
            raw_candidates = sorted(
                customer for customer in undelivered_before if instance.windows_for_customer_day(customer, diagnostic_day)
            )
            diagnostic_solver = RollingHorizonCPSATSolver(
                time_limit_per_day_sec=time_limit_sec,
                max_candidates_per_day=None,
                num_workers=self.num_workers,
                random_seed=self.random_seed,
                optimization_mode="mandatory_stage_only",
                run_incomplete_diagnostics=False,
            )
            start = time.perf_counter()
            route, status = diagnostic_solver._solve_day(instance, diagnostic_day, raw_candidates, time.perf_counter())
            diagnostics.append(
                {
                    "diagnostic_last_day": diagnostic_day,
                    "diagnostic_raw_candidate_count": len(raw_candidates),
                    "diagnostic_mandatory_count": status.get("mandatory_candidate_count", 0),
                    "diagnostic_mandatory_delivered_count": status.get("mandatory_delivered_count", 0),
                    "diagnostic_status": status.get("stage1a_status", status.get("status", "")),
                    "diagnostic_best_bound": status.get("stage1a_best_bound", ""),
                    "diagnostic_all_mandatory_served": status.get("all_mandatory_served", False),
                    "diagnostic_served_customer_ids": route.customer_sequence(),
                    "diagnostic_runtime_sec": time.perf_counter() - start,
                }
            )
        return diagnostics

    @staticmethod
    def _apply_last_day_replay_diagnoses(
        incomplete_diagnostics: list[dict[str, object]],
        last_day_diagnostics: list[dict[str, object]],
    ) -> None:
        """Mark customers serviceable in no-cap diagnostic replay as rolling-horizon myopia."""
        served_by_day: dict[int, set[str]] = {}
        for replay in last_day_diagnostics:
            day = replay.get("diagnostic_last_day")
            if not isinstance(day, int):
                continue
            served_by_day.setdefault(day, set()).update(str(customer_id) for customer_id in replay.get("diagnostic_served_customer_ids", []))
        for row in incomplete_diagnostics:
            customer_id = str(row.get("customer_id", ""))
            last_day = row.get("last_available_day")
            if isinstance(last_day, int) and customer_id in served_by_day.get(last_day, set()):
                row["diagnosis_reason"] = "rolling_horizon_myopia"

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
            "gap_percent": "",
            "gap_percent_basis": "not_computed_for_rolling_horizon",
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

    def _precompute_day(self, instance: Instance, day: int, candidates: list[str]) -> DayPrecomputedData:
        """Precompute daily compatibility and cost data once for both CP phases."""
        depot_id = instance.depot_id
        nodes = [depot_id] + candidates
        windows_by_customer = {customer_id: _windows_for(instance, customer_id, day) for customer_id in candidates}
        service_time = {node_id: _get_service_time(instance, node_id) for node_id in nodes}
        travel_time = {
            (i, j): (0 if i == j else _travel_time_minutes(instance, i, j))
            for i in nodes
            for j in nodes
        }
        distance_cost = {
            (i, j): (0 if i == j else _distance_objective_cost(instance, i, j, self.distance_weight))
            for i in nodes
            for j in nodes
        }

        def window_fits(customer_id: str, window: TimeWindow) -> bool:
            return _service_fits_window(service_time[customer_id], window)

        def depot_can_reach(customer_id: str, window: TimeWindow) -> bool:
            return window_fits(customer_id, window) and travel_time[depot_id, customer_id] <= (
                window.end_minute - service_time[customer_id]
            )

        def window_can_return(customer_id: str, window: TimeWindow) -> bool:
            return (
                window_fits(customer_id, window)
                and window.start_minute + service_time[customer_id] + travel_time[customer_id, depot_id] <= DAY_END_MIN
            )

        def customer_window_pair_can_follow(i: str, j: str, window_i: TimeWindow, window_j: TimeWindow) -> bool:
            return (
                window_fits(i, window_i)
                and window_fits(j, window_j)
                and window_i.start_minute + service_time[i] + travel_time[i, j] <= window_j.end_minute - service_time[j]
            )

        def arc_can_follow(i: str, j: str) -> bool:
            if i == j:
                return True
            if i == depot_id and j != depot_id:
                return any(depot_can_reach(j, window) for window in windows_by_customer[j])
            if i != depot_id and j == depot_id:
                return any(window_can_return(i, window) for window in windows_by_customer[i])
            if i != depot_id and j != depot_id:
                return any(
                    customer_window_pair_can_follow(i, j, window_i, window_j)
                    for window_i in windows_by_customer[i]
                    for window_j in windows_by_customer[j]
                )
            return True

        can_follow = {(i, j): arc_can_follow(i, j) for i in nodes for j in nodes}
        incompatible_window_pairs: set[tuple[str, str, int, int]] = set()
        for i in candidates:
            for j in candidates:
                if i == j:
                    continue
                for window_i_idx, window_i in enumerate(windows_by_customer[i]):
                    for window_j_idx, window_j in enumerate(windows_by_customer[j]):
                        if not customer_window_pair_can_follow(i, j, window_i, window_j):
                            incompatible_window_pairs.add((i, j, window_i_idx, window_j_idx))

        depot_window_incompatibilities = {
            (customer_id, window_idx)
            for customer_id in candidates
            for window_idx, window in enumerate(windows_by_customer[customer_id])
            if not depot_can_reach(customer_id, window)
        }
        return_window_incompatibilities = {
            (customer_id, window_idx)
            for customer_id in candidates
            for window_idx, window in enumerate(windows_by_customer[customer_id])
            if not window_can_return(customer_id, window)
        }
        dominated_window_indices = {
            (customer_id, window_idx)
            for customer_id in candidates
            for window_idx, window in enumerate(windows_by_customer[customer_id])
            if any(
                other_idx != window_idx and _is_window_dominated(window, other, window_idx, other_idx)
                for other_idx, other in enumerate(windows_by_customer[customer_id])
            )
        }
        mandatory_customers = [customer_id for customer_id in candidates if _is_last_available_day(instance, customer_id, day)]
        urgency = {customer_id: self._urgency(instance, day, customer_id) for customer_id in candidates}
        candidate_priorities = {
            customer_id: _candidate_priority(instance, day, customer_id)
            for customer_id in candidates
        }
        return DayPrecomputedData(
            day=day,
            candidates=list(candidates),
            nodes=nodes,
            travel_time=travel_time,
            distance_cost=distance_cost,
            windows_by_customer=windows_by_customer,
            service_time=service_time,
            can_follow=can_follow,
            incompatible_window_pairs=incompatible_window_pairs,
            depot_window_incompatibilities=depot_window_incompatibilities,
            return_window_incompatibilities=return_window_incompatibilities,
            dominated_window_indices=dominated_window_indices,
            mandatory_customers=mandatory_customers,
            urgency=urgency,
            candidate_priorities=candidate_priorities,
        )

    def _build_day_model(
        self,
        instance: Instance,
        day: int,
        candidates: list[str],
        use_decision_strategy: bool | None = None,
        precomputed: DayPrecomputedData | None = None,
    ) -> DayModelData:
        """Build one pure daily CP-SAT model without an objective."""
        if use_decision_strategy is None:
            use_decision_strategy = self.use_decision_strategy
        if precomputed is None:
            precomputed = self._precompute_day(instance, day, candidates)
        model = cp_model.CpModel()
        depot_id = instance.depot_id
        nodes = precomputed.nodes
        node_index = {node_id: index for index, node_id in enumerate(nodes)}

        y = {customer_id: model.NewBoolVar(f"y[{customer_id}]") for customer_id in candidates}
        t = {customer_id: model.NewIntVar(0, DAY_END_MIN, f"T[{customer_id}]") for customer_id in candidates}
        g: dict[tuple[str, int], cp_model.IntVar] = {}
        x = {(i, j): model.NewBoolVar(f"x[{i},{j}]") for i in nodes for j in nodes}
        z = model.NewBoolVar("z")
        departure = model.NewIntVar(0, DAY_END_MIN, "L")
        return_time = model.NewIntVar(0, DAY_END_MIN, "R")
        max_travel = max(
            (precomputed.travel_time[i, j] for i in nodes for j in nodes if i != j),
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
            "depot_interval_enabled": self.use_route_interval_no_overlap,
            "no_overlap_route_intervals_enabled": self.use_route_interval_no_overlap,
            "roundtrip_duration_lb_count": 0,
            "fixed_impossible_customers_count": 0,
            "window_pair_cuts_build_time_sec": 0.0,
            "pair_conflict_cuts_build_time_sec": 0.0,
            "depot_window_cuts_build_time_sec": 0.0,
            "dominated_window_cuts_build_time_sec": 0.0,
            "precedence_cuts_build_time_sec": 0.0,
            "service_no_overlap_build_time_sec": 0.0,
            "route_interval_no_overlap_build_time_sec": 0.0,
        }
        service_intervals: list[cp_model.IntervalVar] = []
        route_intervals: list[cp_model.IntervalVar] = []

        for customer_id in candidates:
            window_vars: list[cp_model.IntVar] = []
            service_time = precomputed.service_time[customer_id]
            windows_today = precomputed.windows_by_customer[customer_id]
            for window_idx, window in enumerate(windows_today):
                var = model.NewBoolVar(f"g[{customer_id},{window_idx}]")
                g[customer_id, window_idx] = var
                window_lookup[customer_id, window_idx] = window
                window_vars.append(var)
                model.Add(t[customer_id] >= window.start_minute).OnlyEnforceIf(var)
                model.Add(t[customer_id] + service_time <= window.end_minute).OnlyEnforceIf(var)
            dominated_start = time.perf_counter()
            if self.use_dominated_window_cuts:
                for window_idx, _window in enumerate(windows_today):
                    if (customer_id, window_idx) not in precomputed.dominated_window_indices:
                        continue
                    model.Add(g[customer_id, window_idx] == 0)
                    tightening_stats["dominated_window_cuts_count"] += 1
            tightening_stats["dominated_window_cuts_build_time_sec"] += time.perf_counter() - dominated_start
            model.Add(sum(window_vars) == y[customer_id])
            model.Add(x[customer_id, customer_id] + y[customer_id] == 1)
            model.Add(y[customer_id] <= z)
            model.Add(t[customer_id] == 0).OnlyEnforceIf(y[customer_id].Not())
            model.Add(
                next_travel[customer_id]
                == sum(
                    precomputed.travel_time[customer_id, j] * x[customer_id, j]
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
            if self.use_route_interval_no_overlap:
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
                not precomputed.can_follow[depot_id, customer_id]
                or not precomputed.can_follow[customer_id, depot_id]
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

        service_no_overlap_start = time.perf_counter()
        if self.use_service_no_overlap and service_intervals:
            model.AddNoOverlap(service_intervals)
        tightening_stats["service_no_overlap_build_time_sec"] += time.perf_counter() - service_no_overlap_start

        model.Add(
            depot_first_travel
            == sum(precomputed.travel_time[depot_id, j] * x[depot_id, j] for j in candidates)
        )
        model.Add(depot_interval_end == departure + depot_first_travel).OnlyEnforceIf(z)
        model.Add(depot_first_travel == 0).OnlyEnforceIf(z.Not())
        model.Add(depot_interval_end == 0).OnlyEnforceIf(z.Not())
        route_no_overlap_start = time.perf_counter()
        if self.use_route_interval_no_overlap:
            depot_interval = model.NewOptionalIntervalVar(
                departure,
                depot_first_travel,
                depot_interval_end,
                z,
                "route_interval[depot]",
            )
            model.AddNoOverlap([depot_interval] + route_intervals)
        tightening_stats["route_interval_no_overlap_build_time_sec"] += time.perf_counter() - route_no_overlap_start

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
            can_follow=precomputed.can_follow,
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
        model.Add(return_time - departure >= sum(precomputed.service_time[customer_id] * y[customer_id] for customer_id in candidates))
        for customer_id in candidates:
            roundtrip_min = (
                precomputed.travel_time[depot_id, customer_id]
                + precomputed.service_time[customer_id]
                + precomputed.travel_time[customer_id, depot_id]
            )
            model.Add(return_time - departure >= roundtrip_min * y[customer_id])
            tightening_stats["roundtrip_duration_lb_count"] += 1

        for i in candidates:
            for j in candidates:
                if i == j:
                    continue
                model.Add(interval_end[i] <= t[j]).OnlyEnforceIf(x[i, j])
                window_pair_start = time.perf_counter()
                if self.use_window_pair_cuts:
                    for window_i_idx, _window_i in enumerate(precomputed.windows_by_customer[i]):
                        for window_j_idx, _window_j in enumerate(precomputed.windows_by_customer[j]):
                            if (i, j, window_i_idx, window_j_idx) in precomputed.incompatible_window_pairs:
                                model.Add(x[i, j] + g[i, window_i_idx] + g[j, window_j_idx] <= 2)
                                tightening_stats["window_pair_cuts_count"] += 1
                tightening_stats["window_pair_cuts_build_time_sec"] += time.perf_counter() - window_pair_start
                precedence_start = time.perf_counter()
                if (
                    self.use_precedence_cuts
                    and precomputed.can_follow[i, j]
                    and not precomputed.can_follow[j, i]
                ):
                    model.Add(interval_end[i] <= t[j]).OnlyEnforceIf([y[i], y[j]])
                    tightening_stats["precedence_cuts_count"] += 1
                tightening_stats["precedence_cuts_build_time_sec"] += time.perf_counter() - precedence_start

        pair_conflict_start = time.perf_counter()
        for i_idx, i in enumerate(candidates):
            for j in candidates[i_idx + 1 :]:
                if (
                    self.use_pair_conflict_cuts
                    and not precomputed.can_follow[i, j]
                    and not precomputed.can_follow[j, i]
                ):
                    model.Add(y[i] + y[j] <= 1)
                    tightening_stats["pair_conflict_cuts_count"] += 1
        tightening_stats["pair_conflict_cuts_build_time_sec"] += time.perf_counter() - pair_conflict_start

        depot_window_start = time.perf_counter()
        for j in candidates:
            model.Add(depot_interval_end == t[j]).OnlyEnforceIf(x[depot_id, j])
            for window_idx, _window in enumerate(precomputed.windows_by_customer[j]):
                if self.use_depot_window_cuts and (j, window_idx) in precomputed.depot_window_incompatibilities:
                    model.Add(x[depot_id, j] + g[j, window_idx] <= 1)
                    tightening_stats["depot_window_cuts_count"] += 1

        for i in candidates:
            model.Add(return_time == interval_end[i]).OnlyEnforceIf(x[i, depot_id])
            for window_idx, _window in enumerate(precomputed.windows_by_customer[i]):
                if self.use_depot_window_cuts and (i, window_idx) in precomputed.return_window_incompatibilities:
                    model.Add(x[i, depot_id] + g[i, window_idx] <= 1)
                    tightening_stats["depot_window_cuts_count"] += 1
        tightening_stats["depot_window_cuts_build_time_sec"] += time.perf_counter() - depot_window_start

        decision_strategy_customer_order: list[str] = []
        if use_decision_strategy:
            mandatory_set = set(precomputed.mandatory_customers)
            decision_strategy_customer_order = [
                *[customer_id for customer_id in precomputed.mandatory_customers if customer_id in y],
                *[customer_id for customer_id in candidates if customer_id not in mandatory_set],
            ]
            model.AddDecisionStrategy(
                [y[customer_id] for customer_id in decision_strategy_customer_order],
                cp_model.CHOOSE_FIRST,
                cp_model.SELECT_MAX_VALUE,
            )

        return DayModelData(
            model=model,
            precomputed=precomputed,
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
            decision_strategy_customer_order=decision_strategy_customer_order,
        )

    def _solve_day(
        self,
        instance: Instance,
        day: int,
        candidates: list[str],
        day_start: float | None = None,
    ) -> tuple[DailyRoute, dict[str, object]]:
        """Solve one daily CP-SAT subproblem."""
        if day_start is None:
            day_start = time.perf_counter()
        precompute_start = time.perf_counter()
        precomputed = self._precompute_day(instance, day, candidates)
        precompute_time = time.perf_counter() - precompute_start
        if self.use_two_phase_objective:
            return self._solve_day_two_phase(instance, day, candidates, precomputed, precompute_time, day_start)
        return self._solve_day_single_phase(instance, day, candidates, precomputed, precompute_time)

    def _solve_day_two_phase(
        self,
        instance: Instance,
        day: int,
        candidates: list[str],
        precomputed: DayPrecomputedData,
        precompute_time: float,
        day_start: float | None = None,
    ) -> tuple[DailyRoute, dict[str, object]]:
        """Run pure two-phase CP: maximize delivered count, then route quality."""
        if self.adaptive_daily_deadline:
            return self._solve_day_adaptive_three_stage(instance, day, candidates, precomputed, precompute_time, day_start)
        return self._solve_day_weighted_two_phase(instance, day, candidates, precomputed, precompute_time)

    def _solve_day_adaptive_three_stage(
        self,
        instance: Instance,
        day: int,
        candidates: list[str],
        precomputed: DayPrecomputedData,
        precompute_time: float,
        day_start: float | None,
    ) -> tuple[DailyRoute, dict[str, object]]:
        """Run adaptive daily CP under one shared wall-clock deadline."""
        if day_start is None:
            day_start = time.perf_counter()
        daily_deadline = day_start + self.time_limit_per_day_sec
        mandatory_count = len(precomputed.mandatory_customers)
        mandatory_delivered_count = 0
        total_delivered_count = 0
        mandatory_count_certified = mandatory_count == 0
        total_count_certified = False
        best_route = DailyRoute(day=day)
        best_solver_return: int | str = ""
        best_objective: float | str = ""
        best_bound: float | str = ""
        best_gap: float | str = ""
        best_gap_scope = ""
        daily_optimal_for = "no_feasible_daily_solution"
        stage1a_data: DayModelData | None = None
        stage1a_solver: cp_model.CpSolver | None = None
        stage1b_data: DayModelData | None = None
        stage1b_solver: cp_model.CpSolver | None = None
        stage1b_feasible = False

        stage1a_build_start = time.perf_counter()
        stage1a_data = self._build_day_model(
            instance,
            day,
            candidates,
            use_decision_strategy=self.use_decision_strategy,
            precomputed=precomputed,
        )
        stage1a_model_build_time = time.perf_counter() - stage1a_build_start
        day_status = self._base_day_status("adaptive_three_stage", day, stage1a_data)
        day_status.update(
            {
                "adaptive_daily_deadline": True,
                "optimization_mode": self.optimization_mode,
                "daily_precompute_time_sec": precompute_time,
                "stage1a_model_build_time_sec": stage1a_model_build_time,
                "stage1b_model_build_time_sec": 0.0,
                "stage2_model_build_time_sec": 0.0,
                "phase1_model_build_time_sec": stage1a_model_build_time,
                "phase2_model_build_time_sec": 0.0,
                "stage1a_solve_time_sec": 0.0,
                "stage1b_solve_time_sec": 0.0,
                "stage2_solve_time_sec": 0.0,
                "phase1_solve_time_sec": 0.0,
                "phase2_solve_time_sec": 0.0,
                "daily_total_runtime_sec": "",
                "unused_daily_budget_sec": "",
                "stage1a_status": "",
                "stage1b_status": "",
                "stage2_status": "",
                "stage1a_ran": False,
                "stage1b_ran": False,
                "stage2_ran": False,
                "stage1b_skipped_reason": "",
                "stage2_skipped_reason": "",
                "mandatory_delivered_count": 0,
                "total_delivered_count": 0,
                "total_count_certified": False,
                "mandatory_count_certified": mandatory_count == 0,
                "all_mandatory_served": mandatory_count == 0,
                "phase2_hint_enabled": False,
                "phase2_hint_y_count": 0,
                "phase2_hint_x_count": 0,
                "phase2_hint_g_count": 0,
            }
        )

        if mandatory_count == 0:
            day_status.update(
                {
                    "stage1a_status": "SKIPPED_NO_MANDATORY",
                    "mandatory_delivered_count": 0,
                    "mandatory_unserved_count": 0,
                    "all_mandatory_served": True,
                    "mandatory_count_certified": True,
                    "phase1_mandatory_delivered_count": 0,
                    "phase1_mandatory_count_certified": True,
                }
            )
        else:
            stage1a_data.model.Maximize(sum(stage1a_data.y[customer_id] for customer_id in precomputed.mandatory_customers))
            remaining_time = self._remaining_daily_time(daily_deadline)
            if remaining_time <= 0.0:
                day_status["stage1a_status"] = "NO_TIME"
                day_status["stage1b_skipped_reason"] = "stage1a_no_time"
                day_status["stage2_skipped_reason"] = "stage1a_no_time"
                return self._finalize_adaptive_day_status(
                    day_status=day_status,
                    route=best_route,
                    status="NO_TIME",
                    objective=best_objective,
                    best_bound=best_bound,
                    gap_percent=best_gap,
                    gap_scope=best_gap_scope,
                    daily_optimal_for=daily_optimal_for,
                    solver_return_time=best_solver_return,
                    daily_deadline=daily_deadline,
                )
            stage1a_solver = self._new_solver(remaining_time)
            stage1a_solve_start = time.perf_counter()
            day_status["stage1a_ran"] = True
            stage1a_status_code = stage1a_solver.Solve(stage1a_data.model)
            day_status["stage1a_solve_time_sec"] = time.perf_counter() - stage1a_solve_start
            stage1a_status = stage1a_solver.StatusName(stage1a_status_code)
            day_status["stage1a_status"] = stage1a_status
            day_status.update(self._solver_fields("stage1a", stage1a_status_code, stage1a_solver))
            if stage1a_status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                day_status["stage1b_skipped_reason"] = "stage1a_no_feasible_solution"
                day_status["stage2_skipped_reason"] = "stage1a_no_feasible_solution"
                return self._finalize_adaptive_day_status(
                    day_status=day_status,
                    route=best_route,
                    status=stage1a_status,
                    objective="",
                    best_bound="",
                    gap_percent="",
                    gap_scope="",
                    daily_optimal_for=daily_optimal_for,
                    solver_return_time="",
                    daily_deadline=daily_deadline,
                )
            mandatory_delivered_count = int(
                sum(stage1a_solver.BooleanValue(stage1a_data.y[customer_id]) for customer_id in precomputed.mandatory_customers)
            )
            total_delivered_count = int(sum(stage1a_solver.BooleanValue(var) for var in stage1a_data.y.values()))
            day_status["stage1a_selected_ids"] = [
                customer_id for customer_id, var in stage1a_data.y.items() if stage1a_solver.BooleanValue(var)
            ]
            day_status["stage1a_delivered_count"] = len(day_status["stage1a_selected_ids"])
            mandatory_count_certified = stage1a_status_code == cp_model.OPTIMAL or mandatory_delivered_count == mandatory_count
            best_route = self._extract_route_from_solution(instance, day, candidates, stage1a_data, stage1a_solver)
            best_solver_return = stage1a_solver.Value(stage1a_data.return_time)
            best_objective = stage1a_solver.ObjectiveValue()
            best_bound = stage1a_solver.BestObjectiveBound()
            best_gap = _gap_percent(best_objective, best_bound)
            best_gap_scope = "stage1a_mandatory_objective"
            daily_optimal_for = "mandatory_count_only" if mandatory_count_certified else "not_certified"

        day_status.update(
            {
                "mandatory_delivered_count": mandatory_delivered_count,
                "mandatory_unserved_count": mandatory_count - mandatory_delivered_count,
                "all_mandatory_served": mandatory_delivered_count == mandatory_count,
                "mandatory_count_certified": mandatory_count_certified,
                "phase1_mandatory_delivered_count": mandatory_delivered_count,
                "phase1_mandatory_count_certified": mandatory_count_certified,
                "total_delivered_count": total_delivered_count,
            }
        )

        if self.optimization_mode == "mandatory_stage_only":
            day_status["stage1b_status"] = "SKIPPED"
            day_status["stage1b_skipped_reason"] = "optimization_mode_mandatory_stage_only"
            day_status["stage2_status"] = "SKIPPED"
            day_status["stage2_skipped_reason"] = "optimization_mode_mandatory_stage_only"
        elif self._remaining_daily_time(daily_deadline) > 0.0:
            stage1b_build_start = time.perf_counter()
            stage1b_data = self._build_day_model(
                instance,
                day,
                candidates,
                use_decision_strategy=self.use_decision_strategy,
                precomputed=precomputed,
            )
            stage1b_model_build_time = time.perf_counter() - stage1b_build_start
            day_status["stage1b_model_build_time_sec"] = stage1b_model_build_time
            day_status["phase1_model_build_time_sec"] = stage1a_model_build_time + stage1b_model_build_time
            self._add_mandatory_count_constraint(stage1b_data, mandatory_delivered_count)
            stage1b_data.model.Maximize(sum(stage1b_data.y.values()))
            if stage1a_solver is not None and stage1a_data is not None:
                day_status.update(self._add_stage_hint("stage1b", stage1b_data, stage1a_data, stage1a_solver))
            remaining_time = self._remaining_daily_time(daily_deadline)
            if remaining_time > 0.0:
                stage1b_solver = self._new_solver(remaining_time)
                stage1b_solve_start = time.perf_counter()
                day_status["stage1b_ran"] = True
                stage1b_status_code = stage1b_solver.Solve(stage1b_data.model)
                day_status["stage1b_solve_time_sec"] = time.perf_counter() - stage1b_solve_start
                stage1b_status = stage1b_solver.StatusName(stage1b_status_code)
                day_status["stage1b_status"] = stage1b_status
                day_status.update(self._solver_fields("stage1b", stage1b_status_code, stage1b_solver))
                if stage1b_status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    stage1b_feasible = True
                    total_delivered_count = int(sum(stage1b_solver.BooleanValue(var) for var in stage1b_data.y.values()))
                    day_status["stage1b_selected_ids"] = [
                        customer_id for customer_id, var in stage1b_data.y.items() if stage1b_solver.BooleanValue(var)
                    ]
                    day_status["stage1b_delivered_count"] = len(day_status["stage1b_selected_ids"])
                    total_count_certified = stage1b_status_code == cp_model.OPTIMAL
                    best_route = self._extract_route_from_solution(instance, day, candidates, stage1b_data, stage1b_solver)
                    best_solver_return = stage1b_solver.Value(stage1b_data.return_time)
                    best_objective = stage1b_solver.ObjectiveValue()
                    best_bound = stage1b_solver.BestObjectiveBound()
                    best_gap = _gap_percent(best_objective, best_bound)
                    best_gap_scope = "stage1b_total_delivered_objective"
                    daily_optimal_for = "mandatory_and_total_count" if total_count_certified else "mandatory_count_incumbent_total_count"
            else:
                day_status["stage1b_status"] = "NO_TIME"
                day_status["stage1b_skipped_reason"] = "no_time_after_model_build"
        else:
            day_status["stage1b_status"] = "NO_TIME"
            day_status["stage1b_skipped_reason"] = "no_time_after_stage1a"

        service_status = day_status.get("stage1b_status", "") if day_status.get("stage1b_ran") else day_status.get("stage1a_status", "")
        service_objective = day_status.get("stage1b_objective", "") if day_status.get("stage1b_ran") else day_status.get("stage1a_objective", "")
        service_best_bound = day_status.get("stage1b_best_bound", "") if day_status.get("stage1b_ran") else day_status.get("stage1a_best_bound", "")
        service_gap = day_status.get("stage1b_gap_percent", "") if day_status.get("stage1b_ran") else day_status.get("stage1a_gap_percent", "")
        day_status.update(
            {
                "total_delivered_count": total_delivered_count,
                "total_count_certified": total_count_certified,
                "phase1_total_delivered_count": total_delivered_count,
                "phase1_total_count_certified": total_count_certified,
                "phase1_delivered_count": total_delivered_count,
                "phase1_delivered_count_certified": total_count_certified,
                "phase1_status": service_status,
                "phase1_objective": service_objective,
                "phase1_best_bound": service_best_bound,
                "phase1_gap_percent": service_gap,
                "phase1_solve_time_sec": day_status["stage1a_solve_time_sec"] + day_status["stage1b_solve_time_sec"],
            }
        )

        if self.optimization_mode != "full_three_stage":
            day_status["stage2_status"] = "SKIPPED"
            if not day_status.get("stage2_skipped_reason"):
                day_status["stage2_skipped_reason"] = f"optimization_mode_{self.optimization_mode}"
        elif stage1b_feasible and stage1b_solver is not None and stage1b_data is not None and self._remaining_daily_time(daily_deadline) > 0.0:
            stage2_build_start = time.perf_counter()
            stage2_data = self._build_day_model(
                instance,
                day,
                candidates,
                use_decision_strategy=False,
                precomputed=precomputed,
            )
            stage2_model_build_time = time.perf_counter() - stage2_build_start
            day_status["stage2_model_build_time_sec"] = stage2_model_build_time
            day_status["phase2_model_build_time_sec"] = stage2_model_build_time
            self._add_phase2_count_constraints(stage2_data, mandatory_delivered_count, total_delivered_count)
            self._set_route_cost_objective(stage2_data, instance, day, candidates)
            hint_stats = self._add_phase1_solution_hint(stage2_data, stage1b_data, stage1b_solver)
            day_status.update(hint_stats)
            remaining_time = min(
                self._remaining_daily_time(daily_deadline),
                self.stage2_max_time_fraction * self.time_limit_per_day_sec,
            )
            if remaining_time > 0.0:
                stage2_solver = self._new_solver(remaining_time)
                stage2_solve_start = time.perf_counter()
                day_status["stage2_ran"] = True
                stage2_status_code = stage2_solver.Solve(stage2_data.model)
                day_status["stage2_solve_time_sec"] = time.perf_counter() - stage2_solve_start
                day_status["phase2_solve_time_sec"] = day_status["stage2_solve_time_sec"]
                stage2_status = stage2_solver.StatusName(stage2_status_code)
                day_status["stage2_status"] = stage2_status
                day_status.update(self._solver_fields("stage2", stage2_status_code, stage2_solver))
                day_status.update(self._solver_fields("phase2", stage2_status_code, stage2_solver))
                if stage2_status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    day_status["stage2_selected_ids"] = [
                        customer_id for customer_id, var in stage2_data.y.items() if stage2_solver.BooleanValue(var)
                    ]
                    day_status["stage2_delivered_count"] = len(day_status["stage2_selected_ids"])
                    best_route = self._extract_route_from_solution(instance, day, candidates, stage2_data, stage2_solver)
                    best_solver_return = stage2_solver.Value(stage2_data.return_time)
                    best_objective = stage2_solver.ObjectiveValue()
                    best_bound = stage2_solver.BestObjectiveBound()
                    best_gap = _gap_percent(best_objective, best_bound)
                    best_gap_scope = "stage2_route_objective"
                    if total_count_certified and stage2_status_code == cp_model.OPTIMAL:
                        daily_optimal_for = "mandatory_total_and_route_cost"
                    elif total_count_certified:
                        daily_optimal_for = "mandatory_and_total_count"
                    else:
                        daily_optimal_for = "not_certified"
            else:
                day_status["stage2_status"] = "NO_TIME"
                day_status["stage2_skipped_reason"] = "no_time_after_stage2_model_build"
        elif self.optimization_mode == "full_three_stage":
            day_status["stage2_status"] = "NO_TIME"
            if not stage1b_feasible:
                day_status["stage2_skipped_reason"] = "stage1b_not_run_or_no_feasible_solution"
            else:
                day_status["stage2_skipped_reason"] = "no_time_after_stage1b"

        final_status = "OPTIMAL" if total_count_certified and day_status.get("stage2_status") == "OPTIMAL" else "FEASIBLE"
        return self._finalize_adaptive_day_status(
            day_status=day_status,
            route=best_route,
            status=final_status,
            objective=best_objective,
            best_bound=best_bound,
            gap_percent=best_gap if not total_count_certified or day_status.get("stage2_status") else "",
            gap_scope=best_gap_scope if (best_gap != "" and (not total_count_certified or day_status.get("stage2_status"))) else "",
            daily_optimal_for=daily_optimal_for,
            solver_return_time=best_solver_return,
            daily_deadline=daily_deadline,
        )

    def _solve_day_weighted_two_phase(
        self,
        instance: Instance,
        day: int,
        candidates: list[str],
        precomputed: DayPrecomputedData,
        precompute_time: float,
    ) -> tuple[DailyRoute, dict[str, object]]:
        """Run the backward-compatible weighted two-phase CP path."""
        phase1_limit, phase2_limit = self._phase_time_limits()
        phase1_build_start = time.perf_counter()
        phase1_data = self._build_day_model(
            instance,
            day,
            candidates,
            use_decision_strategy=self.use_decision_strategy,
            precomputed=precomputed,
        )
        phase1_model_build_time = time.perf_counter() - phase1_build_start
        mandatory_multiplier = len(candidates) + 1
        mandatory_delivered_expr = sum(phase1_data.y[customer_id] for customer_id in precomputed.mandatory_customers)
        total_delivered_expr = sum(phase1_data.y.values())
        phase1_data.model.Maximize(mandatory_multiplier * mandatory_delivered_expr + total_delivered_expr)
        phase1_solver = self._new_solver(phase1_limit)
        phase1_solve_start = time.perf_counter()
        phase1_status_code = phase1_solver.Solve(phase1_data.model)
        phase1_solve_time = time.perf_counter() - phase1_solve_start
        phase1_status = phase1_solver.StatusName(phase1_status_code)
        day_status = self._base_day_status("two_phase", day, phase1_data)
        day_status.update(
            {
                "daily_precompute_time_sec": precompute_time,
                "phase1_model_build_time_sec": phase1_model_build_time,
                "phase1_solve_time_sec": phase1_solve_time,
                "phase2_model_build_time_sec": 0.0,
                "phase2_solve_time_sec": 0.0,
                "daily_total_runtime_sec": "",
                "mandatory_multiplier": mandatory_multiplier,
            }
        )
        day_status.update(self._solver_fields("phase1", phase1_status_code, phase1_solver))

        if phase1_status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            day_status.update(
                {
                    "status": phase1_status,
                    "objective": day_status["phase1_objective"],
                    "best_bound": day_status["phase1_best_bound"],
                    "gap_percent": day_status["phase1_gap_percent"],
                    "gap_scope": "phase1_priority_objective",
                    "phase1_mandatory_delivered_count": 0,
                    "phase1_total_delivered_count": 0,
                    "phase1_priority_objective": day_status["phase1_objective"],
                    "phase1_mandatory_count_certified": False,
                    "phase1_total_count_certified": False,
                    "phase1_delivered_count": 0,
                    "phase1_delivered_count_certified": False,
                    "mandatory_delivered_count": 0,
                    "mandatory_unserved_count": len(precomputed.mandatory_customers),
                    "all_mandatory_served": len(precomputed.mandatory_customers) == 0,
                    "mandatory_count_certified": False,
                    "daily_optimal_for": "no_feasible_daily_solution",
                    "solver_return_time": "",
                    "phase2_hint_enabled": False,
                    "phase2_hint_y_count": 0,
                    "phase2_hint_x_count": 0,
                    "phase2_hint_g_count": 0,
                }
            )
            return DailyRoute(day=day), day_status

        mandatory_delivered_count = int(
            sum(phase1_solver.BooleanValue(phase1_data.y[customer_id]) for customer_id in precomputed.mandatory_customers)
        )
        delivered_count = int(sum(phase1_solver.BooleanValue(var) for var in phase1_data.y.values()))
        mandatory_unserved_count = len(precomputed.mandatory_customers) - mandatory_delivered_count
        phase1_route = self._extract_route_from_solution(instance, day, candidates, phase1_data, phase1_solver)
        day_status["phase1_mandatory_delivered_count"] = mandatory_delivered_count
        day_status["phase1_total_delivered_count"] = delivered_count
        day_status["phase1_priority_objective"] = day_status["phase1_objective"]
        day_status["phase1_mandatory_count_certified"] = phase1_status_code == cp_model.OPTIMAL
        day_status["phase1_total_count_certified"] = phase1_status_code == cp_model.OPTIMAL
        day_status["phase1_delivered_count"] = delivered_count
        day_status["phase1_delivered_count_certified"] = phase1_status_code == cp_model.OPTIMAL
        day_status["mandatory_delivered_count"] = mandatory_delivered_count
        day_status["mandatory_unserved_count"] = mandatory_unserved_count
        day_status["all_mandatory_served"] = mandatory_unserved_count == 0
        day_status["mandatory_count_certified"] = phase1_status_code == cp_model.OPTIMAL

        phase2_allowed = self.solve_phase2 and phase2_limit > 0.0
        if mandatory_unserved_count > 0 and self.phase2_time_limit_sec is None:
            phase2_allowed = False

        if not phase2_allowed:
            day_status.update(
                {
                    "status": "OPTIMAL" if phase1_status_code == cp_model.OPTIMAL else "FEASIBLE",
                    "objective": day_status["phase1_objective"],
                    "best_bound": day_status["phase1_best_bound"],
                    "gap_percent": "" if phase1_status_code == cp_model.OPTIMAL else day_status["phase1_gap_percent"],
                    "gap_scope": "" if phase1_status_code == cp_model.OPTIMAL else "phase1_priority_objective",
                    "daily_optimal_for": "delivered_count_only"
                    if phase1_status_code == cp_model.OPTIMAL
                    else "not_certified",
                    "solver_return_time": phase1_solver.Value(phase1_data.return_time),
                    "phase2_status": "",
                    "phase2_objective": "",
                    "phase2_best_bound": "",
                    "phase2_gap_percent": "",
                    "phase2_hint_enabled": False,
                    "phase2_hint_y_count": 0,
                    "phase2_hint_x_count": 0,
                    "phase2_hint_g_count": 0,
                }
            )
            return phase1_route, day_status

        phase2_build_start = time.perf_counter()
        phase2_data = self._build_day_model(
            instance,
            day,
            candidates,
            use_decision_strategy=False,
            precomputed=precomputed,
        )
        phase2_model_build_time = time.perf_counter() - phase2_build_start
        day_status["phase2_model_build_time_sec"] = phase2_model_build_time
        self._add_phase2_count_constraints(phase2_data, mandatory_delivered_count, delivered_count)
        self._set_route_cost_objective(phase2_data, instance, day, candidates)
        hint_stats = self._add_phase1_solution_hint(phase2_data, phase1_data, phase1_solver)
        day_status.update(hint_stats)
        phase2_solver = self._new_solver(phase2_limit)
        phase2_solve_start = time.perf_counter()
        phase2_status_code = phase2_solver.Solve(phase2_data.model)
        day_status["phase2_solve_time_sec"] = time.perf_counter() - phase2_solve_start
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
                "gap_percent": day_status["phase1_gap_percent"]
                if phase1_status_code != cp_model.OPTIMAL
                else day_status["phase2_gap_percent"],
                "gap_scope": "phase1_priority_objective"
                if phase1_status_code != cp_model.OPTIMAL
                else "phase2_route_objective",
                "daily_optimal_for": daily_optimal_for,
                "solver_return_time": solver_return,
            }
        )
        return route, day_status

    def _add_phase2_count_constraints(
        self,
        data: DayModelData,
        mandatory_delivered_count: int,
        total_delivered_count: int,
    ) -> None:
        """Fix phase-2 mandatory and total delivered counts to phase-1 values."""
        if data.precomputed.mandatory_customers:
            data.model.Add(
                sum(data.y[customer_id] for customer_id in data.precomputed.mandatory_customers)
                == mandatory_delivered_count
            )
        data.model.Add(sum(data.y.values()) == total_delivered_count)

    def _add_mandatory_count_constraint(self, data: DayModelData, mandatory_delivered_count: int) -> None:
        """Fix only the mandatory delivered count for adaptive Stage 1B."""
        if data.precomputed.mandatory_customers:
            data.model.Add(
                sum(data.y[customer_id] for customer_id in data.precomputed.mandatory_customers)
                == mandatory_delivered_count
            )

    def _solve_day_single_phase(
        self,
        instance: Instance,
        day: int,
        candidates: list[str],
        precomputed: DayPrecomputedData,
        precompute_time: float,
    ) -> tuple[DailyRoute, dict[str, object]]:
        """Run the original pure single-phase CP objective."""
        build_start = time.perf_counter()
        data = self._build_day_model(instance, day, candidates, precomputed=precomputed)
        build_time = time.perf_counter() - build_start
        drop_penalty = self.drop_penalty_by_day.get(day, DROP_PENALTY_BY_DAY[SUNDAY])
        objective_terms: list[cp_model.LinearExpr] = []
        objective_terms.extend(drop_penalty * (1 - data.y[customer_id]) for customer_id in candidates)
        for i in data.nodes:
            for j in data.nodes:
                if i == j:
                    continue
                objective_terms.append(precomputed.distance_cost[i, j] * data.x[i, j])
        for customer_id in candidates:
            earliest_day = min(instance.available_days(customer_id))
            objective_terms.append(100 * (day - earliest_day) * data.y[customer_id])
        data.model.Minimize(sum(objective_terms))

        solver = self._new_solver(self.time_limit_per_day_sec)
        solve_start = time.perf_counter()
        status_code = solver.Solve(data.model)
        solve_time = time.perf_counter() - solve_start
        status = solver.StatusName(status_code)
        day_status = self._base_day_status("single_phase", day, data)
        day_status.update(
            {
                "daily_precompute_time_sec": precompute_time,
                "phase1_model_build_time_sec": build_time,
                "phase1_solve_time_sec": solve_time,
                "phase2_model_build_time_sec": 0.0,
                "phase2_solve_time_sec": 0.0,
                "daily_total_runtime_sec": "",
            }
        )
        if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            route = self._extract_route_from_solution(instance, day, candidates, data, solver)
            mandatory_delivered_count = sum(1 for customer_id in precomputed.mandatory_customers if customer_id in route.delivered_customer_ids())
            day_status.update(
                {
                    "status": status,
                    "objective": solver.ObjectiveValue(),
                    "best_bound": solver.BestObjectiveBound(),
                    "gap_percent": _gap_percent(solver.ObjectiveValue(), solver.BestObjectiveBound()),
                    "gap_scope": "single_phase_objective",
                    "daily_optimal_for": "single_phase_objective" if status_code == cp_model.OPTIMAL else "not_certified",
                    "solver_return_time": solver.Value(data.return_time),
                    "mandatory_delivered_count": mandatory_delivered_count,
                    "mandatory_unserved_count": len(precomputed.mandatory_customers) - mandatory_delivered_count,
                    "all_mandatory_served": mandatory_delivered_count == len(precomputed.mandatory_customers),
                    "mandatory_count_certified": status_code == cp_model.OPTIMAL,
                }
            )
            return route, day_status

        day_status.update(
            {
                "status": status,
                "objective": "",
                "best_bound": "",
                "gap_percent": "",
                "gap_scope": "",
                "daily_optimal_for": "no_feasible_daily_solution",
                "solver_return_time": "",
                "mandatory_delivered_count": 0,
                "mandatory_unserved_count": len(precomputed.mandatory_customers),
                "all_mandatory_served": len(precomputed.mandatory_customers) == 0,
                "mandatory_count_certified": False,
            }
        )
        return DailyRoute(day=day), day_status

    def _phase_time_limits(self) -> tuple[float, float]:
        """Return phase time limits from explicit values or configured fractions."""
        phase1 = self.phase1_time_limit_sec
        phase2 = self.phase2_time_limit_sec
        if phase1 is None:
            phase1 = max(0.001, self.phase1_time_fraction * self.time_limit_per_day_sec)
        if phase2 is None:
            phase2 = self.phase2_time_fraction * self.time_limit_per_day_sec
        return float(phase1), float(phase2)

    @staticmethod
    def _remaining_daily_time(daily_deadline: float) -> float:
        """Return nonnegative time remaining before the daily wall-clock deadline."""
        return max(0.0, daily_deadline - time.perf_counter())

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
        """Set official represented Stage-2 terms; exact waiting is not modeled."""
        objective_terms: list[cp_model.LinearExpr] = []
        for i in data.nodes:
            for j in data.nodes:
                if i == j:
                    continue
                objective_terms.append(data.precomputed.distance_cost[i, j] * data.x[i, j])
        for customer_id in candidates:
            earliest_day = min(instance.available_days(customer_id))
            objective_terms.append(100 * (day - earliest_day) * data.y[customer_id])
        data.model.Minimize(sum(objective_terms))

    def _add_phase1_solution_hint(
        self,
        phase2_data: DayModelData,
        phase1_data: DayModelData,
        phase1_solver: cp_model.CpSolver,
    ) -> dict[str, object]:
        """Seed phase 2 only with values from the phase-1 CP incumbent."""
        y_count, x_count, g_count = self._add_solution_hint(phase2_data, phase1_data, phase1_solver)
        return {
            "phase2_hint_enabled": True,
            "phase2_hint_y_count": y_count,
            "phase2_hint_x_count": x_count,
            "phase2_hint_g_count": g_count,
        }

    def _add_stage_hint(
        self,
        prefix: str,
        target_data: DayModelData,
        source_data: DayModelData,
        source_solver: cp_model.CpSolver,
    ) -> dict[str, object]:
        """Seed one adaptive stage from the previous CP incumbent."""
        y_count, x_count, g_count = self._add_solution_hint(target_data, source_data, source_solver)
        return {
            f"{prefix}_hint_enabled": True,
            f"{prefix}_hint_y_count": y_count,
            f"{prefix}_hint_x_count": x_count,
            f"{prefix}_hint_g_count": g_count,
        }

    def _add_solution_hint(
        self,
        target_data: DayModelData,
        source_data: DayModelData,
        source_solver: cp_model.CpSolver,
    ) -> tuple[int, int, int]:
        """Transfer variable values from one CP incumbent into another model."""
        for customer_id, var in target_data.y.items():
            target_data.model.AddHint(var, int(source_solver.BooleanValue(source_data.y[customer_id])))
        for arc, var in target_data.x.items():
            target_data.model.AddHint(var, int(source_solver.BooleanValue(source_data.x[arc])))
        for key, var in target_data.g.items():
            target_data.model.AddHint(var, int(source_solver.BooleanValue(source_data.g[key])))
        target_data.model.AddHint(target_data.z, int(source_solver.BooleanValue(source_data.z)))
        for customer_id, var in target_data.t.items():
            target_data.model.AddHint(var, source_solver.Value(source_data.t[customer_id]))
        target_data.model.AddHint(target_data.departure, source_solver.Value(source_data.departure))
        target_data.model.AddHint(target_data.return_time, source_solver.Value(source_data.return_time))
        for customer_id, var in target_data.next_travel.items():
            target_data.model.AddHint(var, source_solver.Value(source_data.next_travel[customer_id]))
        for customer_id, var in target_data.interval_end.items():
            target_data.model.AddHint(var, source_solver.Value(source_data.interval_end[customer_id]))
        target_data.model.AddHint(target_data.depot_first_travel, source_solver.Value(source_data.depot_first_travel))
        target_data.model.AddHint(target_data.depot_interval_end, source_solver.Value(source_data.depot_interval_end))
        return len(target_data.y), len(target_data.x), len(target_data.g)

    def _finalize_adaptive_day_status(
        self,
        day_status: dict[str, object],
        route: DailyRoute,
        status: str,
        objective: float | str,
        best_bound: float | str,
        gap_percent: float | str,
        gap_scope: str,
        daily_optimal_for: str,
        solver_return_time: int | str,
        daily_deadline: float,
    ) -> tuple[DailyRoute, dict[str, object]]:
        """Fill top-level fields for the adaptive staged solve."""
        day_status.update(
            {
                "status": status,
                "objective": objective,
                "best_bound": best_bound,
                "gap_percent": gap_percent,
                "gap_scope": gap_scope,
                "daily_optimal_for": daily_optimal_for,
                "solver_return_time": solver_return_time,
                "phase2_status": day_status.get("stage2_status", ""),
                "unused_daily_budget_sec": self._remaining_daily_time(daily_deadline),
            }
        )
        return route, day_status

    def _base_day_status(self, objective_mode: str, day: int, data: DayModelData) -> dict[str, object]:
        total_nonself_arcs = data.arc_stats["total_nonself_arcs"]
        fixed_impossible_arcs = data.arc_stats["fixed_impossible_arcs"]
        mandatory_count = len(data.precomputed.mandatory_customers)
        return {
            "objective_mode": objective_mode,
            "optimization_mode": self.optimization_mode,
            "official_default_profile": "adaptive_full_three_stage",
            "adaptive_daily_deadline": self.adaptive_daily_deadline,
            "status": "",
            "objective": "",
            "best_bound": "",
            "gap_percent": "",
            "gap_scope": "",
            "runtime_sec": "",
            "daily_precompute_time_sec": "",
            "phase1_model_build_time_sec": "",
            "phase1_solve_time_sec": "",
            "phase2_model_build_time_sec": "",
            "phase2_solve_time_sec": "",
            "stage1a_status": "",
            "stage1b_status": "",
            "stage2_status": "",
            "stage1a_ran": False,
            "stage1b_ran": False,
            "stage2_ran": False,
            "stage1b_skipped_reason": "",
            "stage2_skipped_reason": "",
            "stage1a_solve_time_sec": "",
            "stage1b_solve_time_sec": "",
            "stage2_solve_time_sec": "",
            "unused_daily_budget_sec": "",
            "daily_total_runtime_sec": "",
            "raw_candidate_count": "",
            "selected_candidate_count": "",
            "filtered_candidate_count": "",
            "candidate_limit_active": "",
            "candidate_limit_value": "",
            "raw_candidate_ids": [],
            "selected_candidate_ids": [],
            "filtered_candidate_ids": [],
            "mandatory_candidate_ids": list(data.precomputed.mandatory_customers),
            "stage1a_selected_ids": [],
            "stage1b_selected_ids": [],
            "stage2_selected_ids": [],
            "extracted_route_ids": [],
            "remaining_customer_ids_after_day": [],
            "customer_day_diagnostics": [],
            "mandatory_last_day_count": "",
            "mandatory_candidate_count": mandatory_count,
            "mandatory_delivered_count": "",
            "mandatory_unserved_count": "",
            "all_mandatory_served": "",
            "mandatory_count_certified": "",
            "total_delivered_count": "",
            "total_count_certified": "",
            "stage1a_delivered_count": "",
            "stage1b_delivered_count": "",
            "stage2_delivered_count": "",
            "extracted_route_customer_count": "",
            "fixed_impossible_arcs": fixed_impossible_arcs,
            "total_nonself_arcs": total_nonself_arcs,
            "fixed_arc_ratio": fixed_impossible_arcs / total_nonself_arcs if total_nonself_arcs else 0.0,
            **data.tightening_stats,
            "service_no_overlap_enabled": self.use_service_no_overlap,
            "use_service_no_overlap": self.use_service_no_overlap,
            "route_interval_no_overlap_enabled": self.use_route_interval_no_overlap,
            "use_route_interval_no_overlap": self.use_route_interval_no_overlap,
            "use_window_pair_cuts": self.use_window_pair_cuts,
            "use_precedence_cuts": self.use_precedence_cuts,
            "use_pair_conflict_cuts": self.use_pair_conflict_cuts,
            "use_depot_window_cuts": self.use_depot_window_cuts,
            "use_dominated_window_cuts": self.use_dominated_window_cuts,
            "phase1_only": not self.solve_phase2,
            "solve_phase2": self.solve_phase2,
            "stage2_max_time_fraction": self.stage2_max_time_fraction,
            "decision_strategy_enabled": self.use_decision_strategy,
            "mandatory_first_decision_strategy": bool(
                self.use_decision_strategy and mandatory_count and data.decision_strategy_customer_order[:mandatory_count]
                == data.precomputed.mandatory_customers
            ),
            "decision_strategy_customer_order": ",".join(data.decision_strategy_customer_order),
            "phase2_hint_enabled": False,
            "phase2_hint_y_count": 0,
            "phase2_hint_x_count": 0,
            "phase2_hint_g_count": 0,
            "extraction_consistency_error": False,
            "extraction_error_message": "",
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
            for window_idx, window in enumerate(data.precomputed.windows_by_customer[customer_id]):
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
