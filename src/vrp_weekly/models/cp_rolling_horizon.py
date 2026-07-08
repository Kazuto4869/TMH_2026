"""Rolling-horizon daily CP-SAT routing model."""

from __future__ import annotations

import logging
import math

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


def _gap_percent(objective: float, best_bound: float) -> float:
    """Return minimization optimality gap in percent."""
    return 100.0 * abs(objective - best_bound) / max(1.0, abs(objective))


def _travel_time_minutes(instance: Instance, i: str, j: str) -> int:
    """Return ceil travel time in minutes at the configured max speed."""
    distance_km = euclidean_distance_km(instance.locations[i], instance.locations[j])
    return int(math.ceil(MINUTES_PER_HOUR * distance_km / MAX_SPEED_KMPH))


def _distance_scaled(instance: Instance, i: str, j: str) -> int:
    """Return distance scaled to an integer objective coefficient."""
    distance_km = euclidean_distance_km(instance.locations[i], instance.locations[j])
    return int(round(distance_km * DISTANCE_SCALE))


def _get_service_time(instance: Instance, i: str) -> int:
    """Return service time for a customer, with zero service at depots."""
    location = instance.locations[i]
    if location.is_depot or i == instance.depot_id:
        return 0
    return location.service_time if location.service_time > 0 else DEFAULT_SERVICE_TIME_MIN


def _windows_for(instance: Instance, customer_id: str, day: int) -> list[TimeWindow]:
    """Return sorted windows for one customer/day."""
    return instance.windows_for_customer_day(customer_id, day)


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
    return_to_depot_time: int | None = None,
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
    route_return = actual_return if return_to_depot_time is None else return_to_depot_time
    if route_return > DAY_END_MIN:
        violations.append(f"Route on day {day} returns after 24:00")

    route_duration = route_return - depot_departure_time if route_sequence else 0
    return DailyRoute(
        day=day,
        stops=stops,
        depot_departure_time=depot_departure_time if route_sequence else 0,
        return_to_depot_time=route_return if route_sequence else 0,
        route_distance_km=route_distance_km if route_sequence else 0.0,
        route_travel_time_min=route_travel_time_min if route_sequence else 0,
        route_waiting_time_min=route_waiting_time_min if route_sequence else 0,
        route_service_time_min=route_service_time_min if route_sequence else 0,
        route_duration_min=route_duration,
        hard_feasible=not violations and all(stop.hard_feasible for stop in stops),
        violations=violations,
    )


class RollingHorizonCPSATSolver:
    """Solve one day at a time with CP-SAT and carry undelivered orders forward."""

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
    ) -> None:
        self.time_limit_per_day_sec = time_limit_per_day_sec
        self.max_candidates_per_day = max_candidates_per_day
        self.drop_penalty_by_day = dict(DROP_PENALTY_BY_DAY if drop_penalty_by_day is None else drop_penalty_by_day)
        self.distance_weight = distance_weight
        self.route_duration_weight = route_duration_weight
        self.urgency_weight = urgency_weight
        self.num_workers = num_workers
        self.log_search_progress = log_search_progress

    def solve(self, instance: Instance) -> WeeklySchedule:
        """Build a weekly schedule by solving a daily CP-SAT model."""
        undelivered = set(instance.customer_ids())
        routes: dict[int, DailyRoute] = {}
        day_statuses: dict[int, dict[str, object]] = {}

        for day in range(MONDAY, SUNDAY + 1):
            LOGGER.info("cp_rolling day=%s start undelivered=%s", day, len(undelivered))
            candidates = sorted(customer for customer in undelivered if _windows_for(instance, customer, day))
            candidates = self._limit_candidates(instance, day, candidates)
            if not candidates:
                routes[day] = DailyRoute(day=day)
                day_statuses[day] = {"status": "NO_CANDIDATES"}
                LOGGER.info("cp_rolling day=%s no candidates", day)
                continue

            LOGGER.info("cp_rolling day=%s solve candidates=%s", day, len(candidates))
            route, day_status = self._solve_day(instance, day, candidates)
            routes[day] = route
            day_statuses[day] = day_status
            undelivered -= route.delivered_customer_ids()
            LOGGER.info(
                "cp_rolling day=%s done status=%s stops=%s remaining=%s",
                day,
                day_status.get("status", ""),
                len(route.stops),
                len(undelivered),
            )

        solved_statuses = [status for status in day_statuses.values() if status["status"] not in {"NO_CANDIDATES"}]
        if all(status["status"] == "OPTIMAL" for status in solved_statuses):
            overall_status = "OPTIMAL"
        elif any(status["status"] in {"OPTIMAL", "FEASIBLE"} for status in solved_statuses):
            overall_status = "FEASIBLE"
        else:
            overall_status = "NO_FEASIBLE_DAILY_SOLUTIONS" if solved_statuses else "NO_CANDIDATES"
        gaps = [float(status["gap_percent"]) for status in solved_statuses if "gap_percent" in status]
        return WeeklySchedule(
            routes=routes,
            solver_status={
                "status": overall_status,
                "gap_percent": max(gaps) if gaps else "",
                "day_statuses": day_statuses,
            },
        )

    def _limit_candidates(self, instance: Instance, day: int, candidates: list[str]) -> list[str]:
        """Apply deterministic candidate screening for large real-data days."""
        if self.max_candidates_per_day is None or len(candidates) <= self.max_candidates_per_day:
            return candidates

        depot_id = instance.depot_id

        def priority(customer_id: str) -> tuple[int, int, int, str]:
            remaining_days = len([d for d in instance.available_days(customer_id) if d >= day])
            earliest_end = min(window.end_minute for window in _windows_for(instance, customer_id, day))
            depot_distance = _distance_scaled(instance, depot_id, customer_id)
            return (remaining_days, earliest_end, depot_distance, customer_id)

        return sorted(candidates, key=priority)[: self.max_candidates_per_day]

    def _solve_day(self, instance: Instance, day: int, candidates: list[str]) -> tuple[DailyRoute, dict[str, object]]:
        """Solve one daily CP-SAT subproblem."""
        model = cp_model.CpModel()
        depot_id = instance.depot_id
        nodes = [depot_id] + candidates
        node_index = {node_id: index for index, node_id in enumerate(nodes)}

        y = {customer_id: model.NewBoolVar(f"y[{customer_id}]") for customer_id in candidates}
        t = {customer_id: model.NewIntVar(0, 1440, f"T[{customer_id}]") for customer_id in candidates}
        g: dict[tuple[str, int], cp_model.IntVar] = {}
        x = {(i, j): model.NewBoolVar(f"x[{i},{j}]") for i in nodes for j in nodes}
        z = model.NewBoolVar("z")
        departure = model.NewIntVar(0, 1440, "L")
        return_time = model.NewIntVar(0, 1440, "R")
        window_lookup: dict[tuple[str, int], TimeWindow] = {}

        for customer_id in candidates:
            window_vars: list[cp_model.IntVar] = []
            service_time = _get_service_time(instance, customer_id)
            for window_idx, window in enumerate(_windows_for(instance, customer_id, day)):
                var = model.NewBoolVar(f"g[{customer_id},{window_idx}]")
                g[customer_id, window_idx] = var
                window_lookup[customer_id, window_idx] = window
                window_vars.append(var)
                model.Add(t[customer_id] >= window.start_minute).OnlyEnforceIf(var)
                model.Add(t[customer_id] + service_time <= window.end_minute).OnlyEnforceIf(var)
            model.Add(sum(window_vars) == y[customer_id])
            model.Add(x[customer_id, customer_id] + y[customer_id] == 1)
            model.Add(y[customer_id] <= z)

        model.Add(z <= sum(y.values()))
        model.Add(x[depot_id, depot_id] + z == 1)
        model.AddCircuit([(node_index[i], node_index[j], x[i, j]) for i in nodes for j in nodes])

        model.Add(departure <= 1440 * z)
        model.Add(return_time <= 1440 * z)
        model.Add(return_time >= departure)
        model.Add(return_time <= 1440)

        for i in candidates:
            for j in candidates:
                if i == j:
                    continue
                travel_time = _travel_time_minutes(instance, i, j)
                service_time = _get_service_time(instance, i)
                model.Add(t[j] >= t[i] + service_time + travel_time).OnlyEnforceIf(x[i, j])

        for j in candidates:
            travel_time = _travel_time_minutes(instance, depot_id, j)
            model.Add(t[j] >= departure + travel_time).OnlyEnforceIf(x[depot_id, j])

        for i in candidates:
            travel_time = _travel_time_minutes(instance, i, depot_id)
            service_time = _get_service_time(instance, i)
            model.Add(return_time >= t[i] + service_time + travel_time).OnlyEnforceIf(x[i, depot_id])

        drop_penalty = self.drop_penalty_by_day.get(day, DROP_PENALTY_BY_DAY[SUNDAY])
        objective_terms: list[cp_model.LinearExpr] = []
        objective_terms.extend(drop_penalty * (1 - y[customer_id]) for customer_id in candidates)
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                objective_terms.append(self.distance_weight * _distance_scaled(instance, i, j) * x[i, j])
        objective_terms.append(self.route_duration_weight * (return_time - departure))
        for customer_id in candidates:
            objective_terms.append(-self.urgency_weight * self._urgency(instance, day, customer_id) * y[customer_id])
        model.Minimize(sum(objective_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_per_day_sec
        solver.parameters.num_search_workers = self.num_workers
        solver.parameters.log_search_progress = self.log_search_progress
        status = solver.Solve(model)
        status_name = solver.StatusName(status)
        day_status: dict[str, object] = {"status": status_name}
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            objective = solver.ObjectiveValue()
            best_bound = solver.BestObjectiveBound()
            day_status.update(
                {
                    "objective": objective,
                    "best_bound": best_bound,
                    "gap_percent": _gap_percent(objective, best_bound),
                }
            )
        LOGGER.info(
            "Rolling CP-SAT day=%s status=%s objective=%s best_bound=%s",
            day,
            status_name,
            solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
            solver.BestObjectiveBound() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        )

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return DailyRoute(day=day), day_status

        selected_arcs = {(i, j) for i in nodes for j in nodes if solver.BooleanValue(x[i, j])}
        sequence = _extract_route_from_arcs(selected_arcs, depot_id, candidates)
        start_times = {customer_id: solver.Value(t[customer_id]) for customer_id in sequence}
        selected_windows: dict[str, TimeWindow] = {}
        for customer_id in sequence:
            for window_idx, window in enumerate(_windows_for(instance, customer_id, day)):
                var = g.get((customer_id, window_idx))
                if var is not None and solver.BooleanValue(var):
                    selected_windows[customer_id] = window
                    break

        route = _build_daily_schedule_from_solution(
            instance=instance,
            day=day,
            route_sequence=sequence,
            service_start_times=start_times,
            selected_windows=selected_windows,
            depot_departure_time=solver.Value(departure),
            return_to_depot_time=solver.Value(return_time),
        )
        return route, day_status

    def _urgency(self, instance: Instance, day: int, customer_id: str) -> int:
        """Return integer urgency score for the daily objective."""
        remaining_available_days = len([d for d in instance.available_days(customer_id) if d >= day])
        if remaining_available_days <= 1:
            return 10
        return round(100 / remaining_available_days)
