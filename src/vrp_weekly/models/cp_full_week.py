"""Full-week CP-SAT routing model."""

from __future__ import annotations

import logging
import math

from ortools.sat.python import cp_model

from vrp_weekly.config import (
    DAY_END_MIN,
    DEFAULT_SERVICE_TIME_MIN,
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


def _earliest_available_day(instance: Instance, customer_id: str) -> int:
    """Return the earliest day where a customer has at least one window."""
    days = instance.available_days(customer_id)
    return min(days) if days else 1


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


class FullWeekCPSATSolver:
    """Full-week CP-SAT mathematical demonstration.

    Use `RollingHorizonCPSATSolver` for real data because this model has
    O(7*n^2) arc variables.
    """

    name = "cp_full_week"

    def __init__(
        self,
        time_limit_sec: int = 60,
        max_customers: int | None = 40,
        incomplete_weight: int = 1_000,
        deferral_weight: int = 100,
        distance_weight: int = 10,
        route_duration_weight: int = 0,
        num_workers: int = 8,
        log_search_progress: bool = False,
    ) -> None:
        self.time_limit_sec = time_limit_sec
        self.max_customers = max_customers
        self.incomplete_weight = incomplete_weight
        self.deferral_weight = deferral_weight
        self.distance_weight = distance_weight
        self.route_duration_weight = route_duration_weight
        self.num_workers = num_workers
        self.log_search_progress = log_search_progress

    def solve(self, instance: Instance) -> WeeklySchedule:
        """Build, solve, and extract a full-week CP-SAT schedule."""
        customers = instance.customer_ids()
        total_customers = len(customers)
        if self.max_customers is None and total_customers > 80:
            raise ValueError(
                "cp_full_week has O(7*n^2) arc variables and is intended only for small instances. "
                "Use cp_rolling for real data."
            )
        if self.max_customers is not None and self.max_customers > 80:
            LOGGER.warning(
                "cp_full_week max_customers=%s can create a large O(7*n^2) model; use cp_rolling for real data.",
                self.max_customers,
            )
        if self.max_customers is not None:
            customers = customers[: self.max_customers]

        if not customers:
            routes = {day: DailyRoute(day=day) for day in range(MONDAY, SUNDAY + 1)}
            schedule = WeeklySchedule(routes=routes)
            metrics = evaluate_weekly_schedule(instance, schedule)
            return WeeklySchedule(routes=routes, solver_status={
                "status": "NO_CUSTOMERS",
                "internal_waiting_objective_enabled": False,
                "instance_customer_count": total_customers,
                "modeled_customer_count": 0,
                "full_instance_model": total_customers == 0,
                **official_objective_status(metrics),
            })

        LOGGER.info(
            "cp_full_week build start customers=%s days=%s max_customers=%s",
            len(customers),
            SUNDAY - MONDAY + 1,
            self.max_customers,
        )
        model = cp_model.CpModel()
        depot_id = instance.depot_id
        days = list(range(MONDAY, SUNDAY + 1))
        nodes = [depot_id] + customers
        node_index = {node_id: index for index, node_id in enumerate(nodes)}

        y: dict[tuple[str, int], cp_model.IntVar] = {}
        g: dict[tuple[str, int, int], cp_model.IntVar] = {}
        t: dict[tuple[str, int], cp_model.IntVar] = {}
        x: dict[tuple[str, str, int], cp_model.IntVar] = {}
        z: dict[int, cp_model.IntVar] = {}
        departure: dict[int, cp_model.IntVar] = {}
        return_time: dict[int, cp_model.IntVar] = {}
        incomplete: dict[str, cp_model.IntVar] = {}
        window_lookup: dict[tuple[str, int, int], TimeWindow] = {}

        for customer_id in customers:
            incomplete[customer_id] = model.NewBoolVar(f"u[{customer_id}]")
            for day in days:
                y[customer_id, day] = model.NewBoolVar(f"y[{customer_id},{day}]")
                t[customer_id, day] = model.NewIntVar(0, 1440, f"T[{customer_id},{day}]")
                windows = _windows_for(instance, customer_id, day)
                if not windows:
                    model.Add(y[customer_id, day] == 0)
                window_vars: list[cp_model.IntVar] = []
                for window_idx, window in enumerate(windows):
                    var = model.NewBoolVar(f"g[{customer_id},{day},{window_idx}]")
                    g[customer_id, day, window_idx] = var
                    window_lookup[customer_id, day, window_idx] = window
                    window_vars.append(var)
                    service_time = _get_service_time(instance, customer_id)
                    model.Add(t[customer_id, day] >= window.start_minute).OnlyEnforceIf(var)
                    model.Add(t[customer_id, day] + service_time <= window.end_minute).OnlyEnforceIf(var)
                model.Add(sum(window_vars) == y[customer_id, day])

            model.Add(sum(y[customer_id, day] for day in days) + incomplete[customer_id] == 1)

        for day in days:
            z[day] = model.NewBoolVar(f"z[{day}]")
            departure[day] = model.NewIntVar(0, 1440, f"L[{day}]")
            return_time[day] = model.NewIntVar(0, 1440, f"R[{day}]")
            for i in nodes:
                for j in nodes:
                    x[i, j, day] = model.NewBoolVar(f"x[{i},{j},{day}]")

            candidate_vars = [y[customer_id, day] for customer_id in customers]
            if candidate_vars:
                for customer_id in customers:
                    model.Add(y[customer_id, day] <= z[day])
                model.Add(z[day] <= sum(candidate_vars))
            else:
                model.Add(z[day] == 0)

            for customer_id in customers:
                model.Add(x[customer_id, customer_id, day] + y[customer_id, day] == 1)
            model.Add(x[depot_id, depot_id, day] + z[day] == 1)

            arcs = [
                (node_index[i], node_index[j], x[i, j, day])
                for i in nodes
                for j in nodes
            ]
            model.AddCircuit(arcs)

            model.Add(departure[day] <= 1440 * z[day])
            model.Add(return_time[day] <= 1440 * z[day])
            model.Add(return_time[day] >= departure[day])

            for i in customers:
                for j in customers:
                    if i == j:
                        continue
                    travel_time = _travel_time_minutes(instance, i, j)
                    service_time = _get_service_time(instance, i)
                    model.Add(t[j, day] >= t[i, day] + service_time + travel_time).OnlyEnforceIf(x[i, j, day])

            for j in customers:
                travel_time = _travel_time_minutes(instance, depot_id, j)
                model.Add(t[j, day] >= departure[day] + travel_time).OnlyEnforceIf(x[depot_id, j, day])

            for i in customers:
                travel_time = _travel_time_minutes(instance, i, depot_id)
                service_time = _get_service_time(instance, i)
                model.Add(return_time[day] >= t[i, day] + service_time + travel_time).OnlyEnforceIf(x[i, depot_id, day])

        objective_terms: list[cp_model.LinearExpr] = []
        objective_terms.extend(self.incomplete_weight * incomplete[customer_id] for customer_id in customers)
        for customer_id in customers:
            earliest_day = _earliest_available_day(instance, customer_id)
            for day in days:
                objective_terms.append(self.deferral_weight * (day - earliest_day) * y[customer_id, day])
        for day in days:
            for i in nodes:
                for j in nodes:
                    if i == j:
                        continue
                    objective_terms.append(_distance_objective_cost(instance, i, j, self.distance_weight) * x[i, j, day])

        model.Minimize(sum(objective_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_sec
        solver.parameters.num_search_workers = self.num_workers
        solver.parameters.log_search_progress = self.log_search_progress
        LOGGER.info(
            "cp_full_week solve start time_limit_sec=%s workers=%s",
            self.time_limit_sec,
            self.num_workers,
        )
        status = solver.Solve(model)
        status_name = solver.StatusName(status)
        solver_status: dict[str, object] = {
            "status": status_name,
            "internal_waiting_objective_enabled": False,
            "internal_objective_decomposition": "1000*incomplete + 100*deferral + 10*distance; exact waiting unavailable",
            "instance_customer_count": total_customers,
            "modeled_customer_count": len(customers),
            "full_instance_model": len(customers) == total_customers,
            "max_customers": self.max_customers,
        }
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            objective = solver.ObjectiveValue()
            best_bound = solver.BestObjectiveBound()
            solver_status.update(
                {
                    "objective": objective,
                    "best_bound": best_bound,
                    "gap_percent": _gap_percent(objective, best_bound),
                }
            )
        LOGGER.info(
            "Full-week CP-SAT status=%s objective=%s best_bound=%s",
            status_name,
            solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
            solver.BestObjectiveBound() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        )

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            routes = {day: DailyRoute(day=day) for day in days}
            schedule = WeeklySchedule(routes=routes, solver_status=solver_status)
            solver_status.update(official_objective_status(evaluate_weekly_schedule(instance, schedule)))
            return WeeklySchedule(routes=routes, solver_status=solver_status)

        routes: dict[int, DailyRoute] = {}
        for day in days:
            selected_arcs = {
                (i, j)
                for i in nodes
                for j in nodes
                if solver.BooleanValue(x[i, j, day])
            }
            sequence = _extract_route_from_arcs(selected_arcs, depot_id, customers)
            start_times = {customer_id: solver.Value(t[customer_id, day]) for customer_id in sequence}
            selected_windows: dict[str, TimeWindow] = {}
            for customer_id in sequence:
                windows = _windows_for(instance, customer_id, day)
                for window_idx, window in enumerate(windows):
                    var = g.get((customer_id, day, window_idx))
                    if var is not None and solver.BooleanValue(var):
                        selected_windows[customer_id] = window
                        break
            routes[day] = _build_daily_schedule_from_solution(
                instance=instance,
                day=day,
                route_sequence=sequence,
                service_start_times=start_times,
                selected_windows=selected_windows,
                depot_departure_time=solver.Value(departure[day]),
            )
            LOGGER.info("cp_full_week day=%s extracted stops=%s", day, len(routes[day].stops))

        schedule = WeeklySchedule(routes=routes, solver_status=solver_status)
        metrics = evaluate_weekly_schedule(instance, schedule)
        solver_status.update(official_objective_status(metrics))
        return WeeklySchedule(routes=routes, solver_status=solver_status)


CpDailySolver = FullWeekCPSATSolver
