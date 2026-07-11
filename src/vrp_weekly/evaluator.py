"""Schedule simulation, validation, and pretty-printing utilities."""

from __future__ import annotations

from collections import Counter

from vrp_weekly.config import (
    ALLOW_WAITING,
    DAY_END_MIN,
    DEFAULT_SERVICE_TIME_MIN,
    FLEXIBLE_DEPOT_DEPARTURE,
    MONDAY,
    OBJECTIVE_VERSION,
    REQUIRE_RETURN_BEFORE_DAY_END,
    REQUIRE_SERVICE_END_WITHIN_WINDOW,
    SUNDAY,
    WEIGHT_DEFERRAL,
    WEIGHT_DISTANCE_KM,
    WEIGHT_INCOMPLETE,
    WEIGHT_WAITING_MIN,
)
from vrp_weekly.distance import euclidean_distance_km, travel_time_between_minutes
from vrp_weekly.core import DailyRoute, EvaluationMetrics, Instance, Stop, TimeWindow, WeeklySchedule
from vrp_weekly.time_utils import format_hhmm


def evaluate_daily_route(instance: Instance, day: int, customer_sequence: list[str]) -> DailyRoute:
    """Simulate one daily route and choose the earliest feasible window per stop."""
    if day < MONDAY or day > SUNDAY:
        raise ValueError(f"day must be in 1..7, got {day}")

    previous_location = instance.depot()
    depot_departure_time = _choose_depot_departure_time(instance, day, customer_sequence)
    current_time = depot_departure_time
    stops: list[Stop] = []
    violations: list[str] = []
    route_distance_km = 0.0
    route_travel_time_min = 0
    route_waiting_time_min = 0
    route_service_time_min = 0

    for customer_id in customer_sequence:
        location = instance.locations.get(customer_id)
        if location is None:
            violation = f"Unknown customer id: {customer_id}"
            violations.append(violation)
            stops.append(
                Stop(
                    customer_id=customer_id,
                    arrival_time=current_time,
                    service_start_time=current_time,
                    service_end_time=current_time,
                    hard_feasible=False,
                    violation=violation,
                )
            )
            continue

        distance_km = euclidean_distance_km(previous_location, location)
        travel_time_min = travel_time_between_minutes(previous_location, location)
        arrival_time = current_time + travel_time_min
        service_time = location.service_time if location.service_time > 0 else DEFAULT_SERVICE_TIME_MIN
        selected_window = choose_earliest_feasible_window(instance, customer_id, day, arrival_time, service_time)

        if selected_window is None:
            service_start_time = arrival_time
            waiting_min = 0
            hard_feasible = False
            violation = f"No feasible time window for {customer_id} on day {day} at arrival {format_hhmm_safe(arrival_time)}"
            violations.append(violation)
        else:
            service_start_time = max(arrival_time, selected_window.start_minute)
            waiting_min = service_start_time - arrival_time
            hard_feasible = True
            violation = None

        service_end_time = service_start_time + service_time
        stops.append(
            Stop(
                customer_id=customer_id,
                arrival_time=arrival_time,
                service_start_time=service_start_time,
                service_end_time=service_end_time,
                selected_time_window=selected_window,
                travel_from_previous_min=travel_time_min,
                waiting_min=waiting_min,
                distance_from_previous_km=distance_km,
                hard_feasible=hard_feasible,
                violation=violation,
            )
        )
        route_distance_km += distance_km
        route_travel_time_min += travel_time_min
        route_waiting_time_min += waiting_min
        route_service_time_min += service_time
        current_time = service_end_time
        previous_location = location

    return_distance_km = euclidean_distance_km(previous_location, instance.depot())
    return_travel_time_min = travel_time_between_minutes(previous_location, instance.depot())
    route_distance_km += return_distance_km
    route_travel_time_min += return_travel_time_min
    return_to_depot_time = current_time + return_travel_time_min
    if REQUIRE_RETURN_BEFORE_DAY_END and return_to_depot_time > DAY_END_MIN:
        violations.append(f"Route on day {day} returns after 24:00 at {return_to_depot_time} minutes")

    route_duration_min = return_to_depot_time - depot_departure_time if stops else 0
    hard_feasible = not violations and all(stop.hard_feasible for stop in stops)
    return DailyRoute(
        day=day,
        stops=stops,
        depot_departure_time=depot_departure_time,
        return_to_depot_time=return_to_depot_time,
        route_distance_km=route_distance_km,
        route_travel_time_min=route_travel_time_min,
        route_waiting_time_min=route_waiting_time_min,
        route_service_time_min=route_service_time_min,
        route_duration_min=route_duration_min,
        hard_feasible=hard_feasible,
        violations=violations,
    )


def choose_earliest_feasible_window(
    instance: Instance,
    customer_id: str,
    day: int,
    arrival_time: int,
    service_time: int | None = None,
) -> TimeWindow | None:
    """Return the earliest same-day time window that can accept an arrival."""
    service_duration = DEFAULT_SERVICE_TIME_MIN if service_time is None else service_time
    for window in instance.windows_for_customer_day(customer_id, day):
        service_start = max(arrival_time, window.start_minute)
        if not ALLOW_WAITING and arrival_time < window.start_minute:
            continue
        if REQUIRE_SERVICE_END_WITHIN_WINDOW:
            if service_start + service_duration <= window.end_minute:
                return window
        elif service_start <= window.end_minute:
            return window
    return None


def _choose_depot_departure_time(instance: Instance, day: int, customer_sequence: list[str]) -> int:
    """Return a practical departure time that avoids waiting at depot before the first stop."""
    if not FLEXIBLE_DEPOT_DEPARTURE or not customer_sequence:
        return 0

    first_customer_id = customer_sequence[0]
    first_location = instance.locations.get(first_customer_id)
    if first_location is None:
        return 0

    travel_time = travel_time_between_minutes(instance.depot(), first_location)
    service_time = first_location.service_time if first_location.service_time > 0 else DEFAULT_SERVICE_TIME_MIN
    selected_window = choose_earliest_feasible_window(instance, first_customer_id, day, travel_time, service_time)
    if selected_window is None:
        return 0
    return max(0, selected_window.start_minute - travel_time)


def evaluate_weekly_schedule(instance: Instance, schedule: WeeklySchedule) -> EvaluationMetrics:
    """Evaluate aggregate metrics for a weekly schedule."""
    violations = validate_schedule(instance, schedule)
    customer_ids = set(instance.customer_ids())
    delivered_by_day: dict[str, int] = {}

    total_distance_km = 0.0
    total_travel_time_min = 0
    total_waiting_time_min = 0
    total_service_time_min = 0
    total_route_duration_min = 0
    active_days = 0

    for day in range(MONDAY, SUNDAY + 1):
        route = schedule.routes.get(day)
        if route is None:
            continue
        if route.stops:
            active_days += 1
        total_distance_km += route.route_distance_km
        total_travel_time_min += route.route_travel_time_min
        total_waiting_time_min += route.route_waiting_time_min
        total_service_time_min += route.route_service_time_min
        total_route_duration_min += route.route_duration_min if route.stops else 0
        for stop in route.stops:
            if stop.hard_feasible and stop.customer_id in customer_ids and stop.customer_id not in delivered_by_day:
                delivered_by_day[stop.customer_id] = day

    incomplete_count = len(customer_ids - set(delivered_by_day))
    total_deferral_days = sum(day - min(instance.available_days(customer_id)) for customer_id, day in delivered_by_day.items())
    objective_value = calculate_objective_value(
        incomplete_count=incomplete_count,
        total_deferral_days=total_deferral_days,
        total_distance_km=total_distance_km,
        total_waiting_time_min=total_waiting_time_min,
    )

    return EvaluationMetrics(
        delivered_count=len(delivered_by_day),
        incomplete_count=incomplete_count,
        total_distance_km=total_distance_km,
        total_travel_time_min=total_travel_time_min,
        total_waiting_time_min=total_waiting_time_min,
        total_service_time_min=total_service_time_min,
        total_route_duration_min=total_route_duration_min,
        number_of_active_days=active_days,
        total_deferral_days=total_deferral_days,
        hard_feasible=not violations,
        objective_value=objective_value,
        violations=violations,
    )


def evaluate_schedule(instance: Instance, schedule: WeeklySchedule) -> EvaluationMetrics:
    """Backward-compatible alias for weekly schedule evaluation."""
    return evaluate_weekly_schedule(instance, schedule)


def validate_schedule(instance: Instance, schedule: WeeklySchedule) -> list[str]:
    """Return hard-feasibility violations for a schedule."""
    violations: list[str] = []
    customer_ids = set(instance.customer_ids())
    seen_counter: Counter[str] = Counter()

    for day, route in sorted(schedule.routes.items()):
        if day < MONDAY or day > SUNDAY:
            violations.append(f"Invalid route day: {day}")
        if route.day != day:
            violations.append(f"Route key {day} does not match route day {route.day}")
        if REQUIRE_RETURN_BEFORE_DAY_END and route.return_to_depot_time > DAY_END_MIN:
            violations.append(f"Route on day {day} returns after 24:00")
        for route_violation in route.violations:
            violations.append(f"Day {day}: {route_violation}")
        for stop in route.stops:
            seen_counter[stop.customer_id] += 1
            if stop.customer_id not in customer_ids:
                violations.append(f"Day {day}: unknown customer {stop.customer_id}")
                continue
            if not stop.hard_feasible:
                violations.append(f"Day {day}: infeasible stop {stop.customer_id}: {stop.violation or 'unknown reason'}")
            if stop.selected_time_window is None:
                violations.append(f"Day {day}: missing selected time window for {stop.customer_id}")
                continue
            if stop.selected_time_window.location_id != stop.customer_id:
                violations.append(f"Day {day}: selected time window belongs to another customer for {stop.customer_id}")
            if stop.selected_time_window.day_of_week != day:
                violations.append(f"Day {day}: selected time window has wrong day for {stop.customer_id}")
            if REQUIRE_SERVICE_END_WITHIN_WINDOW and stop.service_end_time > stop.selected_time_window.end_minute:
                violations.append(f"Day {day}: service end outside selected window for {stop.customer_id}")
            if not (stop.selected_time_window.start_minute <= stop.service_start_time <= stop.selected_time_window.end_minute):
                violations.append(f"Day {day}: service start outside selected window for {stop.customer_id}")

    duplicates = sorted(customer_id for customer_id, count in seen_counter.items() if count > 1)
    if duplicates:
        violations.append(f"Customers delivered more than once: {duplicates}")

    return violations


def calculate_objective_value(
    *,
    incomplete_count: int,
    total_deferral_days: int,
    total_distance_km: float,
    total_waiting_time_min: float,
) -> float:
    """Compute the official weighted objective for a complete schedule."""
    values = {
        "incomplete_count": incomplete_count,
        "total_deferral_days": total_deferral_days,
        "total_distance_km": total_distance_km,
        "total_waiting_time_min": total_waiting_time_min,
    }
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    return (
        WEIGHT_INCOMPLETE * incomplete_count
        + WEIGHT_DEFERRAL * total_deferral_days
        + WEIGHT_DISTANCE_KM * total_distance_km
        + WEIGHT_WAITING_MIN * total_waiting_time_min
    )


def calculate_objective(
    incomplete_count: int,
    total_deferral_days: int,
    total_distance_km: float,
    total_waiting_time_min: float,
    active_days: int = 0,
    total_route_duration_min: int = 0,
) -> float:
    """Backward-compatible wrapper; reporting-only arguments are ignored."""
    del active_days, total_route_duration_min
    return calculate_objective_value(
        incomplete_count=incomplete_count,
        total_deferral_days=total_deferral_days,
        total_distance_km=total_distance_km,
        total_waiting_time_min=total_waiting_time_min,
    )


def objective_breakdown(metrics: EvaluationMetrics) -> dict[str, float]:
    """Return the four official weighted components for display and export."""
    return {
        "incomplete_component": WEIGHT_INCOMPLETE * metrics.incomplete_count,
        "deferral_component": WEIGHT_DEFERRAL * metrics.total_deferral_days,
        "distance_component": WEIGHT_DISTANCE_KM * metrics.total_distance_km,
        "waiting_component": WEIGHT_WAITING_MIN * metrics.total_waiting_time_min,
    }


def official_objective_status(metrics: EvaluationMetrics) -> dict[str, object]:
    """Return canonical objective fields for solver status dictionaries."""
    breakdown = objective_breakdown(metrics)
    return {
        "objective_version": OBJECTIVE_VERSION,
        "objective_value": metrics.objective_value,
        "objective_breakdown": breakdown,
        **breakdown,
    }


def complete_empty_weekly_schedule() -> WeeklySchedule:
    """Return a seven-day schedule with empty feasible routes."""
    return WeeklySchedule(routes={day: evaluate_daily_route_empty(day) for day in range(MONDAY, SUNDAY + 1)})


def evaluate_daily_route_empty(day: int) -> DailyRoute:
    """Return an empty daily route."""
    return DailyRoute(day=day, return_to_depot_time=0, hard_feasible=True)


def print_schedule(schedule: WeeklySchedule) -> None:
    """Print a readable daily route summary."""
    for day in range(MONDAY, SUNDAY + 1):
        route = schedule.routes.get(day)
        if route is None or not route.stops:
            print(f"Day {day}: no deliveries")
            continue
        sequence = " -> ".join(stop.customer_id for stop in route.stops)
        print(
            f"Day {day}: {sequence} | return={format_hhmm_safe(route.return_to_depot_time)} "
            f"| distance={route.route_distance_km:.2f} km | wait={route.route_waiting_time_min} min"
        )


def print_metrics(metrics: EvaluationMetrics) -> None:
    """Print evaluation metrics in a compact key-value format."""
    print("Official objective: 1000 * incomplete + 100 * deferral + 10 * distance + waiting")
    for key, value in metrics.to_dict().items():
        if key == "violations":
            continue
        if isinstance(value, float):
            print(f"{key}={value:.3f}")
        else:
            print(f"{key}={value}")
    if metrics.violations:
        print("violations:")
        for violation in metrics.violations:
            print(f"- {violation}")


def format_hhmm_safe(minutes: int) -> str:
    """Format minutes as HH:MM, allowing values beyond 24:00 for diagnostics."""
    if 0 <= minutes <= DAY_END_MIN:
        return format_hhmm(minutes)
    return f"{minutes}min"

