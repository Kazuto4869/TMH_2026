"""Schedule simulation, validation, and pretty-printing utilities."""

from __future__ import annotations

from collections import Counter

from vrp_weekly.config import DAYS_IN_WEEK, MONDAY, SUNDAY
from vrp_weekly.distance import euclidean_distance_km, travel_time_between_minutes
from vrp_weekly.models import DailyRoute, EvaluationMetrics, Instance, Stop, TimeWindow, WeeklySchedule
from vrp_weekly.time_utils import MINUTES_PER_DAY, format_hhmm

DEFAULT_SERVICE_TIME_MIN = 5


def evaluate_daily_route(instance: Instance, day: int, customer_sequence: list[str]) -> DailyRoute:
    """Simulate one daily route and choose the earliest feasible window per stop."""
    if day < MONDAY or day > SUNDAY:
        raise ValueError(f"day must be in 1..7, got {day}")

    previous_location = instance.depot()
    current_time = 0
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
        selected_window = choose_earliest_feasible_window(instance, customer_id, day, arrival_time)

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
    if return_to_depot_time > MINUTES_PER_DAY:
        violations.append(f"Route on day {day} returns after 24:00 at {return_to_depot_time} minutes")

    hard_feasible = not violations and all(stop.hard_feasible for stop in stops)
    return DailyRoute(
        day=day,
        stops=stops,
        depot_departure_time=0,
        return_to_depot_time=return_to_depot_time,
        route_distance_km=route_distance_km,
        route_travel_time_min=route_travel_time_min,
        route_waiting_time_min=route_waiting_time_min,
        route_service_time_min=route_service_time_min,
        route_duration_min=return_to_depot_time,
        hard_feasible=hard_feasible,
        violations=violations,
    )


def choose_earliest_feasible_window(
    instance: Instance,
    customer_id: str,
    day: int,
    arrival_time: int,
) -> TimeWindow | None:
    """Return the earliest same-day time window that can accept an arrival."""
    for window in instance.windows_for_customer_day(customer_id, day):
        service_start = max(arrival_time, window.start_minute)
        if service_start <= window.end_minute:
            return window
    return None


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
    total_deferral_days = sum(day - 1 for day in delivered_by_day.values())
    objective_value = calculate_objective(
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
        if route.return_to_depot_time > MINUTES_PER_DAY:
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
            if not (stop.selected_time_window.start_minute <= stop.service_start_time <= stop.selected_time_window.end_minute):
                violations.append(f"Day {day}: service start outside selected window for {stop.customer_id}")

    duplicates = sorted(customer_id for customer_id, count in seen_counter.items() if count > 1)
    if duplicates:
        violations.append(f"Customers delivered more than once: {duplicates}")

    return violations


def calculate_objective(
    incomplete_count: int,
    total_deferral_days: int,
    total_distance_km: float,
    total_waiting_time_min: int,
) -> float:
    """Compute the weighted benchmark objective value."""
    return (
        1_000_000 * incomplete_count
        + 10_000 * total_deferral_days
        + 10 * total_distance_km
        + total_waiting_time_min
    )


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
    if 0 <= minutes <= MINUTES_PER_DAY:
        return format_hhmm(minutes)
    return f"{minutes}min"
