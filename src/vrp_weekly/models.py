"""Core typed data structures for the weekly routing problem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrp_weekly.config import DEPOT_ID


@dataclass(frozen=True)
class Location:
    """A depot or customer location in Cartesian kilometer coordinates."""

    location_id: str
    location_name: str
    x_km: float
    y_km: float
    demand_kg: float = 0.0
    service_time: int = 5
    is_depot: bool = False


@dataclass(frozen=True)
class TimeWindow:
    """A delivery time window for a location on a specific day."""

    location_id: str
    day_of_week: int
    start_minute: int
    end_minute: int


@dataclass(frozen=True)
class Instance:
    """A weekly routing instance containing locations and grouped time windows."""

    locations: dict[str, Location]
    time_windows: dict[str, dict[int, list[TimeWindow]]]
    depot_id: str = DEPOT_ID

    def depot(self) -> Location:
        """Return the primary depot location."""
        return self.locations[self.depot_id]

    def depot_ids(self) -> list[str]:
        """Return location ids marked as depots."""
        return sorted(location_id for location_id, location in self.locations.items() if location.is_depot)

    def customer_ids(self) -> list[str]:
        """Return all non-depot location ids in deterministic order."""
        return sorted(location_id for location_id, location in self.locations.items() if not location.is_depot)

    def windows_for_customer_day(self, location_id: str, day_of_week: int) -> list[TimeWindow]:
        """Return sorted windows for a customer on a specific day."""
        windows = self.time_windows.get(location_id, {}).get(day_of_week, [])
        return sorted(windows, key=lambda window: (window.start_minute, window.end_minute))

    def windows_for_customer(self, location_id: str) -> list[TimeWindow]:
        """Return all sorted windows for a customer across the week."""
        windows_by_day = self.time_windows.get(location_id, {})
        windows: list[TimeWindow] = []
        for day in sorted(windows_by_day):
            windows.extend(sorted(windows_by_day[day], key=lambda window: (window.start_minute, window.end_minute)))
        return windows

    def windows_for_day(self, day_of_week: int) -> list[TimeWindow]:
        """Return all time windows available on the given day."""
        windows: list[TimeWindow] = []
        for location_windows in self.time_windows.values():
            windows.extend(location_windows.get(day_of_week, []))
        return sorted(windows, key=lambda window: (window.day_of_week, window.end_minute, window.start_minute))

    def available_days(self, location_id: str) -> set[int]:
        """Return days with at least one time window for a customer."""
        return set(self.time_windows.get(location_id, {}))


@dataclass(frozen=True)
class Stop:
    """A scheduled or attempted customer visit."""

    customer_id: str
    arrival_time: int
    service_start_time: int
    service_end_time: int
    selected_time_window: TimeWindow | None = None
    travel_from_previous_min: int = 0
    waiting_min: int = 0
    distance_from_previous_km: float = 0.0
    hard_feasible: bool = True
    violation: str | None = None

    @property
    def location_id(self) -> str:
        """Backward-compatible alias for customer_id."""
        return self.customer_id

    @property
    def arrival_minute(self) -> int:
        """Backward-compatible alias for arrival_time."""
        return self.arrival_time

    @property
    def service_start_minute(self) -> int:
        """Backward-compatible alias for service_start_time."""
        return self.service_start_time

    @property
    def departure_minute(self) -> int:
        """Backward-compatible alias for service_end_time."""
        return self.service_end_time


@dataclass(frozen=True)
class DailyRoute:
    """A route for one day, starting and ending at the depot."""

    day: int
    stops: list[Stop] = field(default_factory=list)
    depot_departure_time: int = 0
    return_to_depot_time: int = 0
    route_distance_km: float = 0.0
    route_travel_time_min: int = 0
    route_waiting_time_min: int = 0
    route_service_time_min: int = 0
    route_duration_min: int = 0
    hard_feasible: bool = True
    violations: list[str] = field(default_factory=list)

    @property
    def day_of_week(self) -> int:
        """Backward-compatible alias for day."""
        return self.day

    @property
    def depot_departure_minute(self) -> int:
        """Backward-compatible alias for depot_departure_time."""
        return self.depot_departure_time

    @property
    def depot_return_minute(self) -> int:
        """Backward-compatible alias for return_to_depot_time."""
        return self.return_to_depot_time

    def delivered_customer_ids(self) -> set[str]:
        """Return feasible customer ids served by this route."""
        return {stop.customer_id for stop in self.stops if stop.hard_feasible}

    def customer_sequence(self) -> list[str]:
        """Return the ordered customer ids in this route."""
        return [stop.customer_id for stop in self.stops]


@dataclass(frozen=True)
class WeeklySchedule:
    """A candidate seven-day delivery schedule."""

    routes: dict[int, DailyRoute] = field(default_factory=dict)

    def delivered_customer_ids(self) -> set[str]:
        """Return the set of unique feasible customers served during the week."""
        delivered: set[str] = set()
        for route in self.routes.values():
            delivered.update(route.delivered_customer_ids())
        return delivered

    def ordered_routes(self) -> list[DailyRoute]:
        """Return daily routes sorted from Monday through Sunday."""
        return [self.routes[day] for day in sorted(self.routes)]


@dataclass(frozen=True)
class EvaluationMetrics:
    """Summary metrics for a weekly schedule."""

    delivered_count: int
    incomplete_count: int
    total_distance_km: float
    total_travel_time_min: int
    total_waiting_time_min: int
    total_service_time_min: int
    total_route_duration_min: int
    number_of_active_days: int
    total_deferral_days: int
    hard_feasible: bool
    objective_value: float = 0.0
    violations: list[str] = field(default_factory=list)

    @property
    def delivered_orders(self) -> int:
        """Backward-compatible alias for delivered_count."""
        return self.delivered_count

    @property
    def incomplete_orders(self) -> int:
        """Backward-compatible alias for incomplete_count."""
        return self.incomplete_count

    @property
    def feasible(self) -> bool:
        """Backward-compatible alias for hard_feasible."""
        return self.hard_feasible

    def to_dict(self) -> dict[str, Any]:
        """Return metrics as a JSON-serializable dictionary."""
        return {
            "delivered_count": self.delivered_count,
            "incomplete_count": self.incomplete_count,
            "total_distance_km": self.total_distance_km,
            "total_travel_time_min": self.total_travel_time_min,
            "total_waiting_time_min": self.total_waiting_time_min,
            "total_service_time_min": self.total_service_time_min,
            "total_route_duration_min": self.total_route_duration_min,
            "number_of_active_days": self.number_of_active_days,
            "active_days": self.number_of_active_days,
            "total_deferral_days": self.total_deferral_days,
            "hard_feasible": self.hard_feasible,
            "objective_value": self.objective_value,
            "violations": self.violations,
        }
