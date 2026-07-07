"""Distance and travel-time helpers."""

from __future__ import annotations

import math

from vrp_weekly.config import MAX_SPEED_KMPH, MINUTES_PER_HOUR
from vrp_weekly.models import Location


def euclidean_distance_km(origin: Location, destination: Location) -> float:
    """Return Euclidean distance between two locations in kilometers."""
    return math.hypot(destination.x_km - origin.x_km, destination.y_km - origin.y_km)


def travel_time_minutes(distance_km: float, speed_kmph: float = MAX_SPEED_KMPH) -> int:
    """Return ceiling travel time in minutes for a distance and speed."""
    if speed_kmph <= 0:
        raise ValueError("speed_kmph must be positive")
    return math.ceil(MINUTES_PER_HOUR * distance_km / speed_kmph)


def travel_time_between_minutes(origin: Location, destination: Location) -> int:
    """Return ceiling travel time in minutes between two locations."""
    return travel_time_minutes(euclidean_distance_km(origin, destination))
