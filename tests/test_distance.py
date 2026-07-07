"""Tests for distance and travel-time helpers."""

from __future__ import annotations

from vrp_weekly.distance import euclidean_distance_km, travel_time_minutes
from vrp_weekly.models import Location


def test_euclidean_distance_km() -> None:
    """Distance should use Euclidean geometry in kilometer coordinates."""
    origin = Location("A", "A", 0.0, 0.0, 0.0, 0)
    destination = Location("B", "B", 3.0, 4.0, 0.0, 0)

    assert euclidean_distance_km(origin, destination) == 5.0


def test_travel_time_minutes_ceil_at_50_kmph() -> None:
    """Travel time should be ceil(60 * distance / 50)."""
    assert travel_time_minutes(0.0) == 0
    assert travel_time_minutes(10.0) == 12
    assert travel_time_minutes(10.1) == 13
