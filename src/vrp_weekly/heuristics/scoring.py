"""Shared deterministic scoring helpers for heuristic models."""

from __future__ import annotations

from vrp_weekly.config import DAY_END_MIN, MONDAY, SUNDAY
from vrp_weekly.core import Instance
from vrp_weekly.heuristics.route_eval import distance_km, windows_for


def earliest_available_day(instance: Instance, customer_id: str) -> int | None:
    """Return earliest day with at least one customer window."""
    days = available_days(instance, customer_id)
    return days[0] if days else None


def available_days(instance: Instance, customer_id: str) -> list[int]:
    """Return sorted days where a customer has at least one window."""
    return [day for day in range(MONDAY, SUNDAY + 1) if windows_for(instance, customer_id, day)]


def remaining_available_days(instance: Instance, customer_id: str, day: int) -> list[int]:
    """Return sorted available days from today onward."""
    return [available_day for available_day in available_days(instance, customer_id) if available_day >= day]


def is_last_available_day(instance: Instance, customer_id: str, day: int) -> bool:
    """Return true iff today is the customer's last remaining available day."""
    return remaining_available_days(instance, customer_id, day) == [day]


def total_window_width_today(instance: Instance, customer_id: str, day: int) -> int:
    """Return total width of all windows today."""
    return sum(window.end_minute - window.start_minute for window in windows_for(instance, customer_id, day))


def num_windows_today(instance: Instance, customer_id: str, day: int) -> int:
    """Return number of windows today."""
    return len(windows_for(instance, customer_id, day))


def earliest_window_end_today(instance: Instance, customer_id: str, day: int) -> int:
    """Return earliest window end today, or DAY_END_MIN + 1 if unavailable."""
    return min((window.end_minute for window in windows_for(instance, customer_id, day)), default=DAY_END_MIN + 1)


def deadline_pressure(instance: Instance, customer_id: str, day: int) -> float:
    """Return higher pressure for earlier same-day deadlines."""
    return max(0, DAY_END_MIN - earliest_window_end_today(instance, customer_id, day)) / DAY_END_MIN


def spatial_isolation(instance: Instance, customer_id: str, candidates: list[str] | set[str]) -> float:
    """Return a deterministic distance-from-depot isolation proxy."""
    if not candidates:
        return 0.0
    return distance_km(instance, instance.depot_id, customer_id)


def window_width_loss_to_future(instance: Instance, customer_id: str, day: int) -> float:
    """Return higher value when future delivery opportunities are fewer or narrower."""
    remaining = remaining_available_days(instance, customer_id, day)
    if len(remaining) <= 1:
        return 1.0
    future_width = sum(total_window_width_today(instance, customer_id, future_day) for future_day in remaining if future_day > day)
    today_width = total_window_width_today(instance, customer_id, day)
    return today_width / max(1, today_width + future_width)

