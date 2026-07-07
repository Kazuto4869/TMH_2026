"""Utilities for converting and validating time values."""

from __future__ import annotations

MINUTES_PER_DAY = 24 * 60


def parse_hhmm(value: str) -> int:
    """Parse an `HH:MM` string into minutes from midnight.

    The value `24:00` is accepted and maps to 1440 for day-horizon
    boundaries. Other `24:xx` values are rejected.
    """
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time value: {value!r}")

    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid time value: {value!r}") from exc

    if hour == 24 and minute == 0:
        return MINUTES_PER_DAY
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid time value: {value!r}")

    return hour * 60 + minute


def format_hhmm(minutes: int) -> str:
    """Format minutes from midnight as `HH:MM`."""
    if minutes < 0 or minutes > MINUTES_PER_DAY:
        raise ValueError(f"Minutes out of day range: {minutes}")
    if minutes == MINUTES_PER_DAY:
        return "24:00"
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}"
