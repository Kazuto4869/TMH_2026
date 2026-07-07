"""Tests for time parsing utilities."""

from __future__ import annotations

import pytest

from vrp_weekly.time_utils import format_hhmm, parse_hhmm


def test_parse_hhmm_returns_minutes_from_midnight() -> None:
    """HH:MM values should convert to integer minutes."""
    assert parse_hhmm("00:00") == 0
    assert parse_hhmm("09:05") == 545
    assert parse_hhmm("23:59") == 1439
    assert parse_hhmm("24:00") == 1440


def test_parse_hhmm_rejects_invalid_values() -> None:
    """Invalid times should raise ValueError."""
    with pytest.raises(ValueError):
        parse_hhmm("24:01")
    with pytest.raises(ValueError):
        parse_hhmm("12:60")
    with pytest.raises(ValueError):
        parse_hhmm("bad")


def test_format_hhmm_returns_zero_padded_time() -> None:
    """Minute values should format back to HH:MM."""
    assert format_hhmm(0) == "00:00"
    assert format_hhmm(545) == "09:05"
    assert format_hhmm(1440) == "24:00"
