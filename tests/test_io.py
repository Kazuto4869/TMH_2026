"""Tests for CSV input parsing and summaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from vrp_weekly.io import load_instance, read_locations, read_time_windows, summarize_instance


def test_load_instance_parses_locations_and_grouped_time_windows(tmp_path: Path) -> None:
    """CSV inputs should load into an Instance with minute-based grouped windows."""
    locations_path, windows_path = _write_basic_csvs(tmp_path)

    instance = load_instance(locations_path, windows_path)

    assert instance.depot_id == "DEPOT"
    assert len(instance.locations) == 3
    assert instance.locations["C001"].service_time == 7
    assert instance.time_windows["C001"][1][0].start_minute == 510
    assert instance.time_windows["C001"][1][0].end_minute == 600


def test_read_locations_detects_first_row_depot_when_no_hint(tmp_path: Path) -> None:
    """Ambiguous files should use the first row as depot."""
    path = tmp_path / "locations.csv"
    path.write_text(
        "id,name,x,y\n"
        "A,Start,0,0\n"
        "B,Customer,1,1\n",
        encoding="utf-8",
    )

    locations, depot_id = read_locations(path)

    assert depot_id == "A"
    assert locations["A"].is_depot is True


def test_read_time_windows_rejects_invalid_window(tmp_path: Path) -> None:
    """Parser should reject non-positive time windows."""
    path = tmp_path / "time_windows.csv"
    path.write_text(
        "location_id,day_of_week,start_time,end_time\n"
        "C001,1,10:00,10:00\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="start must be before end"):
        read_time_windows(path)


def test_load_instance_rejects_customer_without_window(tmp_path: Path) -> None:
    """Every non-depot customer must have at least one weekly time window."""
    locations_path, windows_path = _write_basic_csvs(tmp_path)
    windows_path.write_text(
        "location_id,day_of_week,start_time,end_time\n"
        "C001,1,08:30,10:00\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Customers without any weekly time window"):
        load_instance(locations_path, windows_path)


def test_summarize_instance_reports_core_counts(tmp_path: Path) -> None:
    """Summary should report customers, depots, time windows, and coordinate bounds."""
    locations_path, windows_path = _write_basic_csvs(tmp_path)
    instance = load_instance(locations_path, windows_path)

    summary = summarize_instance(instance)

    assert summary["number_of_customers"] == 2
    assert summary["number_of_depots"] == 1
    assert summary["number_of_time_windows"] == 2
    assert summary["min_x_km"] == 0.0
    assert summary["max_y_km"] == 4.0


def _write_basic_csvs(tmp_path: Path) -> tuple[Path, Path]:
    """Write a small valid instance."""
    locations_path = tmp_path / "locations.csv"
    windows_path = tmp_path / "time_windows.csv"
    locations_path.write_text(
        "location_id,location_name,x_km,y_km,demand_kg,service_time\n"
        "DEPOT,Kho Trung Tam,0,0,0,0\n"
        "C001,Customer 1,3,4,1.5,7\n"
        "C002,Customer 2,1,1,2.0,5\n",
        encoding="utf-8",
    )
    windows_path.write_text(
        "location_id,day_of_week,start_time,end_time\n"
        "C001,1,08:30,10:00\n"
        "C002,2,09:00,11:00\n",
        encoding="utf-8",
    )
    return locations_path, windows_path
