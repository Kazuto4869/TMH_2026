"""Input parsers and data summaries for contest CSV files."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from statistics import mean
from typing import Any

from vrp_weekly.config import DEPOT_ID, MONDAY, SUNDAY
from vrp_weekly.core import Instance, Location, TimeWindow
from vrp_weekly.time_utils import parse_hhmm

LOGGER = logging.getLogger(__name__)

LOCATION_COLUMN_ALIASES = {
    "location_id": ("location_id", "id", "customer_id", "node_id"),
    "location_name": ("location_name", "name", "customer_name"),
    "x_km": ("x_km", "x", "lon_km", "longitude_km"),
    "y_km": ("y_km", "y", "lat_km", "latitude_km"),
    "demand_kg": ("demand_kg", "demand", "weight_kg", "order_kg"),
    "service_time": ("service_time", "service_time_min", "service_minutes"),
}

WINDOW_COLUMN_ALIASES = {
    "location_id": ("location_id", "id", "customer_id", "node_id"),
    "day_of_week": ("day_of_week", "day", "weekday"),
    "start_time": ("start_time", "window_start", "start", "tw_start"),
    "end_time": ("end_time", "window_end", "end", "tw_end"),
}


def read_locations(path: str | Path) -> tuple[dict[str, Location], str]:
    """Read locations from CSV and return locations plus detected depot id."""
    locations_path = Path(path)
    rows = _read_rows(locations_path)
    if not rows:
        raise ValueError(f"{locations_path} contains no location rows")

    header_map = _build_header_map(rows[0], LOCATION_COLUMN_ALIASES, {"location_id", "x_km", "y_km"})
    depot_id = _detect_depot_id(rows, header_map)
    locations: dict[str, Location] = {}

    for row in rows:
        location_id = _required_cell(row, header_map["location_id"], locations_path).strip()
        location_name = _optional_cell(row, header_map.get("location_name"), location_id).strip()
        service_time = _optional_int(row, header_map.get("service_time"), 5)
        if location_id == depot_id and header_map.get("service_time") is None:
            service_time = 0
        location = Location(
            location_id=location_id,
            location_name=location_name,
            x_km=float(_required_cell(row, header_map["x_km"], locations_path)),
            y_km=float(_required_cell(row, header_map["y_km"], locations_path)),
            demand_kg=_optional_float(row, header_map.get("demand_kg"), 0.0),
            service_time=service_time,
            is_depot=location_id == depot_id,
        )
        if location.location_id in locations:
            raise ValueError(f"Duplicate location_id: {location.location_id}")
        locations[location.location_id] = location

    return locations, depot_id


def read_time_windows(path: str | Path) -> dict[str, dict[int, list[TimeWindow]]]:
    """Read time windows from CSV grouped by `location_id` and `day_of_week`."""
    windows_path = Path(path)
    rows = _read_rows(windows_path)
    if not rows:
        raise ValueError(f"{windows_path} contains no time-window rows")

    header_map = _build_header_map(rows[0], WINDOW_COLUMN_ALIASES, set(WINDOW_COLUMN_ALIASES))
    grouped: dict[str, dict[int, list[TimeWindow]]] = {}

    for row in rows:
        location_id = _required_cell(row, header_map["location_id"], windows_path).strip()
        day = int(_required_cell(row, header_map["day_of_week"], windows_path))
        if day < MONDAY or day > SUNDAY:
            raise ValueError(f"Invalid day_of_week for {location_id}: {day}")

        start_minute = parse_hhmm(_required_cell(row, header_map["start_time"], windows_path))
        end_minute = parse_hhmm(_required_cell(row, header_map["end_time"], windows_path))
        if start_minute >= end_minute:
            raise ValueError(f"Time window start must be before end for {location_id} on day {day}")

        window = TimeWindow(
            location_id=location_id,
            day_of_week=day,
            start_minute=start_minute,
            end_minute=end_minute,
        )
        grouped.setdefault(location_id, {}).setdefault(day, []).append(window)

    for windows_by_day in grouped.values():
        for day, windows in windows_by_day.items():
            windows_by_day[day] = sorted(windows, key=lambda window: (window.start_minute, window.end_minute))

    return grouped


def load_locations(path: str | Path) -> dict[str, Location]:
    """Backward-compatible wrapper returning only parsed locations."""
    locations, _ = read_locations(path)
    return locations


def load_time_windows(path: str | Path) -> dict[str, dict[int, list[TimeWindow]]]:
    """Backward-compatible wrapper returning grouped time windows."""
    return read_time_windows(path)


def load_instance(locations_path: str | Path, time_windows_path: str | Path) -> Instance:
    """Load and validate a complete weekly routing instance from CSV inputs."""
    locations, depot_id = read_locations(locations_path)
    time_windows = read_time_windows(time_windows_path)

    unknown_ids = sorted(set(time_windows) - set(locations))
    if unknown_ids:
        raise ValueError(f"Time windows reference unknown location ids: {unknown_ids}")

    customer_ids = [location_id for location_id, location in locations.items() if not location.is_depot]
    customers_without_windows = sorted(location_id for location_id in customer_ids if location_id not in time_windows)
    if customers_without_windows:
        raise ValueError(f"Customers without any weekly time window: {customers_without_windows}")

    return Instance(locations=locations, time_windows=time_windows, depot_id=depot_id)


def summarize_instance(instance: Instance) -> dict[str, Any]:
    """Return a compact data summary for reporting and sanity checks."""
    customer_ids = instance.customer_ids()
    depot_ids = instance.depot_ids()
    available_day_counts = [len(instance.available_days(customer_id)) for customer_id in customer_ids]
    x_values = [location.x_km for location in instance.locations.values()]
    y_values = [location.y_km for location in instance.locations.values()]
    service_times = [instance.locations[customer_id].service_time for customer_id in customer_ids]
    time_window_count = sum(
        len(windows)
        for windows_by_day in instance.time_windows.values()
        for windows in windows_by_day.values()
    )

    summary: dict[str, Any] = {
        "number_of_customers": len(customer_ids),
        "number_of_depots": len(depot_ids),
        "depot_id": instance.depot_id,
        "number_of_time_windows": time_window_count,
        "min_available_days_per_customer": min(available_day_counts) if available_day_counts else 0,
        "avg_available_days_per_customer": mean(available_day_counts) if available_day_counts else 0.0,
        "max_available_days_per_customer": max(available_day_counts) if available_day_counts else 0,
        "min_x_km": min(x_values) if x_values else 0.0,
        "max_x_km": max(x_values) if x_values else 0.0,
        "min_y_km": min(y_values) if y_values else 0.0,
        "max_y_km": max(y_values) if y_values else 0.0,
        "total_demand_kg": sum(instance.locations[customer_id].demand_kg for customer_id in customer_ids),
    }
    if service_times:
        summary.update(
            {
                "min_service_time_min": min(service_times),
                "avg_service_time_min": mean(service_times),
                "max_service_time_min": max(service_times),
            }
        )
    return summary


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows with UTF-8 handling."""
    with path.open(newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        return [dict(row) for row in reader]


def _build_header_map(
    sample_row: dict[str, str],
    aliases: dict[str, tuple[str, ...]],
    required_columns: set[str],
) -> dict[str, str]:
    """Map canonical column names to actual CSV headers."""
    normalized = {_normalize_header(header): header for header in sample_row}
    header_map: dict[str, str] = {}
    for canonical, names in aliases.items():
        for name in names:
            actual = normalized.get(_normalize_header(name))
            if actual is not None:
                header_map[canonical] = actual
                break
    missing = sorted(required_columns - set(header_map))
    if missing:
        raise ValueError(f"CSV file is missing required columns: {missing}")
    return header_map


def _detect_depot_id(rows: list[dict[str, str]], header_map: dict[str, str]) -> str:
    """Detect the primary depot id from id/name hints, falling back to the first row."""
    id_column = header_map["location_id"]
    name_column = header_map.get("location_name")
    candidates: list[str] = []
    for row in rows:
        location_id = row[id_column].strip()
        location_name = row.get(name_column, "") if name_column else ""
        if _looks_like_depot(location_id) or _looks_like_depot(location_name):
            candidates.append(location_id)

    if len(candidates) == 1:
        return candidates[0]

    first_id = rows[0][id_column].strip()
    if len(candidates) > 1:
        LOGGER.warning("Ambiguous depot candidates %s; using first row %s as depot", candidates, first_id)
    else:
        LOGGER.warning("No depot-like row detected; using first row %s as depot", first_id)
    return first_id


def _looks_like_depot(value: str) -> bool:
    """Return true when text suggests a depot or warehouse."""
    text = value.strip().lower()
    return any(token in text for token in ("depot", "warehouse", "kho", "trung_tam", "trung tam"))


def _normalize_header(value: str) -> str:
    """Normalize a header for alias matching."""
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _required_cell(row: dict[str, str], column: str, path: Path) -> str:
    """Return a required cell value or raise a clear error."""
    value = row.get(column, "")
    if value is None or value == "":
        raise ValueError(f"{path} contains an empty required value in column {column!r}")
    return value


def _optional_cell(row: dict[str, str], column: str | None, default: str) -> str:
    """Return an optional string cell."""
    if column is None:
        return default
    value = row.get(column)
    return default if value is None or value == "" else value


def _optional_float(row: dict[str, str], column: str | None, default: float) -> float:
    """Return an optional float cell."""
    if column is None or row.get(column, "") == "":
        return default
    return float(row[column])


def _optional_int(row: dict[str, str], column: str | None, default: int) -> int:
    """Return an optional integer cell."""
    if column is None or row.get(column, "") == "":
        return default
    return int(float(row[column]))

