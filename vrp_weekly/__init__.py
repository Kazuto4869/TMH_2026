"""Development import shim for running the src-layout package in place."""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "vrp_weekly"
if _SRC_PACKAGE.is_dir():
    __path__.append(str(_SRC_PACKAGE))

from vrp_weekly.models import (  # noqa: E402
    DailyRoute,
    EvaluationMetrics,
    Instance,
    Location,
    Stop,
    TimeWindow,
    WeeklySchedule,
)

__all__ = [
    "DailyRoute",
    "EvaluationMetrics",
    "Instance",
    "Location",
    "Stop",
    "TimeWindow",
    "WeeklySchedule",
]
