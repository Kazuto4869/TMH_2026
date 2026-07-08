"""Factory for available routing models."""

from __future__ import annotations

from typing import Any

from vrp_weekly.config import (
    DROP_PENALTY_BY_DAY,
    INSERTION_WEIGHT,
    REGRET_WEIGHT,
    URGENCY_WEIGHT,
    WAITING_WEIGHT,
)
from vrp_weekly.models.cp_full_week import FullWeekCPSATSolver
from vrp_weekly.models.cp_rolling_horizon import RollingHorizonCPSATSolver
from vrp_weekly.models.deadline import EarliestDeadlineSolver
from vrp_weekly.models.nearest import NearestNeighborSolver
from vrp_weekly.models.regret import RegretInsertionSolver


def create_solver(solver_key: str, **kwargs: Any) -> object:
    """Create a model instance from a CLI or benchmark key."""
    normalized = "deadline" if solver_key == "earliest" else solver_key
    if normalized == "nearest":
        return NearestNeighborSolver()
    if normalized == "deadline":
        return EarliestDeadlineSolver()
    if normalized == "regret":
        return RegretInsertionSolver(
            regret_weight=float(kwargs.get("regret_weight", REGRET_WEIGHT)),
            insertion_weight=float(kwargs.get("insertion_weight", INSERTION_WEIGHT)),
            urgency_weight=float(kwargs.get("urgency_weight", URGENCY_WEIGHT)),
            waiting_weight=float(kwargs.get("waiting_weight", WAITING_WEIGHT)),
            seed=kwargs.get("seed"),
        )
    if normalized in ("cp", "cp_full_week"):
        return FullWeekCPSATSolver(
            time_limit_sec=int(kwargs.get("cp_time_limit_sec", 60)),
            max_customers=kwargs.get("cp_max_customers", 300),
            incomplete_weight=int(kwargs.get("incomplete_weight", 1_000_000)),
            deferral_weight=int(kwargs.get("deferral_weight", 10_000)),
            distance_weight=int(kwargs.get("distance_weight", 10)),
            route_duration_weight=int(kwargs.get("route_duration_weight", 1)),
            num_workers=int(kwargs.get("cp_workers", 8)),
            log_search_progress=bool(kwargs.get("cp_log_search", False)),
        )
    if normalized == "cp_rolling":
        return RollingHorizonCPSATSolver(
            time_limit_per_day_sec=int(kwargs.get("cp_time_limit_per_day_sec", 10)),
            max_candidates_per_day=kwargs.get("cp_max_candidates_per_day"),
            drop_penalty_by_day=kwargs.get("drop_penalty_by_day", DROP_PENALTY_BY_DAY),
            distance_weight=int(kwargs.get("distance_weight", 10)),
            route_duration_weight=int(kwargs.get("route_duration_weight", 1)),
            urgency_weight=int(kwargs.get("urgency_weight", 100)),
            num_workers=int(kwargs.get("cp_workers", 8)),
            log_search_progress=bool(kwargs.get("cp_log_search", False)),
        )
    raise ValueError(f"Unknown solver: {solver_key}")


def solver_names() -> list[str]:
    """Return supported model names."""
    return ["nearest", "deadline", "regret", "cp_full_week", "cp_rolling"]
