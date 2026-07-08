"""Factory for available routing models."""

from __future__ import annotations

from typing import Any

from vrp_weekly.config import (
    CP_TIME_LIMIT_PER_DAY_SEC,
    DROP_PENALTY_BY_DAY,
    INSERTION_WEIGHT,
    REGRET_WEIGHT,
    URGENCY_WEIGHT,
    WAITING_WEIGHT,
)
from vrp_weekly.models.cp import CpDailySolver
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
    if normalized == "cp":
        return CpDailySolver(
            time_limit_per_day=int(kwargs.get("cp_time_limit_per_day", CP_TIME_LIMIT_PER_DAY_SEC)),
            drop_penalty_by_day=kwargs.get("drop_penalty_by_day", DROP_PENALTY_BY_DAY),
            seed=kwargs.get("seed"),
            threads=int(kwargs.get("cp_threads", 1)),
            log_search=bool(kwargs.get("cp_log_search", False)),
        )
    raise ValueError(f"Unknown solver: {solver_key}")


def solver_names() -> list[str]:
    """Return supported model names."""
    return ["nearest", "deadline", "regret", "cp"]
