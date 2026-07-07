"""Solver factory used by CLI and benchmark entry points."""

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
from vrp_weekly.solvers.base import Solver
from vrp_weekly.solvers.cp_daily import CpDailySolver
from vrp_weekly.solvers.earliest_deadline import EarliestDeadlineSolver
from vrp_weekly.solvers.nearest_neighbor import NearestNeighborSolver
from vrp_weekly.solvers.regret_insertion import RegretInsertionSolver


def create_solver(solver_key: str, **kwargs: Any) -> Solver:
    """Create a solver instance from a CLI or benchmark solver key."""
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
        )
    raise ValueError(f"Unknown solver: {solver_key}")


def solver_names() -> list[str]:
    """Return supported solver names."""
    return ["nearest", "deadline", "regret", "cp"]
