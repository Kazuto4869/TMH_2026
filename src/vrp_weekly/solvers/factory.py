"""Solver factory used by CLI and benchmark entry points."""

from __future__ import annotations

from typing import Any

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
            regret_weight=float(kwargs.get("regret_weight", 1.0)),
            insertion_weight=float(kwargs.get("insertion_weight", 1.0)),
            urgency_weight=float(kwargs.get("urgency_weight", 100.0)),
            waiting_weight=float(kwargs.get("waiting_weight", 0.2)),
            seed=kwargs.get("seed"),
        )
    if normalized == "cp":
        return CpDailySolver(
            time_limit_per_day=int(kwargs.get("cp_time_limit_per_day", 10)),
            drop_penalty_base=int(kwargs.get("drop_penalty_base", 10_000)),
            drop_penalty_growth=float(kwargs.get("drop_penalty_growth", 2.0)),
            seed=kwargs.get("seed"),
        )
    raise ValueError(f"Unknown solver: {solver_key}")


def solver_names() -> list[str]:
    """Return supported solver names."""
    return ["nearest", "deadline", "regret", "cp"]
