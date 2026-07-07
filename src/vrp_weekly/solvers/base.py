"""Common solver interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from vrp_weekly.models import Instance, WeeklySchedule


class Solver(ABC):
    """Base interface implemented by all weekly routing solvers."""

    name: str

    @abstractmethod
    def solve(self, instance: Instance) -> WeeklySchedule:
        """Return a weekly schedule for the given instance."""
        raise NotImplementedError
