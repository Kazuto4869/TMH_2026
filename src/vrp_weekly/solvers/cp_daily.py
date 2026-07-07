"""Rolling-horizon daily CP solver using OR-Tools when available."""

from __future__ import annotations

import logging

from vrp_weekly.config import CP_TIME_LIMIT_PER_DAY_SEC, DEFAULT_SERVICE_TIME_MIN, DROP_PENALTY_BY_DAY, MONDAY, SUNDAY
from vrp_weekly.evaluator import evaluate_daily_route
from vrp_weekly.models import DailyRoute, Instance, WeeklySchedule
from vrp_weekly.solvers.base import Solver
from vrp_weekly.solvers.earliest_deadline import EarliestDeadlineSolver

LOGGER = logging.getLogger(__name__)


class CpDailySolver(Solver):
    """Daily CP/OR-Tools solver with a deterministic fallback when OR-Tools is unavailable."""

    name = "cp"

    def __init__(
        self,
        time_limit_per_day: int = CP_TIME_LIMIT_PER_DAY_SEC,
        drop_penalty_by_day: dict[int, int] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize CP solver parameters."""
        self.time_limit_per_day = time_limit_per_day
        self.drop_penalty_by_day = dict(DROP_PENALTY_BY_DAY if drop_penalty_by_day is None else drop_penalty_by_day)
        self.seed = seed

    def solve(self, instance: Instance) -> WeeklySchedule:
        """Solve with OR-Tools if installed, otherwise use the deadline baseline."""
        try:
            import ortools.constraint_solver.pywrapcp as pywrapcp  # noqa: F401
            import ortools.constraint_solver.routing_enums_pb2 as routing_enums_pb2  # noqa: F401
        except ImportError:
            LOGGER.warning("OR-Tools is not installed; cp solver is falling back to deadline baseline")
            return EarliestDeadlineSolver().solve(instance)

        return self._solve_with_ortools(instance)

    def _solve_with_ortools(self, instance: Instance) -> WeeklySchedule:
        """Build daily routes with a lightweight OR-Tools rolling horizon model."""
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2

        from vrp_weekly.distance import travel_time_between_minutes
        from vrp_weekly.time_utils import MINUTES_PER_DAY

        undelivered = set(instance.customer_ids())
        routes: dict[int, DailyRoute] = {}

        for day in range(MONDAY, SUNDAY + 1):
            candidates = sorted(customer for customer in undelivered if instance.windows_for_customer_day(customer, day))
            if not candidates:
                routes[day] = evaluate_daily_route(instance, day, [])
                continue

            node_ids = [instance.depot_id] + candidates
            manager = pywrapcp.RoutingIndexManager(len(node_ids), 1, 0)
            routing = pywrapcp.RoutingModel(manager)

            def transit_callback(from_index: int, to_index: int) -> int:
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                from_location = instance.locations[node_ids[from_node]]
                to_location = instance.locations[node_ids[to_node]]
                service_time = 0 if from_node == 0 else max(from_location.service_time, DEFAULT_SERVICE_TIME_MIN)
                return travel_time_between_minutes(from_location, to_location) + service_time

            transit_callback_index = routing.RegisterTransitCallback(transit_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            routing.AddDimension(transit_callback_index, MINUTES_PER_DAY, MINUTES_PER_DAY, True, "Time")
            time_dimension = routing.GetDimensionOrDie("Time")

            for node_index, customer_id in enumerate(node_ids[1:], start=1):
                index = manager.NodeToIndex(node_index)
                cumul = time_dimension.CumulVar(index)
                cumul.SetRange(0, MINUTES_PER_DAY)
                windows = instance.windows_for_customer_day(customer_id, day)
                service_time = max(instance.locations[customer_id].service_time, DEFAULT_SERVICE_TIME_MIN)
                valid_intervals = [(window.start_minute, max(window.start_minute, window.end_minute - service_time)) for window in windows]
                cursor = 0
                for start, end in sorted(valid_intervals):
                    if end < start:
                        continue
                    if cursor < start:
                        cumul.RemoveInterval(cursor, start - 1)
                    cursor = max(cursor, end + 1)
                if cursor <= MINUTES_PER_DAY:
                    cumul.RemoveInterval(cursor, MINUTES_PER_DAY)
                day_penalty = self.drop_penalty_by_day.get(day, DROP_PENALTY_BY_DAY[SUNDAY])
                routing.AddDisjunction([index], day_penalty)

            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            search_parameters.time_limit.seconds = self.time_limit_per_day
            solution = routing.SolveWithParameters(search_parameters)
            sequence: list[str] = []

            if solution is not None:
                index = routing.Start(0)
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    if node_index != 0:
                        sequence.append(node_ids[node_index])
                    index = solution.Value(routing.NextVar(index))

            route = evaluate_daily_route(instance, day, sequence)
            routes[day] = route
            undelivered -= route.delivered_customer_ids()

        return WeeklySchedule(routes=routes)
