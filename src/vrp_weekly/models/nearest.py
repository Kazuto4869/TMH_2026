"""Nearest Neighbor baseline solver."""

from __future__ import annotations

from vrp_weekly.config import MONDAY, SUNDAY
from vrp_weekly.distance import euclidean_distance_km
from vrp_weekly.evaluator import evaluate_daily_route
from vrp_weekly.core import DailyRoute, Instance, WeeklySchedule


class NearestNeighborSolver:
    """Greedy nearest-neighbor weekly routing baseline."""

    name = "nearest"

    def solve(self, instance: Instance) -> WeeklySchedule:
        """Build routes by repeatedly appending the nearest feasible customer."""
        undelivered = set(instance.customer_ids())
        routes: dict[int, DailyRoute] = {}

        for day in range(MONDAY, SUNDAY + 1):
            sequence: list[str] = []
            current_location = instance.depot()

            while True:
                feasible_choices: list[tuple[float, str]] = []
                for customer_id in sorted(undelivered):
                    if not instance.windows_for_customer_day(customer_id, day):
                        continue
                    trial_route = evaluate_daily_route(instance, day, sequence + [customer_id])
                    if trial_route.hard_feasible:
                        distance = euclidean_distance_km(current_location, instance.locations[customer_id])
                        feasible_choices.append((distance, customer_id))

                if not feasible_choices:
                    break

                _, chosen_customer = min(feasible_choices, key=lambda item: (item[0], item[1]))
                sequence.append(chosen_customer)
                undelivered.remove(chosen_customer)
                current_location = instance.locations[chosen_customer]

            routes[day] = evaluate_daily_route(instance, day, sequence)

        return WeeklySchedule(routes=routes)

