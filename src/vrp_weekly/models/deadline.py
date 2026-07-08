"""Earliest time-window-end baseline solver."""

from __future__ import annotations

from vrp_weekly.config import MONDAY, SUNDAY
from vrp_weekly.distance import travel_time_between_minutes
from vrp_weekly.evaluator import evaluate_daily_route
from vrp_weekly.core import DailyRoute, Instance, WeeklySchedule


class EarliestDeadlineSolver:
    """Greedy baseline that chooses the feasible next customer with earliest window end."""

    name = "deadline"

    def solve(self, instance: Instance) -> WeeklySchedule:
        """Build routes using earliest feasible time-window end as priority."""
        undelivered = set(instance.customer_ids())
        routes: dict[int, DailyRoute] = {}

        for day in range(MONDAY, SUNDAY + 1):
            sequence: list[str] = []
            current_location = instance.depot()

            while True:
                feasible_choices: list[tuple[int, int, str]] = []
                for customer_id in sorted(undelivered):
                    if not instance.windows_for_customer_day(customer_id, day):
                        continue
                    trial_route = evaluate_daily_route(instance, day, sequence + [customer_id])
                    if not trial_route.hard_feasible or not trial_route.stops:
                        continue
                    selected_window = trial_route.stops[-1].selected_time_window
                    if selected_window is None:
                        continue
                    travel_time = travel_time_between_minutes(current_location, instance.locations[customer_id])
                    feasible_choices.append((selected_window.end_minute, travel_time, customer_id))

                if not feasible_choices:
                    break

                _, _, chosen_customer = min(feasible_choices, key=lambda item: (item[0], item[1], item[2]))
                sequence.append(chosen_customer)
                undelivered.remove(chosen_customer)
                current_location = instance.locations[chosen_customer]

            routes[day] = evaluate_daily_route(instance, day, sequence)

        return WeeklySchedule(routes=routes)

