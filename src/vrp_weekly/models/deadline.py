"""Earliest time-window-end baseline solver."""

from __future__ import annotations

import logging

from vrp_weekly.config import MONDAY, SUNDAY
from vrp_weekly.distance import travel_time_between_minutes
from vrp_weekly.evaluator import evaluate_daily_route
from vrp_weekly.core import DailyRoute, Instance, WeeklySchedule

LOGGER = logging.getLogger(__name__)


class EarliestDeadlineSolver:
    """Greedy baseline that chooses the feasible next customer with earliest window end."""

    name = "deadline"

    def solve(self, instance: Instance) -> WeeklySchedule:
        """Build routes using earliest feasible time-window end as priority."""
        undelivered = set(instance.customer_ids())
        routes: dict[int, DailyRoute] = {}

        for day in range(MONDAY, SUNDAY + 1):
            LOGGER.info("deadline day=%s start undelivered=%s", day, len(undelivered))
            sequence: list[str] = []
            current_location = instance.depot()

            while True:
                feasible_choices: list[tuple[int, int, int, str]] = []
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
                    feasible_choices.append((selected_window.end_minute, selected_window.start_minute, travel_time, customer_id))

                if not feasible_choices:
                    break

                _, _, _, chosen_customer = min(feasible_choices, key=lambda item: (item[0], item[1], item[2], item[3]))
                sequence.append(chosen_customer)
                undelivered.remove(chosen_customer)
                current_location = instance.locations[chosen_customer]

            routes[day] = evaluate_daily_route(instance, day, sequence)
            LOGGER.info(
                "deadline day=%s done stops=%s return=%s distance=%.2f remaining=%s",
                day,
                len(routes[day].stops),
                routes[day].return_to_depot_time,
                routes[day].route_distance_km,
                len(undelivered),
            )

        return WeeklySchedule(routes=routes)

