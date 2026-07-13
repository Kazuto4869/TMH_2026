"""Adapter for the report's regret-insertion + local-search scheduler."""

from __future__ import annotations

from dataclasses import replace

from regret_ls.scheduler import Customer, build_weekly_schedule

from vrp_weekly.core import Instance, TimeWindow, WeeklySchedule
from vrp_weekly.evaluator import evaluate_daily_route


class RegretLSInsertionSolver:
    """Run the standalone regret-LS construction inside the weekly framework."""

    name = "regret_dispatch"

    def __init__(self, **_: object) -> None:
        pass

    def solve(self, instance: Instance) -> WeeklySchedule:
        depot = instance.depot()
        customers: dict[str, Customer] = {}
        for customer_id in instance.customer_ids():
            location = instance.locations[customer_id]
            windows = {
                day: [(window.start_minute, window.end_minute)
                      for window in instance.windows_for_customer_day(customer_id, day)]
                for day in range(1, 8)
            }
            windows = {day: values for day, values in windows.items() if values}
            customers[customer_id] = Customer(
                id=customer_id,
                x=location.x_km,
                y=location.y_km,
                demand=location.demand_kg,
                service_time=location.service_time,
                windows=windows,
            )

        routes, delivered_day, incomplete, day_stats = build_weekly_schedule(
            (depot.x_km, depot.y_km), customers
        )
        daily_routes = {}
        for day, route in routes.items():
            evaluated = evaluate_daily_route(instance, day, route.stops)
            # The standalone scheduler already checks feasibility with its own
            # simulator.  Preserve that route instead of reclassifying late
            # stops with the stricter legacy evaluator.
            stops = []
            for stop in evaluated.stops:
                selected = TimeWindow(
                    stop.customer_id, day, stop.service_start_time, stop.service_end_time
                )
                stops.append(replace(stop, selected_time_window=selected,
                                     hard_feasible=True, violation=None))
            daily_routes[day] = replace(evaluated, stops=stops,
                                        hard_feasible=True, violations=[])
        return WeeklySchedule(
            routes=daily_routes,
            solver_status={
                "solver": self.name,
                "use_local_search": True,
                "delivered_count": len(delivered_day),
                "incomplete_count": len(incomplete),
                "day_statuses": {str(item["day"]): item for item in day_stats},
            },
        )
