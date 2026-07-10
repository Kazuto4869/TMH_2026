"""Pure CP-SAT repair layer for the rolling-horizon CP schedule."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ortools.sat.python import cp_model

from vrp_weekly.config import DAY_END_MIN, MONDAY, SUNDAY
from vrp_weekly.core import DailyRoute, Instance, TimeWindow, WeeklySchedule
from vrp_weekly.evaluator import evaluate_weekly_schedule
from vrp_weekly.models.cp_rolling_horizon import (
    RollingHorizonCPSATSolver,
    _build_daily_schedule_from_solution,
    _can_follow,
    _distance_scaled,
    _extract_route_from_arcs,
    _get_service_time,
    _travel_time_minutes,
)


@dataclass
class RepairModelData:
    """Variables and model metadata for the restricted repair CP."""

    model: cp_model.CpModel
    selected_days: list[int]
    repair_customers: list[str]
    original_incomplete: set[str]
    y: dict[tuple[str, int], cp_model.IntVar]
    g: dict[tuple[str, int, int], cp_model.IntVar]
    t: dict[tuple[str, int], cp_model.IntVar]
    x: dict[tuple[int, str, str], cp_model.IntVar]
    z: dict[int, cp_model.IntVar]
    departure: dict[int, cp_model.IntVar]
    return_time: dict[int, cp_model.IntVar]
    windows: dict[tuple[str, int], list[TimeWindow]]
    nodes_by_day: dict[int, list[str]]


class RollingHorizonCPRepairSolver:
    """Run rolling CP, then repair a restricted weekly neighborhood with CP-SAT."""

    name = "cp_rolling_repair"

    def __init__(
        self,
        repair_time_limit_sec: int = 300,
        repair_max_days: int = 2,
        repair_max_customers: int = 120,
        repair_random_seed: int = 1,
        repair_num_workers: int = 4,
        repair_use_decision_strategy: bool = True,
        repair_optimize_route_cost: bool = True,
        **rolling_kwargs: Any,
    ) -> None:
        self.repair_time_limit_sec = repair_time_limit_sec
        self.repair_max_days = repair_max_days
        self.repair_max_customers = repair_max_customers
        self.repair_random_seed = repair_random_seed
        self.repair_num_workers = repair_num_workers
        self.repair_use_decision_strategy = repair_use_decision_strategy
        self.repair_optimize_route_cost = repair_optimize_route_cost
        self.rolling_kwargs = dict(rolling_kwargs)
        self.rolling_kwargs.setdefault("adaptive_daily_deadline", True)
        self.rolling_kwargs.setdefault("optimization_mode", "full_three_stage")
        self.rolling_kwargs.setdefault("stage2_max_time_fraction", 0.10)

    def solve(self, instance: Instance) -> WeeklySchedule:
        """Return a base rolling schedule or an accepted repaired schedule."""
        total_start = time.perf_counter()
        base_solver = RollingHorizonCPSATSolver(**self.rolling_kwargs)
        base_schedule = base_solver.solve(instance)
        base_metrics = evaluate_weekly_schedule(instance, base_schedule)
        base_delivered = base_schedule.delivered_customer_ids()
        incomplete_ids = sorted(set(instance.customer_ids()) - base_delivered)
        status = self._base_status(base_schedule, base_metrics, incomplete_ids)

        if base_metrics.incomplete_count == 0:
            status.update(
                {
                    "repair_ran": False,
                    "repair_reason": "no_incomplete_orders",
                    "repair_total_runtime_sec": time.perf_counter() - total_start,
                }
            )
            return self._with_status(base_schedule, status)

        selected_days = self._select_repair_days(instance, base_schedule, incomplete_ids)
        repair_customers = self._repair_customers(instance, base_schedule, selected_days, incomplete_ids)
        while selected_days and len(repair_customers) > self.repair_max_customers:
            selected_days = selected_days[:-1]
            repair_customers = self._repair_customers(instance, base_schedule, selected_days, incomplete_ids)
        if not selected_days or not repair_customers:
            status.update(
                {
                    "repair_ran": False,
                    "repair_reason": "repair_neighborhood_too_large",
                    "repair_total_runtime_sec": time.perf_counter() - total_start,
                }
            )
            return self._with_status(base_schedule, status)

        deadline = time.perf_counter() + self.repair_time_limit_sec
        repair_delivered_base = {
            customer_id
            for day in selected_days
            for customer_id in base_schedule.routes.get(day, DailyRoute(day=day)).customer_sequence()
        }
        data = self._build_repair_model(instance, selected_days, repair_customers, set(incomplete_ids), repair_delivered_base)
        hint_counts = self._add_base_hints(data, instance, base_schedule)
        self._add_decision_strategy(data)

        r1_start = time.perf_counter()
        rescued_terms = [
            var for (customer_id, _day), var in data.y.items() if customer_id in data.original_incomplete
        ]
        data.model.Maximize(sum(rescued_terms))
        r1_solver = self._solve(data.model, max(0.0, deadline - time.perf_counter()))
        r1_status = self._status_name(r1_solver)
        r1_time = time.perf_counter() - r1_start
        rescued_count = sum(int(r1_solver.BooleanValue(var)) for var in rescued_terms) if self._has_solution(r1_solver) else 0
        status.update(
            {
                "repair_ran": True,
                "repair_reason": "attempted",
                "repair_selected_days": selected_days,
                "repair_customer_count": len(repair_customers),
                "repair_stage_r1_status": r1_status,
                "repair_stage_r1_time_sec": r1_time,
                **hint_counts,
            }
        )
        if rescued_count == 0:
            status.update(
                {
                    "repair_accepted": False,
                    "repair_reason": "no_incomplete_customer_rescued",
                    "repair_rescued_count": 0,
                    "repair_total_runtime_sec": time.perf_counter() - total_start,
                }
            )
            return self._with_status(base_schedule, status)

        r2_data = self._build_repair_model(
            instance,
            selected_days,
            repair_customers,
            set(incomplete_ids),
            repair_delivered_base,
            fixed_rescued_count=rescued_count,
        )
        self._hint_from_solution(r2_data, data, r1_solver)
        self._add_decision_strategy(r2_data)
        r2_start = time.perf_counter()
        deferral_terms = [
            (day - min(instance.available_days(customer_id))) * var
            for (customer_id, day), var in r2_data.y.items()
            if instance.available_days(customer_id)
        ]
        r2_data.model.Minimize(sum(deferral_terms))
        r2_solver = self._solve(r2_data.model, max(0.0, deadline - time.perf_counter()))
        r2_status = self._status_name(r2_solver)
        r2_time = time.perf_counter() - r2_start
        if not self._has_solution(r2_solver):
            status.update(
                {
                    "repair_accepted": False,
                    "repair_reason": "repair_stage_r2_no_solution",
                    "repair_stage_r2_status": r2_status,
                    "repair_stage_r2_time_sec": r2_time,
                    "repair_total_runtime_sec": time.perf_counter() - total_start,
                }
            )
            return self._with_status(base_schedule, status)
        repaired_deferral_value = int(round(r2_solver.ObjectiveValue()))

        final_data = r2_data
        final_solver = r2_solver
        r3_status = ""
        r3_time = 0.0
        if self.repair_optimize_route_cost and deadline - time.perf_counter() > 0:
            r3_data = self._build_repair_model(
                instance,
                selected_days,
                repair_customers,
                set(incomplete_ids),
                repair_delivered_base,
                fixed_rescued_count=rescued_count,
                fixed_deferral_value=repaired_deferral_value,
            )
            self._hint_from_solution(r3_data, r2_data, r2_solver)
            self._add_decision_strategy(r3_data)
            r3_start = time.perf_counter()
            distance_terms = [
                _distance_scaled(instance, i, j) * var
                for (day, i, j), var in r3_data.x.items()
                if i != j and i != instance.depot_id and j != instance.depot_id
            ]
            depot_terms = [
                _distance_scaled(instance, i, j) * var
                for (day, i, j), var in r3_data.x.items()
                if i != j and (i == instance.depot_id or j == instance.depot_id)
            ]
            duration_terms = [r3_data.return_time[day] - r3_data.departure[day] for day in selected_days]
            r3_data.model.Minimize(sum(distance_terms) + sum(depot_terms) + sum(duration_terms))
            r3_solver = self._solve(r3_data.model, max(0.0, deadline - time.perf_counter()))
            r3_status = self._status_name(r3_solver)
            r3_time = time.perf_counter() - r3_start
            if self._has_solution(r3_solver):
                final_data = r3_data
                final_solver = r3_solver

        repaired_routes = dict(base_schedule.routes)
        for day in selected_days:
            repaired_routes[day] = self._extract_day_route(instance, final_data, final_solver, day)
        candidate_schedule = WeeklySchedule(routes=repaired_routes, solver_status=dict(base_schedule.solver_status))
        repaired_metrics = evaluate_weekly_schedule(instance, candidate_schedule)
        no_duplicates = self._no_duplicates(candidate_schedule)
        preserved = base_delivered <= candidate_schedule.delivered_customer_ids()
        lexicographic_better = self._lexicographic_better(base_metrics, repaired_metrics)
        accepted = repaired_metrics.hard_feasible and no_duplicates and preserved and lexicographic_better
        final_schedule = candidate_schedule if accepted else base_schedule
        repaired_incomplete_ids = sorted(set(instance.customer_ids()) - final_schedule.delivered_customer_ids())
        rescued_customer_ids = sorted(set(incomplete_ids) - set(repaired_incomplete_ids))

        status.update(
            {
                "repair_accepted": accepted,
                "repair_reason": "accepted" if accepted else "candidate_not_lexicographically_better_or_infeasible",
                "repair_stage_r2_status": r2_status,
                "repair_stage_r2_time_sec": r2_time,
                "repair_stage_r3_status": r3_status,
                "repair_stage_r3_time_sec": r3_time,
                "repair_rescued_count": len(rescued_customer_ids),
                "repair_rescued_customer_ids": rescued_customer_ids,
                "repair_remaining_incomplete_ids": repaired_incomplete_ids,
                "repair_total_runtime_sec": time.perf_counter() - total_start,
                "repaired_incomplete_count": repaired_metrics.incomplete_count if accepted else base_metrics.incomplete_count,
                "repaired_total_deferral_days": repaired_metrics.total_deferral_days if accepted else base_metrics.total_deferral_days,
                "repaired_total_distance_km": repaired_metrics.total_distance_km if accepted else base_metrics.total_distance_km,
                "repair_hard_feasible": repaired_metrics.hard_feasible,
                "repair_no_duplicates": no_duplicates,
                "repair_preserved_base_deliveries": preserved,
            }
        )
        return self._with_status(final_schedule, status)

    def _base_status(self, base_schedule: WeeklySchedule, base_metrics: Any, incomplete_ids: list[str]) -> dict[str, Any]:
        return {
            "solver": self.name,
            "base_solver": "cp_rolling",
            "base_incomplete_count": base_metrics.incomplete_count,
            "base_total_deferral_days": base_metrics.total_deferral_days,
            "base_total_distance_km": base_metrics.total_distance_km,
            "repair_time_limit_sec": self.repair_time_limit_sec,
            "repair_max_days": self.repair_max_days,
            "repair_max_customers": self.repair_max_customers,
            "repair_num_workers": self.repair_num_workers,
            "repair_ran": False,
            "repair_reason": "",
            "repair_accepted": False,
            "repair_selected_days": [],
            "repair_customer_count": 0,
            "repair_original_incomplete_ids": incomplete_ids,
            "repair_stage_r1_status": "",
            "repair_stage_r2_status": "",
            "repair_stage_r3_status": "",
            "repair_rescued_count": 0,
            "repair_rescued_customer_ids": [],
            "repair_remaining_incomplete_ids": incomplete_ids,
            "repair_stage_r1_time_sec": 0.0,
            "repair_stage_r2_time_sec": 0.0,
            "repair_stage_r3_time_sec": 0.0,
            "repair_total_runtime_sec": 0.0,
            "repaired_incomplete_count": base_metrics.incomplete_count,
            "repaired_total_deferral_days": base_metrics.total_deferral_days,
            "repaired_total_distance_km": base_metrics.total_distance_km,
            "repair_hard_feasible": base_metrics.hard_feasible,
            "repair_no_duplicates": self._no_duplicates(base_schedule),
            "repair_preserved_base_deliveries": True,
            "repair_hint_y_count": 0,
            "repair_hint_x_count": 0,
            "repair_hint_g_count": 0,
            "repair_hint_time_count": 0,
        }

    @staticmethod
    def _with_status(schedule: WeeklySchedule, status: dict[str, Any]) -> WeeklySchedule:
        merged = dict(schedule.solver_status)
        merged.update(status)
        return WeeklySchedule(routes=schedule.routes, solver_status=merged)

    def _select_repair_days(self, instance: Instance, base_schedule: WeeklySchedule, incomplete_ids: list[str]) -> list[int]:
        scores: list[tuple[int, int, int]] = []
        for day in range(MONDAY, SUNDAY + 1):
            available = [customer_id for customer_id in incomplete_ids if day in instance.available_days(customer_id)]
            if not available:
                continue
            last_day_count = sum(1 for customer_id in available if max(instance.available_days(customer_id)) == day)
            base_count = len(base_schedule.routes.get(day, DailyRoute(day=day)).stops)
            score = 1000 * len(available) + 100 * last_day_count - base_count
            scores.append((score, day, day))
        selected = [day for _score, _tie, day in sorted(scores, key=lambda item: (-item[0], -item[1], item[2]))]
        return selected[: self.repair_max_days]

    @staticmethod
    def _repair_customers(
        instance: Instance,
        base_schedule: WeeklySchedule,
        selected_days: list[int],
        incomplete_ids: list[str],
    ) -> list[str]:
        repair_set: set[str] = set()
        for customer_id in incomplete_ids:
            if any(instance.windows_for_customer_day(customer_id, day) for day in selected_days):
                repair_set.add(customer_id)
        for day in selected_days:
            repair_set.update(base_schedule.routes.get(day, DailyRoute(day=day)).customer_sequence())
        return sorted(repair_set)

    def _build_repair_model(
        self,
        instance: Instance,
        selected_days: list[int],
        repair_customers: list[str],
        original_incomplete: set[str],
        repair_delivered_base: set[str],
        fixed_rescued_count: int | None = None,
        fixed_deferral_value: int | None = None,
    ) -> RepairModelData:
        model = cp_model.CpModel()
        depot_id = instance.depot_id
        windows: dict[tuple[str, int], list[TimeWindow]] = {}
        nodes_by_day: dict[int, list[str]] = {}
        y: dict[tuple[str, int], cp_model.IntVar] = {}
        g: dict[tuple[str, int, int], cp_model.IntVar] = {}
        t: dict[tuple[str, int], cp_model.IntVar] = {}
        x: dict[tuple[int, str, str], cp_model.IntVar] = {}
        z: dict[int, cp_model.IntVar] = {}
        departure: dict[int, cp_model.IntVar] = {}
        return_time: dict[int, cp_model.IntVar] = {}

        for day in selected_days:
            valid_customers = [customer_id for customer_id in repair_customers if instance.windows_for_customer_day(customer_id, day)]
            nodes = [depot_id, *valid_customers]
            nodes_by_day[day] = nodes
            z[day] = model.NewBoolVar(f"z[{day}]")
            departure[day] = model.NewIntVar(0, DAY_END_MIN, f"departure[{day}]")
            return_time[day] = model.NewIntVar(0, DAY_END_MIN, f"return[{day}]")
            model.Add(departure[day] <= return_time[day])
            node_index = {node: index for index, node in enumerate(nodes)}
            arcs: list[tuple[int, int, cp_model.IntVar]] = []
            for i in nodes:
                for j in nodes:
                    var = model.NewBoolVar(f"x[{day},{i},{j}]")
                    x[day, i, j] = var
                    arcs.append((node_index[i], node_index[j], var))
                    if i != j and not _can_follow(instance, day, i, j):
                        model.Add(var == 0)
            model.AddCircuit(arcs)
            model.Add(x[day, depot_id, depot_id] + z[day] == 1)

            for customer_id in valid_customers:
                key = (customer_id, day)
                day_windows = instance.windows_for_customer_day(customer_id, day)
                windows[key] = day_windows
                y[key] = model.NewBoolVar(f"y[{customer_id},{day}]")
                t[key] = model.NewIntVar(0, DAY_END_MIN, f"T[{customer_id},{day}]")
                model.Add(x[day, customer_id, customer_id] + y[key] == 1)
                model.Add(y[key] <= z[day])
                window_vars: list[cp_model.IntVar] = []
                service = _get_service_time(instance, customer_id)
                for window_index, window in enumerate(day_windows):
                    g_var = model.NewBoolVar(f"g[{customer_id},{day},{window_index}]")
                    g[customer_id, day, window_index] = g_var
                    window_vars.append(g_var)
                    model.Add(t[key] >= window.start_minute).OnlyEnforceIf(g_var)
                    model.Add(t[key] + service <= window.end_minute).OnlyEnforceIf(g_var)
                model.Add(sum(window_vars) == y[key])
                model.Add(t[key] >= departure[day] + _travel_time_minutes(instance, depot_id, customer_id)).OnlyEnforceIf(
                    x[day, depot_id, customer_id]
                )
                model.Add(
                    return_time[day]
                    >= t[key] + service + _travel_time_minutes(instance, customer_id, depot_id)
                ).OnlyEnforceIf(x[day, customer_id, depot_id])

            for i in valid_customers:
                for j in valid_customers:
                    if i == j:
                        continue
                    model.Add(
                        t[j, day]
                        >= t[i, day] + _get_service_time(instance, i) + _travel_time_minutes(instance, i, j)
                    ).OnlyEnforceIf(x[day, i, j])

            day_y_vars = [y[customer_id, day] for customer_id in valid_customers]
            if day_y_vars:
                model.Add(z[day] <= sum(day_y_vars))
            else:
                model.Add(z[day] == 0)

        for customer_id in repair_customers:
            vars_for_customer = [var for (raw_customer, _day), var in y.items() if raw_customer == customer_id]
            if not vars_for_customer:
                continue
            if customer_id in repair_delivered_base:
                model.Add(sum(vars_for_customer) == 1)
            elif customer_id in original_incomplete:
                model.Add(sum(vars_for_customer) <= 1)

        rescued_terms = [var for (customer_id, _day), var in y.items() if customer_id in original_incomplete]
        if fixed_rescued_count is not None:
            model.Add(sum(rescued_terms) == fixed_rescued_count)
        if fixed_deferral_value is not None:
            model.Add(
                sum(
                    (day - min(instance.available_days(customer_id))) * var
                    for (customer_id, day), var in y.items()
                    if instance.available_days(customer_id)
                )
                == fixed_deferral_value
            )

        return RepairModelData(
            model=model,
            selected_days=selected_days,
            repair_customers=repair_customers,
            original_incomplete=original_incomplete,
            y=y,
            g=g,
            t=t,
            x=x,
            z=z,
            departure=departure,
            return_time=return_time,
            windows=windows,
            nodes_by_day=nodes_by_day,
        )

    def _add_base_hints(self, data: RepairModelData, instance: Instance, base_schedule: WeeklySchedule) -> dict[str, int]:
        y_count = 0
        x_count = 0
        g_count = 0
        time_count = 0
        delivered_by_customer = {
            stop.customer_id: (day, stop)
            for day, route in base_schedule.routes.items()
            if day in data.selected_days
            for stop in route.stops
        }
        for (customer_id, day), var in data.y.items():
            if customer_id in delivered_by_customer:
                data.model.AddHint(var, 1 if delivered_by_customer[customer_id][0] == day else 0)
                y_count += 1
        for day in data.selected_days:
            route = base_schedule.routes.get(day, DailyRoute(day=day))
            sequence = route.customer_sequence()
            arcs = [(instance.depot_id, sequence[0])] if sequence else []
            arcs.extend((sequence[index], sequence[index + 1]) for index in range(len(sequence) - 1))
            if sequence:
                arcs.append((sequence[-1], instance.depot_id))
            arc_set = set(arcs)
            for key, var in data.x.items():
                raw_day, i, j = key
                if raw_day != day:
                    continue
                if i in data.nodes_by_day[day] and j in data.nodes_by_day[day]:
                    data.model.AddHint(var, 1 if (i, j) in arc_set or (i == j and i not in sequence) else 0)
                    x_count += 1
            if day in data.departure:
                data.model.AddHint(data.departure[day], route.depot_departure_time)
                data.model.AddHint(data.return_time[day], route.return_to_depot_time)
                time_count += 2
            for stop in route.stops:
                key = (stop.customer_id, day)
                if key in data.t:
                    data.model.AddHint(data.t[key], stop.service_start_time)
                    time_count += 1
                if stop.selected_time_window is None:
                    continue
                for window_index, window in enumerate(data.windows.get(key, [])):
                    g_key = (stop.customer_id, day, window_index)
                    if g_key in data.g:
                        data.model.AddHint(data.g[g_key], 1 if window == stop.selected_time_window else 0)
                        g_count += 1
        return {
            "repair_hint_y_count": y_count,
            "repair_hint_x_count": x_count,
            "repair_hint_g_count": g_count,
            "repair_hint_time_count": time_count,
        }

    @staticmethod
    def _hint_from_solution(target: RepairModelData, source: RepairModelData, solver: cp_model.CpSolver) -> None:
        for key, var in target.y.items():
            if key in source.y:
                target.model.AddHint(var, int(solver.BooleanValue(source.y[key])))
        for key, var in target.x.items():
            if key in source.x:
                target.model.AddHint(var, int(solver.BooleanValue(source.x[key])))
        for key, var in target.g.items():
            if key in source.g:
                target.model.AddHint(var, int(solver.BooleanValue(source.g[key])))
        for key, var in target.t.items():
            if key in source.t:
                target.model.AddHint(var, solver.Value(source.t[key]))
        for day, var in target.departure.items():
            if day in source.departure:
                target.model.AddHint(var, solver.Value(source.departure[day]))
                target.model.AddHint(target.return_time[day], solver.Value(source.return_time[day]))

    def _add_decision_strategy(self, data: RepairModelData) -> None:
        if not self.repair_use_decision_strategy:
            return
        ordered = [data.y[key] for key in sorted(data.y)]
        if ordered:
            data.model.AddDecisionStrategy(ordered, cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE)

    def _solve(self, model: cp_model.CpModel, time_limit_sec: float) -> cp_model.CpSolver:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(0.001, time_limit_sec)
        solver.parameters.num_search_workers = self.repair_num_workers
        solver.parameters.random_seed = self.repair_random_seed
        status = solver.Solve(model)
        setattr(solver, "_vrp_status", status)
        return solver

    @staticmethod
    def _has_solution(solver: cp_model.CpSolver) -> bool:
        return RollingHorizonCPRepairSolver._status_name(solver) in {"OPTIMAL", "FEASIBLE"}

    @staticmethod
    def _status_name(solver: cp_model.CpSolver) -> str:
        return solver.StatusName(getattr(solver, "_vrp_status"))

    def _extract_day_route(
        self,
        instance: Instance,
        data: RepairModelData,
        solver: cp_model.CpSolver,
        day: int,
    ) -> DailyRoute:
        selected_arcs = {
            (i, j)
            for (raw_day, i, j), var in data.x.items()
            if raw_day == day and solver.BooleanValue(var)
        }
        customers = [node for node in data.nodes_by_day[day] if node != instance.depot_id]
        route_sequence = _extract_route_from_arcs(selected_arcs, instance.depot_id, customers)
        service_start_times = {
            customer_id: solver.Value(data.t[customer_id, day])
            for customer_id in route_sequence
            if (customer_id, day) in data.t
        }
        selected_windows: dict[str, TimeWindow] = {}
        for customer_id in route_sequence:
            for window_index, window in enumerate(data.windows.get((customer_id, day), [])):
                if solver.BooleanValue(data.g[customer_id, day, window_index]):
                    selected_windows[customer_id] = window
                    break
        return _build_daily_schedule_from_solution(
            instance,
            day,
            route_sequence,
            service_start_times,
            selected_windows,
            solver.Value(data.departure[day]),
        )

    @staticmethod
    def _no_duplicates(schedule: WeeklySchedule) -> bool:
        delivered: list[str] = []
        for route in schedule.routes.values():
            delivered.extend(route.customer_sequence())
        return len(delivered) == len(set(delivered))

    @staticmethod
    def _lexicographic_better(base_metrics: Any, repaired_metrics: Any) -> bool:
        base_tuple = (
            base_metrics.incomplete_count,
            base_metrics.total_deferral_days,
            round(base_metrics.total_distance_km, 6),
        )
        repaired_tuple = (
            repaired_metrics.incomplete_count,
            repaired_metrics.total_deferral_days,
            round(repaired_metrics.total_distance_km, 6),
        )
        return repaired_tuple < base_tuple
