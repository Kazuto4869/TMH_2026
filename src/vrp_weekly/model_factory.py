"""Factory for available routing models."""

from __future__ import annotations

from typing import Any

from vrp_weekly.config import (
    DROP_PENALTY_BY_DAY,
    WEIGHT_DEFERRAL,
    WEIGHT_DISTANCE_KM,
    WEIGHT_INCOMPLETE,
)
from vrp_weekly.models import (
    EarliestDeadlineSolver,
    InferiorInsertionSolver,
    MinDeferralSolver,
    NearestNeighborSolver,
    RegretLSInsertionSolver,
    RollingHorizonCPSATSolver,
)


def create_solver(solver_key: str, **kwargs: Any) -> object:
    """Create a model instance from a CLI or benchmark key."""
    normalized = "deadline" if solver_key == "earliest" else solver_key
    if normalized not in solver_names():
        raise ValueError(f"Unsupported solver after cleanup: {solver_key}")
    if normalized == "nearest":
        return NearestNeighborSolver()
    if normalized == "deadline":
        return EarliestDeadlineSolver()
    if normalized == "min_deferral":
        return MinDeferralSolver(
            distance_weight=float(kwargs.get("distance_weight", 10.0)),
            waiting_weight=float(kwargs.get("waiting_weight", 0.2)),
            duration_weight=float(kwargs.get("duration_weight", 0.0)),
        )
    if normalized in {"inferior_insertion", "inferior_insertion_ls"}:
        solver = InferiorInsertionSolver(
            use_local_search=normalized == "inferior_insertion_ls" or bool(kwargs.get("heuristic_use_local_search", False)),
            local_search_time_limit_sec=int(kwargs.get("local_search_time_limit_sec", 10)),
            local_search_max_iterations=int(kwargs.get("local_search_max_iterations", 100)),
            max_candidates_per_day=kwargs.get("heuristic_max_candidates_per_day"),
            distance_weight=float(kwargs.get("distance_weight", 10.0)),
            waiting_weight=float(kwargs.get("waiting_weight", 1.0)),
            duration_weight=0.0,
            random_seed=int(kwargs.get("heuristic_random_seed", kwargs.get("seed", 1))),
        )
        solver.name = normalized
        return solver
    if normalized in {"regret_dispatch", "regret_ls"}:
        if normalized == "regret_ls" or normalized == "regret_dispatch":
            solver = RegretLSInsertionSolver()
            solver.name = normalized
            return solver
    if normalized == "hybrid_genetic_vns":
        heuristic_use_local_search = kwargs.get("heuristic_use_local_search")
        return HybridGeneticVNSSolver(
            population_size=int(kwargs.get("ga_population_size", 30)),
            generations=int(kwargs.get("ga_generations", 50)),
            elite_size=int(kwargs.get("ga_elite_size", 5)),
            mutation_rate=float(kwargs.get("ga_mutation_rate", 0.10)),
            crossover_rate=float(kwargs.get("ga_crossover_rate", 0.80)),
            time_limit_sec=kwargs.get("ga_time_limit_sec", 120),
            random_seed=int(kwargs.get("heuristic_random_seed", kwargs.get("seed", 1))),
            use_local_search=True if heuristic_use_local_search is None else bool(heuristic_use_local_search),
            local_search_time_limit_sec=int(kwargs.get("local_search_time_limit_sec", 10)),
            local_search_max_iterations=int(kwargs.get("local_search_max_iterations", 100)),
            max_candidates_per_day=kwargs.get("heuristic_max_candidates_per_day"),
            distance_weight=float(kwargs.get("distance_weight", 10.0)),
            waiting_weight=float(kwargs.get("waiting_weight", 1.0)),
            duration_weight=0.0,
        )
    if normalized in ("cp", "cp_full_week"):
        return FullWeekCPSATSolver(
            time_limit_sec=int(kwargs.get("cp_time_limit_sec", 60)),
            max_customers=kwargs.get("cp_max_customers", 40),
            incomplete_weight=int(kwargs.get("incomplete_weight", WEIGHT_INCOMPLETE)),
            deferral_weight=int(kwargs.get("deferral_weight", WEIGHT_DEFERRAL)),
            distance_weight=int(kwargs.get("distance_weight", WEIGHT_DISTANCE_KM)),
            route_duration_weight=0,
            num_workers=int(kwargs.get("cp_workers", 8)),
            log_search_progress=bool(kwargs.get("cp_log_search", False)),
        )
    if normalized == "cp_rolling":
        return RollingHorizonCPSATSolver(
            time_limit_per_day_sec=int(kwargs.get("cp_time_limit_per_day_sec", 60)),
            max_candidates_per_day=kwargs.get("cp_max_candidates_per_day", 80),
            drop_penalty_by_day=kwargs.get("drop_penalty_by_day", DROP_PENALTY_BY_DAY),
            distance_weight=int(kwargs.get("distance_weight", 10)),
            route_duration_weight=0,
            urgency_weight=0,
            num_workers=int(kwargs.get("cp_workers", 4)),
            log_search_progress=bool(kwargs.get("cp_log_search", False)),
            use_two_phase_objective=bool(kwargs.get("cp_two_phase_objective", True)),
            phase1_time_limit_sec=kwargs.get("cp_phase1_time_limit_sec"),
            phase2_time_limit_sec=kwargs.get("cp_phase2_time_limit_sec"),
            phase1_time_fraction=float(kwargs.get("cp_phase1_time_fraction", 0.85)),
            phase2_time_fraction=float(kwargs.get("cp_phase2_time_fraction", 0.15)),
            random_seed=int(kwargs.get("cp_random_seed", kwargs.get("seed", 1))),
            use_decision_strategy=bool(kwargs.get("cp_use_decision_strategy", True)),
            use_service_no_overlap=bool(kwargs.get("cp_use_service_no_overlap", True)),
            use_route_interval_no_overlap=bool(kwargs.get("cp_use_route_interval_no_overlap", True)),
            use_window_pair_cuts=bool(kwargs.get("cp_use_window_pair_cuts", True)),
            use_precedence_cuts=bool(kwargs.get("cp_use_precedence_cuts", True)),
            use_pair_conflict_cuts=bool(kwargs.get("cp_use_pair_conflict_cuts", True)),
            use_depot_window_cuts=bool(kwargs.get("cp_use_depot_window_cuts", True)),
            use_dominated_window_cuts=bool(kwargs.get("cp_use_dominated_window_cuts", True)),
            candidate_strategy=str(kwargs.get("cp_candidate_strategy", "hybrid")),
            solve_phase2=bool(kwargs.get("cp_solve_phase2", True)),
            adaptive_daily_deadline=bool(kwargs.get("cp_adaptive_daily_deadline", True)),
            optimization_mode=kwargs.get("cp_optimization_mode", "full_three_stage"),
            stage2_max_time_fraction=float(kwargs.get("cp_stage2_max_time_fraction", 0.10)),
            run_incomplete_diagnostics=bool(kwargs.get("cp_run_incomplete_diagnostics", False)),
            incomplete_diagnostic_time_limit_sec=int(kwargs.get("cp_incomplete_diagnostic_time_limit_sec", 60)),
        )
    if normalized == "cp_rolling_repair":
        return RollingHorizonCPRepairSolver(
            time_limit_per_day_sec=int(kwargs.get("cp_time_limit_per_day_sec", 60)),
            max_candidates_per_day=kwargs.get("cp_max_candidates_per_day", 80),
            drop_penalty_by_day=kwargs.get("drop_penalty_by_day", DROP_PENALTY_BY_DAY),
            distance_weight=int(kwargs.get("distance_weight", 10)),
            route_duration_weight=0,
            urgency_weight=0,
            num_workers=int(kwargs.get("cp_workers", 4)),
            log_search_progress=bool(kwargs.get("cp_log_search", False)),
            use_two_phase_objective=bool(kwargs.get("cp_two_phase_objective", True)),
            phase1_time_limit_sec=kwargs.get("cp_phase1_time_limit_sec"),
            phase2_time_limit_sec=kwargs.get("cp_phase2_time_limit_sec"),
            phase1_time_fraction=float(kwargs.get("cp_phase1_time_fraction", 0.85)),
            phase2_time_fraction=float(kwargs.get("cp_phase2_time_fraction", 0.15)),
            random_seed=int(kwargs.get("cp_random_seed", kwargs.get("seed", 1))),
            use_decision_strategy=bool(kwargs.get("cp_use_decision_strategy", True)),
            use_service_no_overlap=bool(kwargs.get("cp_use_service_no_overlap", True)),
            use_route_interval_no_overlap=bool(kwargs.get("cp_use_route_interval_no_overlap", True)),
            use_window_pair_cuts=bool(kwargs.get("cp_use_window_pair_cuts", True)),
            use_precedence_cuts=bool(kwargs.get("cp_use_precedence_cuts", True)),
            use_pair_conflict_cuts=bool(kwargs.get("cp_use_pair_conflict_cuts", True)),
            use_depot_window_cuts=bool(kwargs.get("cp_use_depot_window_cuts", True)),
            use_dominated_window_cuts=bool(kwargs.get("cp_use_dominated_window_cuts", True)),
            candidate_strategy=str(kwargs.get("cp_candidate_strategy", "hybrid")),
            solve_phase2=bool(kwargs.get("cp_solve_phase2", True)),
            adaptive_daily_deadline=bool(kwargs.get("cp_adaptive_daily_deadline", True)),
            optimization_mode=kwargs.get("cp_optimization_mode", "full_three_stage"),
            stage2_max_time_fraction=float(kwargs.get("cp_stage2_max_time_fraction", 0.10)),
            run_incomplete_diagnostics=bool(kwargs.get("cp_run_incomplete_diagnostics", False)),
            incomplete_diagnostic_time_limit_sec=int(kwargs.get("cp_incomplete_diagnostic_time_limit_sec", 60)),
            repair_time_limit_sec=int(kwargs.get("cp_repair_time_limit_sec", 300)),
            repair_max_days=int(kwargs.get("cp_repair_max_days", 2)),
            repair_max_customers=int(kwargs.get("cp_repair_max_customers", 120)),
            repair_random_seed=int(kwargs.get("cp_repair_random_seed", 1)),
            repair_num_workers=int(kwargs.get("cp_repair_workers", kwargs.get("cp_repair_num_workers", 4))),
            repair_use_decision_strategy=bool(kwargs.get("cp_repair_use_decision_strategy", True)),
            repair_optimize_route_cost=bool(kwargs.get("cp_repair_optimize_route_cost", True)),
        )
    raise ValueError(f"Unknown solver: {solver_key}")


def solver_names() -> list[str]:
    """Return supported model names."""
    return [
        "nearest",
        "deadline",
        "min_deferral",
        "inferior_insertion",
        "regret_dispatch",
        "regret_ls",
        "cp_rolling",
    ]
