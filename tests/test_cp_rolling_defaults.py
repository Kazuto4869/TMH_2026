from __future__ import annotations

import inspect

from vrp_weekly.cli import build_parser
from vrp_weekly.model_factory import create_solver
from vrp_weekly.models.cp_rolling_horizon import RollingHorizonCPSATSolver
import main as interactive_main


def test_cp_rolling_default_mode_is_full_three_stage() -> None:
    assert RollingHorizonCPSATSolver().optimization_mode == "full_three_stage"


def test_cp_rolling_default_adaptive_deadline_true() -> None:
    assert RollingHorizonCPSATSolver().adaptive_daily_deadline is True


def test_stage2_default_fraction_point_one() -> None:
    assert RollingHorizonCPSATSolver().stage2_max_time_fraction == 0.10


def test_factory_default_mode_full_three_stage() -> None:
    solver = create_solver("cp_rolling")
    assert solver.optimization_mode == "full_three_stage"
    assert solver.adaptive_daily_deadline is True
    assert solver.stage2_max_time_fraction == 0.10
    assert solver.time_limit_per_day_sec == 60
    assert solver.max_candidates_per_day == 80
    assert solver.num_workers == 4


def test_cli_default_mode_full_three_stage() -> None:
    args = build_parser().parse_args(["--locations", "loc.csv", "--time-windows", "tw.csv", "--solver", "cp_rolling"])
    assert args.cp_optimization_mode == "full_three_stage"
    assert args.cp_adaptive_daily_deadline is True
    assert args.cp_stage2_max_time_fraction == 0.10
    assert args.cp_time_limit_per_day_sec == 60
    assert args.cp_max_candidates_per_day == 80
    assert args.cp_workers == 4


def test_main_default_mode_full_three_stage() -> None:
    source = inspect.getsource(interactive_main.main)
    assert 'cp_optimization_mode = "full_three_stage"' in source
    assert "cp_adaptive_daily_deadline = True" in source
    assert "cp_stage2_max_time_fraction = 0.10" in source


def test_legacy_phase1_only_maps_to_service_phases_only() -> None:
    args = build_parser().parse_args(
        ["--locations", "loc.csv", "--time-windows", "tw.csv", "--solver", "cp_rolling", "--cp-phase1-only"]
    )
    assert args.cp_optimization_mode == "service_phases_only"
