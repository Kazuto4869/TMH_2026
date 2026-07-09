from __future__ import annotations

from vrp_weekly.benchmark import run_benchmark
from vrp_weekly.cli import build_parser
from vrp_weekly.core import Instance, Location, TimeWindow
from vrp_weekly.evaluator import evaluate_weekly_schedule
from vrp_weekly.model_factory import create_solver, solver_names


def make_instance() -> Instance:
    locations = {
        "DEPOT": Location("DEPOT", "Depot", 0.0, 0.0, service_time=0, is_depot=True),
        "A": Location("A", "A", 1.0, 0.0, service_time=5),
        "B": Location("B", "B", 2.0, 0.0, service_time=5),
    }
    windows = {
        "A": {1: [TimeWindow("A", 1, 480, 900)]},
        "B": {2: [TimeWindow("B", 2, 480, 900)]},
    }
    return Instance(locations=locations, time_windows=windows)


def test_model_factory_creates_each_new_solver_and_keeps_old_solvers() -> None:
    for solver_name in [
        "nearest",
        "deadline",
        "min_deferral",
        "inferior_insertion",
        "inferior_insertion_ls",
        "regret_dispatch",
        "regret_dispatch_ls",
        "hybrid_genetic_vns",
        "cp_full_week",
        "cp_rolling",
    ]:
        assert create_solver(solver_name).name == solver_name


def test_ls_aliases_return_hard_feasible_schedule() -> None:
    instance = make_instance()
    for solver_name in ["inferior_insertion_ls", "regret_dispatch_ls"]:
        schedule = create_solver(solver_name, local_search_time_limit_sec=1).solve(instance)
        assert evaluate_weekly_schedule(instance, schedule).hard_feasible


def test_cli_parser_accepts_new_solvers_and_flags() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--locations",
            "data/locations.csv",
            "--time-windows",
            "data/time_windows.csv",
            "--solver",
            "hybrid_genetic_vns",
            "--heuristic-max-candidates-per-day",
            "10",
            "--use-local-search",
            "--local-search-time-limit-sec",
            "1",
            "--ga-population-size",
            "4",
            "--ga-generations",
            "1",
        ]
    )

    assert args.solver == "hybrid_genetic_vns"
    assert args.ga_population_size == 4


def test_benchmark_accepts_new_solver_list(tmp_path) -> None:
    frame = run_benchmark(
        make_instance(),
        ["inferior_insertion", "regret_dispatch"],
        results_dir=tmp_path,
        local_search_time_limit_sec=1,
    )

    assert {row["solver"] for row in frame.rows} == {"inferior_insertion", "regret_dispatch"}
    assert (tmp_path / "comparison" / "benchmark_summary.csv").exists()


def test_solver_names_include_new_models() -> None:
    for name in ["inferior_insertion", "inferior_insertion_ls", "regret_dispatch", "regret_dispatch_ls", "hybrid_genetic_vns"]:
        assert name in solver_names()

