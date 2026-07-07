"""Tests for benchmark utilities."""

from __future__ import annotations

from pathlib import Path

from vrp_weekly.benchmark import run_benchmark
from vrp_weekly.evaluator import calculate_objective
from vrp_weekly.models import Instance, Location, TimeWindow


def test_calculate_objective_weights_priority_terms() -> None:
    """Objective should match the configured weighted formula."""
    assert calculate_objective(1, 2, 3.5, 4) == 1_020_039.0


def test_run_benchmark_writes_summary_and_schedule_json(tmp_path: Path) -> None:
    """Benchmark should produce a table and output files."""
    instance = Instance(
        locations={
            "DEPOT": Location("DEPOT", "Kho", 0, 0, 0, 0, True),
            "C001": Location("C001", "Customer", 1, 0, 1, 5),
        },
        time_windows={"C001": {1: [TimeWindow("C001", 1, 0, 1440)]}},
        depot_id="DEPOT",
    )

    frame = run_benchmark(instance, ["nearest"], results_dir=tmp_path)

    assert frame.loc[0, "solver"] == "nearest"
    assert (tmp_path / "benchmark_summary.csv").exists()
    assert (tmp_path / "schedules" / "nearest.json").exists()
