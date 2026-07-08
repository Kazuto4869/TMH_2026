# VRP Weekly Modeling Framework

Python framework for experimenting with a weekly single-courier delivery routing problem.

## Problem Summary

- One courier, one vehicle, one central depot.
- Planning horizon is 7 days, Monday `1` to Sunday `7`.
- Every day the courier leaves the depot, serves a sequence of customers, then returns to the depot.
- Vehicle speed is capped at 50 km/h.
- Vehicle capacity is ignored in this first version because the input assumes capacity is large enough.
- Each customer has one order and one or more valid delivery time windows, possibly on different days.
- Orders not served on an earlier day may be deferred to a later available day in the same week.
- Orders still unserved after Sunday are counted as incomplete.

## Modeling Choices

- Coordinates are Cartesian kilometers.
- All times are integer minutes from `00:00`; `24:00` is `1440`.
- Travel time is computed as `ceil(60 * distance_km / 50)`.
- Service must start and finish inside a valid customer time window for that day.
- Waiting is allowed when the courier arrives before a time window opens.
- Depot departure is flexible, so waiting before the first customer is handled as a later departure instead of waiting outside from `00:00`.
- Default service time is 5 minutes if the input does not provide one.
- A route is hard feasible only if it returns to the depot by `1440` and has no duplicated weekly deliveries.

The canonical defaults are in `src/vrp_weekly/config.py`. A report-friendly copy is in `params/default_params.txt`.

## Project Layout

```text
data/                       Input CSV files
src/vrp_weekly/             Package source
src/vrp_weekly/models/      Baseline, heuristic, and CP routing models
tests/                      Unit tests
results/                    Generated benchmark and report outputs
```

## Source Files

Core package files:

- `src/vrp_weekly/core.py`: typed data structures used across the project, including `Location`, `TimeWindow`, `Instance`, `Stop`, `DailyRoute`, `WeeklySchedule`, and `EvaluationMetrics`.
- `src/vrp_weekly/config.py`: default constants for planning horizon, travel speed, objective weights, heuristic parameters, CP parameters, and observed data summary.
- `src/vrp_weekly/io.py`: reads `locations.csv` and `time_windows.csv`, validates input, detects the depot, groups time windows, and builds an `Instance`.
- `src/vrp_weekly/distance.py`: computes Euclidean distance and travel time in minutes.
- `src/vrp_weekly/time_utils.py`: parses and formats `HH:MM` time strings.
- `src/vrp_weekly/evaluator.py`: simulates daily routes, selects feasible time windows, validates weekly schedules, calculates metrics, and prints schedules/metrics.
- `src/vrp_weekly/export.py`: writes result JSON, daily schedule CSV, incomplete order CSV, benchmark CSV, and comparison plots.
- `src/vrp_weekly/model_factory.py`: maps model names such as `nearest`, `deadline`, `regret`, and `cp` to model classes.
- `src/vrp_weekly/cli.py`: command-line entry point for inspecting data or running one model.
- `src/vrp_weekly/benchmark.py`: runs multiple models and writes comparison outputs under `results/comparison/`.
- `src/vrp_weekly/compare_results.py`: compares already-saved result files under `results/schedules/{solver}/result.json` without rerunning models.
- `src/vrp_weekly/__init__.py`: public package exports for the core data structures.

Model files:

- `src/vrp_weekly/models/nearest.py`: nearest-neighbor greedy baseline.
- `src/vrp_weekly/models/deadline.py`: earliest-deadline greedy baseline.
- `src/vrp_weekly/models/regret.py`: rolling-horizon regret insertion model with local search.
- `src/vrp_weekly/models/cp.py`: daily OR-Tools CP routing model with deadline fallback if OR-Tools is unavailable.
- `src/vrp_weekly/models/__init__.py`: public imports for the four model classes.

## Models

- `nearest`: greedy nearest feasible next customer baseline.
- `deadline`: greedy earliest feasible time-window end baseline, with travel-time tie break.
- `regret`: rolling-horizon regret-2 insertion with urgency scoring and bounded local search.
- `cp`: rolling-horizon daily OR-Tools routing model. Each day is solved as a single-vehicle optional-customer model, then evaluated by the same weekly evaluator as the heuristics.

All models return a `WeeklySchedule`; metrics and feasibility are computed centrally in `evaluator.py`.

## Metrics

Benchmark comparison is sorted by:

1. `incomplete_count` ascending.
2. `total_deferral_days` ascending.
3. `total_distance_km` ascending.

Deferral is measured as `delivered_day - earliest_available_day`, so customers that cannot receive delivery on Monday are not unfairly penalized. The reported objective is:

```text
1_000_000 * incomplete_count
+ 10_000 * total_deferral_days
+ 10 * total_distance_km
+ total_waiting_time_min
+ 100 * active_days
```

This objective strongly prioritizes finishing orders, then serving earlier in the week, then reducing distance and waiting.

## Setup

From this directory:

```bash
python -m pip install -e ".[dev,benchmark]"
```

For the OR-Tools CP model:

```bash
python -m pip install ortools
```

The benchmark code writes CSV directly and does not import pandas at runtime, which avoids a native `pandas`/`ortools` import conflict observed in the current Python environment.

## Inspect Data

```bash
python -m vrp_weekly.cli \
  --locations data/locations.csv \
  --time-windows data/time_windows.csv \
  --summary
```

## Interactive Run

Run the terminal menu:

```bash
python main.py
```

The menu lets you choose one model, multiple models, or all models. It also asks for the CP time limit per day, CP thread count, and whether to show the OR-Tools CP run log in the terminal.

Results are always written under `results/`. Each model run writes:

- `results/schedules/{solver}/result.json`
- `results/schedules/{solver}/daily_schedule.csv`
- `results/schedules/{solver}/incomplete_orders.csv`
- `results/schedules/{solver}/run_log_{solver}_{timestamp}.csv`

## Run One Model

```bash
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver nearest
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver deadline
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver regret
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver cp --cp-time-limit-per-day 10 --cp-threads 4 --cp-log-search
```

Add `--save-results` to write:

- `results/schedules/{solver}/result.json`
- `results/schedules/{solver}/daily_schedule.csv`
- `results/schedules/{solver}/incomplete_orders.csv`

## Benchmark

Run the main comparison:

```bash
python -m vrp_weekly.benchmark \
  --locations data/locations.csv \
  --time-windows data/time_windows.csv \
  --solvers nearest deadline regret cp \
  --cp-time-limit-per-day 10
```

Run the report export:

```bash
python -m vrp_weekly.benchmark \
  --locations data/locations.csv \
  --time-windows data/time_windows.csv \
  --solvers nearest deadline regret cp \
  --cp-time-limit-per-day 10 \
  --export-report
```

Outputs:

- `results/comparison/benchmark_summary.csv`
- `results/schedules/{solver}/result.json`
- `results/schedules/{solver}/daily_schedule.csv`
- `results/schedules/{solver}/incomplete_orders.csv`
- `results/comparison/delivered_count_by_solver.png`
- `results/comparison/incomplete_count_by_solver.png`
- `results/comparison/total_distance_km_by_solver.png`
- `results/comparison/total_waiting_time_min_by_solver.png`

## Compare Saved Results

If model results already exist under `results/schedules/{solver}/result.json`, compare them without rerunning models:

```bash
python -m vrp_weekly.compare_results --results-dir results
```

Add `--export-plots` to regenerate the comparison PNG files.

## Tests

```bash
python -m pytest
```
