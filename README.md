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
src/vrp_weekly/solvers/     Baseline, heuristic, and CP solvers
tests/                      Unit tests
results/                    Generated benchmark and report outputs
```

## Solvers

- `nearest`: greedy nearest feasible next customer baseline.
- `deadline`: greedy earliest feasible time-window end baseline, with travel-time tie break.
- `regret`: rolling-horizon regret-2 insertion with urgency scoring and bounded local search.
- `cp`: rolling-horizon daily OR-Tools routing model. Each day is solved as a single-vehicle optional-customer model, then evaluated by the same weekly evaluator as the heuristics.

All solvers return a `WeeklySchedule`; metrics and feasibility are computed centrally in `evaluator.py`.

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

For the OR-Tools CP solver:

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

## Run One Solver

```bash
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver nearest
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver deadline
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver regret
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver cp --cp-time-limit-per-day 10
```

Add `--save-results` to write:

- `results/{solver}_result.json`
- `results/schedules/{solver}_daily_schedule.csv`
- `results/schedules/{solver}_incomplete_orders.csv`

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

- `results/benchmark_summary.csv`
- `results/schedules/{solver}.json`
- `results/schedules/{solver}_daily_schedule.csv`
- `results/schedules/{solver}_incomplete_orders.csv`
- `results/delivered_count_by_solver.png`
- `results/incomplete_count_by_solver.png`
- `results/total_distance_km_by_solver.png`
- `results/total_waiting_time_min_by_solver.png`

## Tests

```bash
python -m pytest
```
