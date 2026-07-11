"""Default configuration constants for the weekly VRP problem."""

from __future__ import annotations

# =========================
# General planning horizon
# =========================

NUM_DAYS = 7
DAYS_IN_WEEK = NUM_DAYS
MONDAY = 1
SUNDAY = 7

DAY_START_MIN = 0
DAY_END_MIN = 24 * 60
MINUTES_PER_HOUR = 60

ALLOW_EVENING_DELIVERY = True

# =========================
# Vehicle / travel settings
# =========================

MAX_SPEED_KMPH = 50.0
DISTANCE_UNIT = "km"
TRAVEL_TIME_ROUNDING = "ceil"

NUM_VEHICLES = 1
USE_CAPACITY = False
VEHICLE_CAPACITY_KG = float("inf")

# =========================
# Service time settings
# =========================

USE_SERVICE_TIME_FROM_DATA = True
DEFAULT_SERVICE_TIME_MIN = 5
DEPOT_SERVICE_TIME_MIN = 0

# =========================
# Depot settings
# =========================

DEPOT_ID = "DEPOT"
FALLBACK_FIRST_ROW_AS_DEPOT = True

# =========================
# Time window feasibility
# =========================

ALLOW_MULTIPLE_WINDOWS_PER_DAY = True
ALLOW_WAITING = True
REQUIRE_SERVICE_END_WITHIN_WINDOW = True

# =========================
# Daily route settings
# =========================

REQUIRE_RETURN_TO_DEPOT_EACH_DAY = True
REQUIRE_RETURN_BEFORE_DAY_END = True
FLEXIBLE_DEPOT_DEPARTURE = True

# =========================
# Evaluation priority
# =========================

OBJECTIVE_VERSION = "weighted_1000_100_10_1"

WEIGHT_INCOMPLETE = 1_000
WEIGHT_DEFERRAL = 100
WEIGHT_DISTANCE_KM = 10
WEIGHT_WAITING_MIN = 1
# Reporting-only metric; not part of the official objective.
WEIGHT_ACTIVE_DAY = 0
# Reporting-only metric; not part of the official objective.
WEIGHT_ROUTE_DURATION_MIN = 0

METRIC_COLUMNS = [
    "solver",
    "delivered_count",
    "incomplete_count",
    "total_deferral_days",
    "total_distance_km",
    "total_waiting_time_min",
    "total_route_duration_min",
    "active_days",
    "objective_value",
    "runtime_sec",
    "hard_feasible",
    "total_travel_time_min",
    "total_service_time_min",
    "max_day_gap_percent",
    "total_fixed_impossible_arcs",
    "average_fixed_arc_ratio",
    "total_route_interval_count",
    "route_no_overlap_days",
    "total_remaining_after_week",
]

SORT_BY = [
    "objective_value",
    "incomplete_count",
    "total_deferral_days",
    "total_distance_km",
    "total_waiting_time_min",
    "runtime_sec",
]

# =========================
# Min-deferral baseline
# =========================

WAITING_WEIGHT = 0.2

ENABLE_LOCAL_SEARCH = True
LOCAL_SEARCH_OPERATORS = ["relocate", "swap", "two_opt"]
MAX_LOCAL_SEARCH_ITERATIONS = 1_000
FIRST_IMPROVEMENT = True

# =========================
# CP daily solver settings
# =========================

CP_TIME_LIMIT_PER_DAY_SEC = 10
CP_FIRST_SOLUTION_STRATEGY = "PATH_CHEAPEST_ARC"
CP_LOCAL_SEARCH_METAHEURISTIC = "GUIDED_LOCAL_SEARCH"
ALLOW_DROPPING_CUSTOMERS_DAILY = True
DROP_PENALTY_BY_DAY = {
    1: 10_000,
    2: 15_000,
    3: 25_000,
    4: 40_000,
    5: 70_000,
    6: 120_000,
    7: 1_000_000,
}

# =========================
# Observed data summary
# =========================

N_LOCATIONS = 301
N_CUSTOMERS = 300
N_TIME_WINDOWS = 1359

SERVICE_TIME_VALUES = [5, 7, 10]
TOTAL_DEMAND_KG = 818.7

X_RANGE_KM = (-22.9187, 20.0479)
Y_RANGE_KM = (-21.1603, 21.8201)

TIME_WINDOW_START_RANGE_MIN = (420, 1110)
TIME_WINDOW_END_RANGE_MIN = (690, 1290)

MIN_WINDOW_DURATION_MIN = 180
MAX_WINDOW_DURATION_MIN = 780
AVG_WINDOW_DURATION_MIN = 320.95

MIN_AVAILABLE_DAYS_PER_CUSTOMER = 2
AVG_AVAILABLE_DAYS_PER_CUSTOMER = 3.66
MAX_AVAILABLE_DAYS_PER_CUSTOMER = 6
