from __future__ import annotations

from vrp_weekly.core import Instance, Location, TimeWindow
from vrp_weekly.evaluator import evaluate_weekly_schedule
from vrp_weekly.model_factory import create_solver
from vrp_weekly.models.hybrid_genetic_vns import Chromosome, HybridGeneticVNSSolver


def make_instance(
    coords: dict[str, tuple[float, float]],
    windows: dict[str, list[tuple[int, int, int]]],
    service_time: int = 5,
) -> Instance:
    locations = {"DEPOT": Location("DEPOT", "Depot", 0.0, 0.0, service_time=0, is_depot=True)}
    for customer_id, (x_km, y_km) in coords.items():
        locations[customer_id] = Location(customer_id, customer_id, x_km, y_km, service_time=service_time)
    grouped: dict[str, dict[int, list[TimeWindow]]] = {}
    for customer_id, raw_windows in windows.items():
        for day, start, end in raw_windows:
            grouped.setdefault(customer_id, {}).setdefault(day, []).append(TimeWindow(customer_id, day, start, end))
    return Instance(locations=locations, time_windows=grouped)


def small_instance() -> Instance:
    return make_instance({"A": (1, 0), "B": (2, 0), "C": (3, 0)}, {"A": [(1, 480, 900)], "B": [(2, 480, 900)], "C": [(1, 480, 900), (3, 480, 900)]})


def test_chromosome_decode_hard_feasible_small_instance() -> None:
    instance = small_instance()
    solver = HybridGeneticVNSSolver(population_size=4, generations=1, use_local_search=False)

    schedule = solver.decode_chromosome(instance, Chromosome({"A": 1, "B": 2, "C": 1}, ["A", "C", "B"]))

    assert evaluate_weekly_schedule(instance, schedule).hard_feasible


def test_day_gene_repair_moves_to_available_day() -> None:
    instance = small_instance()
    solver = HybridGeneticVNSSolver(population_size=4, generations=1)

    repaired = solver.repair_chromosome(instance, Chromosome({"B": 1}, ["B", "A", "C"]))

    assert repaired.day_gene["B"] == 2


def test_priority_gene_is_permutation_after_crossover() -> None:
    instance = small_instance()
    solver = HybridGeneticVNSSolver(population_size=4, generations=1)
    p1 = solver.repair_chromosome(instance, Chromosome({"A": 1, "B": 2, "C": 3}, ["A", "B", "C"]))
    p2 = solver.repair_chromosome(instance, Chromosome({"A": 1, "B": 2, "C": 3}, ["C", "B", "A"]))

    child = solver.crossover_priority_ox(p1, p2)

    assert sorted(child) == ["A", "B", "C"]


def test_mutation_preserves_permutation() -> None:
    instance = small_instance()
    solver = HybridGeneticVNSSolver(population_size=4, generations=1)
    chromosome = solver.repair_chromosome(instance, Chromosome({"A": 1, "B": 2, "C": 3}, ["A", "B", "C"]))

    mutated = solver.mutate_chromosome(instance, chromosome)

    assert sorted(mutated.priority_gene) == ["A", "B", "C"]


def test_hybrid_genetic_vns_returns_weekly_schedule() -> None:
    instance = small_instance()

    schedule = HybridGeneticVNSSolver(population_size=4, generations=1, time_limit_sec=2, use_local_search=False).solve(instance)

    assert evaluate_weekly_schedule(instance, schedule).hard_feasible


def test_no_duplicate_delivery() -> None:
    instance = small_instance()

    schedule = HybridGeneticVNSSolver(population_size=4, generations=1, time_limit_sec=2, use_local_search=False).solve(instance)

    assert sum("A" in route.delivered_customer_ids() for route in schedule.routes.values()) == 1


def test_seed_population_contains_multiple_sources() -> None:
    instance = small_instance()

    seeds = HybridGeneticVNSSolver(population_size=6, generations=1).create_initial_population(instance)

    assert len(seeds) == 6
    assert len({tuple(seed.priority_gene) for seed in seeds}) > 1


def test_hybrid_genetic_vns_uses_local_search_when_enabled() -> None:
    instance = small_instance()

    schedule = HybridGeneticVNSSolver(population_size=4, generations=1, time_limit_sec=2, use_local_search=True, local_search_time_limit_sec=1).solve(instance)

    assert schedule.solver_status["local_search_enabled"] is True


def test_model_factory_can_create_hybrid_genetic_vns() -> None:
    assert create_solver("hybrid_genetic_vns").name == "hybrid_genetic_vns"

