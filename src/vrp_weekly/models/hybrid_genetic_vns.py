"""Hybrid genetic heuristic with VNS-style local repair."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from vrp_weekly.config import MONDAY, SUNDAY
from vrp_weekly.core import DailyRoute, Instance, WeeklySchedule
from vrp_weekly.evaluator import evaluate_daily_route, evaluate_weekly_schedule
from vrp_weekly.heuristics.local_search import LocalSearchParams, improve_daily_route
from vrp_weekly.heuristics.route_eval import (
    HeuristicWeights,
    best_feasible_insertion,
    distance_km,
    validate_no_duplicates,
    weekly_score,
    windows_for,
)
from vrp_weekly.heuristics.scoring import (
    available_days,
    deadline_pressure,
    earliest_available_day,
    earliest_window_end_today,
    is_last_available_day,
    remaining_available_days,
    spatial_isolation,
    total_window_width_today,
    window_width_loss_to_future,
)


@dataclass
class Chromosome:
    """Genetic representation: planned day plus global priority order."""

    day_gene: dict[str, int]
    priority_gene: list[str]


@dataclass
class Individual:
    """Decoded chromosome with a fitness value."""

    chromosome: Chromosome
    schedule: WeeklySchedule | None
    fitness: float


class HybridGeneticVNSSolver:
    """Population-based heuristic with insertion repair and local search."""

    name = "hybrid_genetic_vns"

    def __init__(
        self,
        population_size: int = 30,
        generations: int = 50,
        elite_size: int = 5,
        mutation_rate: float = 0.10,
        crossover_rate: float = 0.80,
        time_limit_sec: int | None = 120,
        random_seed: int = 1,
        use_local_search: bool = True,
        local_search_time_limit_sec: int = 10,
        local_search_max_iterations: int = 100,
        max_candidates_per_day: int | None = None,
        distance_weight: float = 10.0,
        waiting_weight: float = 1.0,
        duration_weight: float = 1.0,
    ) -> None:
        """Initialize GA/VNS parameters."""
        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.time_limit_sec = time_limit_sec
        self.random_seed = random_seed
        self.use_local_search = use_local_search
        self.local_search_time_limit_sec = local_search_time_limit_sec
        self.local_search_max_iterations = local_search_max_iterations
        self.max_candidates_per_day = max_candidates_per_day
        self.distance_weight = distance_weight
        self.waiting_weight = waiting_weight
        self.duration_weight = duration_weight
        self._rng = random.Random(random_seed)

    def solve(self, instance: Instance) -> WeeklySchedule:
        """Run the genetic/VNS heuristic and return the best decoded schedule."""
        start = time.perf_counter()
        deadline = None if self.time_limit_sec is None else start + max(0, self.time_limit_sec)
        seed_chromosomes = self.create_initial_population(instance)
        population: list[Individual] = []
        for chromosome in seed_chromosomes:
            if population and deadline is not None and time.perf_counter() >= deadline:
                break
            population.append(self._evaluate_chromosome(instance, chromosome, apply_local_search=False))
        population.sort(key=lambda individual: individual.fitness)
        best = population[0]
        best_generation = 0
        generations_completed = 0

        for generation in range(1, self.generations + 1):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            population.sort(key=lambda individual: individual.fitness)
            new_population = population[: max(1, min(self.elite_size, len(population)))]
            while len(new_population) < self.population_size:
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                if self._rng.random() < self.crossover_rate:
                    child = Chromosome(
                        day_gene=self.crossover_day_gene(parent1.chromosome, parent2.chromosome),
                        priority_gene=self.crossover_priority_ox(parent1.chromosome, parent2.chromosome),
                    )
                else:
                    better = parent1 if parent1.fitness <= parent2.fitness else parent2
                    child = Chromosome(dict(better.chromosome.day_gene), list(better.chromosome.priority_gene))
                if self._rng.random() < self.mutation_rate:
                    child = self.mutate_chromosome(instance, child)
                new_population.append(self._evaluate_chromosome(instance, child, apply_local_search=False))
            population = new_population
            generations_completed = generation
            candidate_best = min(population, key=lambda individual: individual.fitness)
            if candidate_best.fitness < best.fitness:
                best = candidate_best
                best_generation = generation

        schedule = self.decode_chromosome(instance, best.chromosome, apply_local_search=self.use_local_search)
        metrics = evaluate_weekly_schedule(instance, schedule)
        status = {
            "solver": self.name,
            "status": "HEURISTIC_FEASIBLE" if metrics.hard_feasible else "HEURISTIC_INFEASIBLE",
            "gap_percent": "",
            "population_size": self.population_size,
            "generations_completed": generations_completed,
            "elite_size": self.elite_size,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "best_score": best.fitness,
            "best_generation": best_generation,
            "seed_count": len(seed_chromosomes),
            "local_search_enabled": self.use_local_search,
            "runtime_sec": time.perf_counter() - start,
            "delivered_count": metrics.delivered_count,
            "incomplete_count": metrics.incomplete_count,
            "total_deferral_days": metrics.total_deferral_days,
            "total_distance_km": metrics.total_distance_km,
            "total_waiting_time_min": metrics.total_waiting_time_min,
            "total_route_duration_min": metrics.total_route_duration_min,
            "hard_feasible": metrics.hard_feasible,
            "no_duplicate_delivery": validate_no_duplicates(schedule),
        }
        return WeeklySchedule(routes=schedule.routes, solver_status=status)

    def create_initial_population(self, instance: Instance) -> list[Chromosome]:
        """Create a mixed deterministic/random seed population."""
        customers = instance.customer_ids()
        seeds = [
            self._chromosome_from_priority(instance, sorted(customers, key=lambda c: (distance_km(instance, instance.depot_id, c), c))),
            self._chromosome_from_priority(instance, sorted(customers, key=lambda c: (_global_earliest_deadline(instance, c), c))),
            self._chromosome_from_priority(instance, sorted(customers, key=lambda c: (earliest_available_day(instance, c) or SUNDAY, c))),
            self._chromosome_from_priority(instance, sorted(customers, key=lambda c: (-_inferior_seed_score(instance, MONDAY, c, customers), c))),
            self._chromosome_from_priority(instance, sorted(customers, key=lambda c: (-_defer_seed_score(instance, MONDAY, c), c))),
        ]
        while len(seeds) < max(1, self.population_size):
            priority = list(customers)
            self._rng.shuffle(priority)
            day_gene = {}
            for customer in customers:
                days = available_days(instance, customer)
                day_gene[customer] = self._rng.choice(days) if days else SUNDAY
            seeds.append(Chromosome(day_gene=day_gene, priority_gene=priority))
        return [self.repair_chromosome(instance, chromosome) for chromosome in seeds[: self.population_size]]

    def chromosome_from_schedule(self, instance: Instance, schedule: WeeklySchedule) -> Chromosome:
        """Convert a weekly schedule into a chromosome."""
        priority: list[str] = []
        day_gene: dict[str, int] = {}
        for day, route in sorted(schedule.routes.items()):
            for stop in route.stops:
                if stop.customer_id not in priority:
                    priority.append(stop.customer_id)
                    day_gene[stop.customer_id] = day
        for customer in instance.customer_ids():
            if customer not in priority:
                priority.append(customer)
            day_gene.setdefault(customer, earliest_available_day(instance, customer) or SUNDAY)
        return self.repair_chromosome(instance, Chromosome(day_gene=day_gene, priority_gene=priority))

    def repair_chromosome(self, instance: Instance, chromosome: Chromosome) -> Chromosome:
        """Repair invalid day genes and priority permutations."""
        customers = instance.customer_ids()
        seen: set[str] = set()
        priority = []
        for customer in chromosome.priority_gene:
            if customer in instance.locations and customer in customers and customer not in seen:
                priority.append(customer)
                seen.add(customer)
        priority.extend(customer for customer in customers if customer not in seen)

        day_gene: dict[str, int] = {}
        for customer in customers:
            days = available_days(instance, customer)
            planned = chromosome.day_gene.get(customer)
            if not days:
                day_gene[customer] = SUNDAY
            elif planned in days:
                day_gene[customer] = int(planned)
            else:
                day_gene[customer] = min(days, key=lambda day: (abs(day - int(planned or days[0])), day))
        return Chromosome(day_gene=day_gene, priority_gene=priority)

    def decode_chromosome(
        self,
        instance: Instance,
        chromosome: Chromosome,
        apply_local_search: bool | None = None,
    ) -> WeeklySchedule:
        """Decode a chromosome into a feasible weekly schedule via insertion repair."""
        if apply_local_search is None:
            apply_local_search = self.use_local_search
        chromosome = self.repair_chromosome(instance, chromosome)
        undelivered = set(instance.customer_ids())
        priority_index = {customer: index for index, customer in enumerate(chromosome.priority_gene)}
        routes: dict[int, DailyRoute] = {}
        weights = HeuristicWeights(self.distance_weight, self.waiting_weight, self.duration_weight)

        for day in range(MONDAY, SUNDAY + 1):
            planned_today = [customer for customer in chromosome.priority_gene if chromosome.day_gene.get(customer) == day and customer in undelivered]
            day_limit = self._effective_day_limit()
            planned_today = planned_today[:day_limit]
            sequence: list[str] = []
            current_route = evaluate_daily_route(instance, day, sequence)
            for customer in planned_today:
                if not windows_for(instance, customer, day):
                    continue
                insertion = best_feasible_insertion(instance, day, sequence, customer, base_route=current_route, weights=weights)
                if insertion is not None:
                    sequence = insertion.sequence
                    current_route = insertion.route
                    undelivered.remove(customer)

            post_fill = sorted(
                [customer for customer in undelivered if windows_for(instance, customer, day)],
                key=lambda customer: (priority_index.get(customer, 10**9), -_defer_seed_score(instance, day, customer), customer),
            )
            post_fill = post_fill[:day_limit]
            for customer in post_fill:
                insertion = best_feasible_insertion(instance, day, sequence, customer, base_route=current_route, weights=weights)
                if insertion is not None:
                    sequence = insertion.sequence
                    current_route = insertion.route
                    undelivered.remove(customer)

            if apply_local_search:
                current_route = improve_daily_route(
                    instance,
                    day,
                    current_route,
                    undelivered_today=[customer for customer in undelivered if windows_for(instance, customer, day)],
                    params=LocalSearchParams(
                        max_iterations=self.local_search_max_iterations,
                        time_limit_sec=self.local_search_time_limit_sec,
                        distance_weight=self.distance_weight,
                        waiting_weight=self.waiting_weight,
                        duration_weight=self.duration_weight,
                    ),
                )
                undelivered -= set(current_route.delivered_customer_ids())
            routes[day] = current_route

        return WeeklySchedule(routes=routes)

    def evaluate_fitness(self, instance: Instance, schedule: WeeklySchedule) -> float:
        """Return lower-is-better fitness."""
        return weekly_score(instance, schedule)

    def crossover_day_gene(self, parent1: Chromosome, parent2: Chromosome) -> dict[str, int]:
        """Uniform crossover for day genes."""
        return {
            customer: parent1.day_gene.get(customer, parent2.day_gene.get(customer, SUNDAY))
            if self._rng.random() < 0.5
            else parent2.day_gene.get(customer, parent1.day_gene.get(customer, SUNDAY))
            for customer in parent1.priority_gene
        }

    def crossover_priority_ox(self, parent1: Chromosome, parent2: Chromosome) -> list[str]:
        """Order crossover for priority permutations."""
        n = len(parent1.priority_gene)
        if n <= 2:
            return list(parent1.priority_gene)
        a, b = sorted(self._rng.sample(range(n), 2))
        child: list[str | None] = [None] * n
        child[a:b] = parent1.priority_gene[a:b]
        used = set(parent1.priority_gene[a:b])
        fill_values = [customer for customer in parent2.priority_gene if customer not in used]
        fill_iter = iter(fill_values)
        for index in list(range(0, a)) + list(range(b, n)):
            child[index] = next(fill_iter)
        return [customer for customer in child if customer is not None]

    def mutate_chromosome(self, instance: Instance, chromosome: Chromosome) -> Chromosome:
        """Mutate a chromosome while preserving the priority permutation."""
        priority = list(chromosome.priority_gene)
        day_gene = dict(chromosome.day_gene)
        if len(priority) >= 2:
            move = self._rng.choice(["swap", "relocate", "day", "shuffle"])
            if move == "swap":
                i, j = self._rng.sample(range(len(priority)), 2)
                priority[i], priority[j] = priority[j], priority[i]
            elif move == "relocate":
                i, j = self._rng.sample(range(len(priority)), 2)
                customer = priority.pop(i)
                priority.insert(j, customer)
            elif move == "shuffle":
                i = self._rng.randrange(len(priority))
                j = min(len(priority), i + self._rng.randint(2, min(5, len(priority))))
                segment = priority[i:j]
                self._rng.shuffle(segment)
                priority[i:j] = segment
            else:
                customer = self._rng.choice(priority)
                days = available_days(instance, customer)
                if days:
                    day_gene[customer] = self._rng.choice(days)
        return self.repair_chromosome(instance, Chromosome(day_gene=day_gene, priority_gene=priority))

    def tournament_selection(self, population: list[Individual], k: int = 3) -> Individual:
        """Select the best individual among k random candidates."""
        sample_size = min(k, len(population))
        contenders = self._rng.sample(population, sample_size)
        return min(contenders, key=lambda individual: individual.fitness)

    def _evaluate_chromosome(self, instance: Instance, chromosome: Chromosome, apply_local_search: bool) -> Individual:
        repaired = self.repair_chromosome(instance, chromosome)
        schedule = self.decode_chromosome(instance, repaired, apply_local_search=apply_local_search)
        return Individual(chromosome=repaired, schedule=schedule, fitness=self.evaluate_fitness(instance, schedule))

    def _chromosome_from_priority(self, instance: Instance, priority: list[str]) -> Chromosome:
        day_gene = {customer: earliest_available_day(instance, customer) or SUNDAY for customer in instance.customer_ids()}
        return Chromosome(day_gene=day_gene, priority_gene=priority)

    def _effective_day_limit(self) -> int:
        """Return a practical daily decode pool limit for GA repair."""
        return self.max_candidates_per_day if self.max_candidates_per_day is not None else 80


def _global_earliest_deadline(instance: Instance, customer: str) -> int:
    """Return earliest window end across all available days."""
    return min((window.end_minute for window in instance.windows_for_customer(customer)), default=10**9)


def _inferior_seed_score(instance: Instance, day: int, customer: str, candidates: list[str]) -> float:
    """Internal inferior-style priority score for GA seeds."""
    remaining = remaining_available_days(instance, customer, day)
    return (
        1000 * int(is_last_available_day(instance, customer, day))
        + 200 / max(1, len(remaining))
        + 500 / max(1, total_window_width_today(instance, customer, day))
        + 100 * deadline_pressure(instance, customer, day)
        + 20 * spatial_isolation(instance, customer, candidates)
    )


def _defer_seed_score(instance: Instance, day: int, customer: str) -> float:
    """Internal regret/defer-style priority score for GA seeds."""
    remaining = remaining_available_days(instance, customer, day)
    return (
        1000 * int(is_last_available_day(instance, customer, day))
        + 300 / max(1, len(remaining))
        + 200 * window_width_loss_to_future(instance, customer, day)
        + 100 * deadline_pressure(instance, customer, day)
        + max(0, SUNDAY - (earliest_available_day(instance, customer) or SUNDAY))
    )
