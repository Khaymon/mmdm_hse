import argparse
from copy import deepcopy
import random
from tqdm import tqdm
import typing as T

import numpy as np
from scipy import stats

import crossovers
import flowers_data as data
import mutations
from solution import Solution


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--case", type=str, choices=['a', 'b'], required=True, help="Number of function to optimize")
    parser.add_argument("--n-populations", type=int, default=2, required=False, help="Number of populations")
    parser.add_argument("--population-size", type=int, default=100, required=False, help="Size of the population")
    parser.add_argument("--iterations", type=int, default=1000, required=False, help="Number of iterations")
    parser.add_argument("--elite-percent", type=float, default=0.1, required=False, help="Elite percent of population")
    parser.add_argument(
        "--elite-crossover-group-size",
        type=int,
        default=2,
        required=False,
        help="Number of genes to sample from population for crossover"
    )
    parser.add_argument("--offspring-percent", type=float, default=0.4, required=False, help="Percent of offspring")
    parser.add_argument("--mutation-proba", type=float, default=0.1, required=False, help="Probability of mutation")
    parser.add_argument("--max-mutations", type=int, default=1, required=False, help="Max number of possible mutations")
    parser.add_argument(
        "--leave-max-proba",
        type=float,
        default=0.9,
        required=False,
        help="Probability of leaving max flowers at some point",
    )
    
    return parser.parse_args()


class Optimizer:
    def __init__(
        self,
        n_populations: int,
        population_size: int,
        n_iterations: int,
        elite_percent: float,
        elite_crossover_group_size: int,
        offspring_percent: float,
        mutation_proba: float,
        max_mutations: int,
        leave_max_proba: float,
    ):
        assert n_populations >= 2
        assert 0 <= elite_percent <= 1
        assert 0 <= offspring_percent <= 1
        assert elite_percent + offspring_percent <= 1
        assert 0 <= mutation_proba <= 1
        assert 0 <= leave_max_proba <= 1

        self._n_populations = n_populations
        self._population_size = population_size
        self._n_iterations = n_iterations
        self._n_elites = int(population_size * elite_percent)
        self._elite_crossover_group_size = elite_crossover_group_size
        self._n_offsprings = int(population_size * offspring_percent)
        self._n_random = self._population_size - self._n_elites - self._n_offsprings
        self._mutation_proba = mutation_proba
        self._max_mutations = max_mutations
        self._leave_max_proba = leave_max_proba

        assert self._n_random >= 0

    def _build_feasible(self, solution: Solution, task: data.Task) -> Solution:
        assert solution.check_capacities(task.max_loads)

        feasible_solution = deepcopy(solution)

        current_capacities = deepcopy(task.max_loads)
        loaded_couriers = set()
        for courier in range(task.n_couriers):
            current_capacities[courier] -= sum(load for load in feasible_solution.loads[courier] if load is not None)
            assert current_capacities[courier] >= 0

            if current_capacities[courier] > 0:
                loaded_couriers.add(courier)

        current_demands = deepcopy(task.demands)
        demand_points = set()
        for point in range(task.n_points):
            point_flowers = 0
            for courier in range(task.n_couriers):
                for courier_point, courier_point_flowers in zip(
                    feasible_solution.paths[courier], feasible_solution.loads[courier]
                ):
                    if courier_point == point:
                        point_flowers += courier_point_flowers or 0

            current_demands[point] -= point_flowers
            if current_demands[point] > 0:
                demand_points.add(point)

        while len(demand_points) > 0:
            assert len(loaded_couriers) > 0

            courier = random.choice(list(loaded_couriers))
            none_idx = None
            for idx, load in enumerate(feasible_solution.loads[courier]):
                if load is None:
                    none_idx = idx
                    break
            if none_idx is not None:
                point = feasible_solution.paths[courier][none_idx]
            else:
                point = random.choice(list(demand_points))

            assert current_capacities[courier] > 0

            max_flowers = min(current_demands[point], current_capacities[courier])
            if stats.bernoulli(p=self._leave_max_proba).rvs() or point not in demand_points:
                flowers = max_flowers
            else:
                flowers = random.randint(1, max_flowers)

            current_capacities[courier] -= flowers
            current_demands[point] -= flowers

            if none_idx is not None:
                feasible_solution.change_index(courier, none_idx, point, flowers)
            else:
                feasible_solution.add_point(courier, point, flowers)

            if current_capacities[courier] == 0:
                loaded_couriers.remove(courier)
            if current_demands[point] == 0 and point in demand_points:
                demand_points.remove(point)

        for courier in range(task.n_couriers):
            feasible_solution.paths[courier] = feasible_solution.paths[courier][:len(feasible_solution.loads[courier])]

        assert feasible_solution.is_feasible(task)

        return feasible_solution

    def _create_gene(self, task: data.Task) -> Solution:
        return self._build_feasible(Solution.get_empty(task.n_couriers), task)

    def _create_population(self, task: data.Task, population_size: int) -> T.List[Solution]:
        return [self._create_gene(task) for _ in range(population_size)]
    
    def _crossover(self, first_parent: Solution, second_parent: Solution, task: data.Task) -> Solution:
        offspring = Solution.get_empty(task.n_couriers)

        for courier in range(task.n_couriers):
            crossover = crossovers.get_random_crossover()
            offspring.paths[courier] = crossover.cross(first_parent.paths[courier], second_parent.paths[courier])
            offspring.loads[courier] = []
        
        return offspring

    def _create_offsprings(self, genes: T.List[T.List[Solution]], task: data.Task, n_offsprings: int) -> T.List[Solution]:
        offsprings = []
        for _ in range(n_offsprings):
            best_genes = []
            for population_elites in genes:
                assert len(population_elites) >= self._elite_crossover_group_size

                selected_genes: T.List[Solution] = np.random.choice(
                    population_elites, size=self._elite_crossover_group_size, replace=False
                )
                best_population_gene = max(selected_genes, key=lambda gene: gene.cost(task))
                best_genes.append(best_population_gene)

            crossover = crossovers.get_random_crossover()
            first_parent, second_parent = np.random.choice(best_genes, size=2, replace=False)
            offspring = crossover.cross(first_parent, second_parent)

            feasible_offspring = self._build_feasible(offspring, task)

            offsprings.append(feasible_offspring)
        
        return offsprings
    
    def _mutate_offsprings(self, offsprings: T.List[Solution], task: data.Task) -> T.List[Solution]:
        mutated_offsprings = []
        for offspring in offsprings:
            mutated_offspring = offspring

            for _ in range(self._max_mutations):
                mutation = mutations.get_random_mutation()
                mutated_offspring = mutation.mutate(mutated_offspring, self._mutation_proba, task=task)
                mutated_offspring = self._build_feasible(mutated_offspring, task)
            mutated_offspring.improve(task)
            mutated_offsprings.append(mutated_offspring)

        return mutated_offsprings

    def optimize(self, task: data.Task) -> Solution:
        populations = [self._create_population(task, self._population_size) for _ in range(self._n_populations)]

        bar = tqdm(range(self._n_iterations))
        result = None
        global_min = None
        for _ in bar:
            sorted_cost_genes = [
                sorted(population, key=lambda gene: gene.cost(task)) for population in populations
            ]
            elite_genes = [
                sorted_population_genes[:self._n_elites] for sorted_population_genes in sorted_cost_genes
            ]

            min_cost = min(
                min(population_elite_genes, key=lambda gene: gene.cost(task)).cost(task)
                for population_elite_genes in elite_genes
            )

            if global_min is None or min_cost < global_min:
                global_min = min_cost
                found = False
                for population in populations:
                    for gene in population:
                        if gene.cost(task) == global_min:
                            result = gene
                            found = True
                            break
                    if found:
                        break
                if not found:
                    raise RuntimeError(f"Unable to find best gene with cost {global_min}")
                     
            bar.set_description(f"Min cost: {result.cost(task)}")

            offsprings = self._create_offsprings(deepcopy(elite_genes), task, self._n_offsprings)
            mutated = [
                self._mutate_offsprings(deepcopy(population_elites + offsprings), task) for population_elites in elite_genes
            ]
            random_genes = [self._create_population(task, self._n_random) for _ in range(self._n_populations)]

            populations = [
                set(population_elites + population_mutated + random_population_genes)
                for population_elites, population_mutated, random_population_genes in zip(elite_genes, mutated, random_genes)]

        return result


def main():
    args = _parse_args()

    if args.case == 'a':
        task = data.task_a
    elif args.case == 'b':
        task = data.task_b
    else:
        raise ValueError(f"Unknown test case {args.case}")
    
    optimizer = Optimizer(
        n_populations=args.n_populations,
        population_size=args.population_size,
        n_iterations=args.iterations,
        elite_percent=args.elite_percent,
        elite_crossover_group_size=args.elite_crossover_group_size,
        offspring_percent=args.offspring_percent,
        mutation_proba=args.mutation_proba,
        max_mutations=args.max_mutations,
        leave_max_proba=args.leave_max_proba,
    )

    solution = optimizer.optimize(task)
    print(solution)


if __name__ == "__main__":
    main()