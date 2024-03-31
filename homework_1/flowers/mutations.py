from copy import deepcopy
import random
import typing as T

import numpy as np
from scipy import stats

from solution import Solution

class Mutation:
    @classmethod
    def _mutate(cls, solution: Solution, **kwargs) -> Solution:
        raise NotImplementedError()

    @classmethod
    def mutate(cls, solution: Solution, mutation_proba: float, **kwargs) -> Solution:
        if stats.bernoulli(p=mutation_proba).rvs():
            return solution

        return cls._mutate(solution, **kwargs)


class ShuffleMutation(Mutation):
    @classmethod
    def _mutate(cls, solution: Solution, **kwargs) -> Solution:
        mutated_solution = deepcopy(solution)
        for courier in range(mutated_solution.n_couriers):
            if stats.bernoulli(p=0.5).rvs():
                continue

            ids = np.arange(len(mutated_solution.paths[courier]))
            random.shuffle(ids)

            result_path = []
            result_loads = []
            for idx in ids:
                result_path.append(mutated_solution.paths[courier][idx])
                result_loads.append(mutated_solution.loads[courier][idx])
            
            mutated_solution.paths[courier] = result_path
            mutated_solution.loads[courier] = result_loads

        return mutated_solution


class CyclicShiftMutation(Mutation):
    @classmethod
    def _mutate(cls, solution: Solution, **kwargs) -> Solution:
        mutated_solution = deepcopy(solution)
        for courier in range(mutated_solution.n_couriers):
            if stats.bernoulli(p=0.5).rvs():
                continue

            ids = np.arange(len(mutated_solution.paths[courier]))
            start_idx = random.choice(ids)
            ids = np.concatenate([ids[start_idx:], ids[:start_idx]])

            result_path = []
            result_loads = []
            for idx in ids:
                result_path.append(mutated_solution.paths[courier][idx])
                result_loads.append(mutated_solution.loads[courier][idx])
            
            mutated_solution.paths[courier] = result_path
            mutated_solution.loads[courier] = result_loads

        return mutated_solution


class ChangePointsMutation(Mutation):
    @classmethod
    def _mutate(cls, solution: Solution, **kwargs) -> Solution:
        mutated_solution = deepcopy(solution)
        
        first_courier, second_courier = np.random.choice(mutated_solution.n_couriers, size=2, replace=False)

        first_path = mutated_solution.paths[first_courier]
        second_path = mutated_solution.paths[second_courier]

        first_points = set(first_path)
        second_points = set(second_path)

        first_point_selection = random.choice(list(first_points))
        second_point_selection = random.choice(list(second_points))

        first_point_idx = first_path.index(first_point_selection)
        second_point_idx = second_path.index(second_point_selection)

        mutated_solution.paths[first_courier][first_point_idx] = second_point_selection
        mutated_solution.loads[first_courier][first_point_idx] = None

        mutated_solution.paths[second_courier][second_point_idx] = first_point_selection
        mutated_solution.loads[second_courier][second_point_idx] = None

        return mutated_solution


class TakePointsMutation(Mutation):
    @classmethod
    def _mutate(cls, solution: Solution, **kwargs) -> Solution:
        mutated_solution = deepcopy(solution)
        
        first_courier, second_courier = np.random.choice(solution.n_couriers, size=2, replace=False)

        take_point_idx = np.random.choice(len(solution.paths[second_courier]), size=1)[0]
        point = solution.paths[second_courier][take_point_idx]

        solution.paths[second_courier].pop(take_point_idx)
        solution.loads[second_courier].pop(take_point_idx)

        solution.paths[first_courier].append(point)
        solution.loads[first_courier].append(None)

        return mutated_solution


MUTATIONS = [
    ShuffleMutation,
    ChangePointsMutation,
    TakePointsMutation,
]


def get_random_mutation() -> T.List[Mutation]:
    return random.choice(MUTATIONS)
