from dataclasses import dataclass
from collections import defaultdict
import itertools
import typing as T

import numpy as np

import flowers_data as data


@dataclass
class Solution:
    paths: T.List[T.List[int]]
    loads: T.List[T.List[int]]

    @classmethod
    def get_empty(cls, n_couriers: int) -> "Solution":
        paths = [[] for _ in range(n_couriers)]
        loads = [[] for _ in range(n_couriers)]

        return cls(paths, loads)
    
    def add_point(self, courier: int, point: int, load: int):
        assert len(self.paths[courier]) == len(self.loads[courier])

        self.paths[courier].append(point)
        self.loads[courier].append(load)

    def change_index(self, courier: int, idx: int, point: int, load: int):
        assert self.paths[courier][idx] == point
        self.loads[courier][idx] = load

    def __post_init__(self):
        assert len(self.paths) == len(self.loads)

        self.n_couriers = len(self.paths)

    def check_capacities(self, capacities: T.List[int]) -> bool:
        assert len(self.loads) == len(capacities)

        for courier_loads, courier_capacity in zip(self.loads, capacities):
            if sum(load for load in courier_loads if load is not None) > courier_capacity:
                return False

        return True

    def check_paths(self, n_points: int) -> bool:
        for courier_path in self.paths:
            assert all(0 <= point < n_points for point in courier_path)

        return True
    
    def check_demands(self, demands: T.List[int]) -> bool:
        delivered_flowers = [0] * len(demands)
        for path, load in zip(self.paths, self.loads):
            for point, point_flowers in zip(path, load):
                delivered_flowers[point] += point_flowers or 0

        return all(flowers >= demand for flowers, demand in zip(delivered_flowers, demands))

    def is_feasible(self, task: data.Task) -> bool:
        assert len(self.paths) == task.n_couriers
        assert len(self.paths) == len(task.max_loads)
            
        if not self.check_capacities(task.max_loads):
            return False
        if not self.check_paths(task.n_points):
            return False
        if not self.check_demands(task.demands):
            return False

        return True

    def unique_visitors(self) -> bool:
        used = set()
        for path in self.paths:
            if used.intersection(path):
                return False
            used.update(path)

        return True

    def cost(self, task: data.Task) -> float:
        if not self.unique_visitors():
            return np.inf

        cost = 0
        for path, salary in zip(self.paths, task.salaries):
            distance = task.path_distance(path)
            cost += distance * salary

        return cost

    def improve(self, task: data.Task):
        for courier in range(self.n_couriers):
            points = set(self.paths[courier])
            if len(points) > 7:
                continue

            
            point_loads_dict = defaultdict(int)
            for point, load in zip(self.paths[courier], self.loads[courier]):
                point_loads_dict[point] += load or 0

            min_path_distance = None
            min_path = None
            for path in itertools.permutations(points):
                current_path_distance = task.path_distance(list(path))
                if min_path is None or current_path_distance < min_path_distance:
                    min_path_distance = current_path_distance
                    min_path = path
            
            result_loads = [point_loads_dict[point] for point in min_path]


            assert len(min_path) == len(result_loads)

            self.paths[courier] = list(min_path)
            self.loads[courier] = list(result_loads)

    def __hash__(self) -> int:
        current_hash = 0
        for path in self.paths:
            current_hash += hash(tuple(path))

        return current_hash
