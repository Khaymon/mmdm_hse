from copy import deepcopy
import random

from solution import Solution


class Crossover:
    @classmethod
    def _cross(cls, first_parent: Solution, second_parent: Solution) -> Solution:
        raise NotImplementedError()

    @classmethod
    def cross(cls, first_parent: Solution, second_parent: Solution) -> Solution:        
        return cls._cross(first_parent, second_parent)
        

class PMXCrossover(Crossover):
    """
    Partially Mapped Crossover
    
    Changes a middle part of parents paths
    """

    @classmethod
    def _cross(cls, first_parent: Solution, second_parent: Solution) -> Solution:
        assert first_parent.n_couriers == second_parent.n_couriers

        offspring = Solution.get_empty(first_parent.n_couriers)

        for courier in range(first_parent.n_couriers):
            first_path = first_parent.paths[courier]
            second_path = second_parent.paths[courier]

            first_from = random.randint(0, len(first_path) - 1)
            first_to = random.randint(first_from + 1, len(first_path))

            second_from = random.randint(0, len(second_path) - 1)
            second_to = random.randint(second_from + 1, len(second_path))

            first_load = first_parent.loads[courier]

            changed_path = first_path[:first_from] + second_path[second_from:second_to] + first_path[first_to:]
            changed_load = first_load[:first_from] + [None] * (second_to - second_from) + first_load[first_to:]

            result_path = []
            result_load = []
            used_points = set()
            for point, load in zip(changed_path, changed_load):
                if point not in used_points:
                    used_points.add(point)
                    result_path.append(point)
                    result_load.append(load)

            offspring.paths[courier] = result_path
            offspring.loads[courier] = result_load

        return offspring


class CycleCrossover(Crossover):
    """
    Cycle Crossover

    Finds the cycles in paths and creates new one
    """

    @classmethod
    def _cross(cls, first_parent: Solution, second_parent: Solution) -> Solution:
        assert first_parent.n_couriers == second_parent.n_couriers

        offspring = Solution.get_empty(first_parent.n_couriers)

        for courier in range(first_parent.n_couriers):
            first_path = first_parent.paths[courier]
            first_loads = first_parent.loads[courier]

            second_path = second_parent.paths[courier]
            second_loads = second_parent.loads[courier]

            first_dict = {}
            for i in range(len(first_path) - 1):
                first_dict[first_path[i]] = (i + 1, first_path[i + 1])
            first_dict[first_path[-1]] = (None, None)
            
            second_dict = {}
            for i in range(len(second_path) - 1):
                second_dict[second_path[i]] = (i + 1, second_path[i + 1])
            second_dict[second_path[-1]] = (None, None)

            current_point = first_path[0]
            current_dict = second_dict
            current_loads = second_loads
            is_current_first = False

            result_path = [current_point]
            result_loads = [first_loads[0]]
            used = {current_point}
            while current_point in current_dict:
                next_point_idx, next_point = current_dict[current_point]
                if next_point is None or next_point in used:
                    break
                current_point = next_point
                used.add(current_point)
                result_path.append(current_point)
                result_loads.append(current_loads[next_point_idx])
                
                if is_current_first:
                    current_dict = second_dict
                    current_loads = second_loads
                else:
                    current_dict = first_dict
                    current_loads = first_loads
                is_current_first = not is_current_first


            assert len(result_path) == len(result_loads)
            offspring.paths[courier] = result_path
            offspring.loads[courier] = result_loads

        return offspring


class AddPointsCrossover(Crossover):
    @classmethod
    def _cross(cls, first_parent: Solution, second_parent: Solution) -> Solution:
        assert first_parent.n_couriers == second_parent.n_couriers

        offspring = Solution.get_empty(first_parent.n_couriers)

        for courier in range(first_parent.n_couriers):
            first_path = first_parent.paths[courier]
            second_path = second_parent.paths[courier]

            result_path = deepcopy(first_path)
            result_loads = deepcopy(first_parent.loads[courier])

            second_path_unique_points = set(first_path).difference(second_path)
            for point in second_path_unique_points:
                result_path.append(point)
                result_loads.append(None)
                
            offspring.paths[courier] = result_path
            offspring.loads[courier] = result_loads

        return offspring


def get_random_crossover() -> Crossover:
    return random.choice([
        PMXCrossover,
        # CycleCrossover,
        AddPointsCrossover,
    ])
