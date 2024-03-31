import argparse
from tqdm import tqdm
import typing as T

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--variables", type=int, required=False, default=3, help="Number of variables")
    parser.add_argument("--population-size", type=int, default=100, required=False, help="Size of the population")
    parser.add_argument("--iterations", type=int, default=1000, required=False, help="Number of iterations")
    parser.add_argument("--keep-percent", type=float, default=0.5, required=False, help="Percent from previous population")
    parser.add_argument("--mutation-proba", type=float, default=0.5, required=False, help="Probability of mutation")
    
    return parser.parse_args()


class Function:
    def __init__(self, n_variables: int):
        self.n_variables = n_variables
        
    def _calculate(self, point: np.ndarray) -> float:
        raise NotImplementedError()
        
    def __call__(self, point: np.ndarray) -> float:
        assert len(point) == self.n_variables
        
        return self._calculate(point)
    
    
class FunctionA(Function):
    def __init__(self, n_variables: int):
        super().__init__(n_variables=n_variables)

    def _calculate(self, point: np.ndarray) -> float:
        inside_sum = np.sum((point - np.full(shape=len(point), fill_value=1 / np.sqrt(len(point)))) ** 2)
        
        return 1 - np.exp(-inside_sum)
    

class FunctionB(Function):
    def __init__(self, n_variables: int):
        super().__init__(n_variables=n_variables)

    def _calculate(self, point: np.ndarray) -> float:
        inside_sum = np.sum((point + np.full(shape=len(point), fill_value=1 / np.sqrt(len(point)))) ** 2)
        
        return 1 - np.exp(-inside_sum)


Sampler = T.Callable[[int], float]
DominatesComparator = T.Callable[[np.ndarray, np.ndarray], bool]


def min_dominates_comparator(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    return all(lhs <= rhs) and any(lhs < rhs)


def max_dominates_comparator(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    return all(lhs >= rhs) and any(lhs > rhs)


class CrowdDistance:
    def sort(self, results: np.ndarray, eps: float = 1e-6) -> T.List[int]:
        distances = [0 for _ in range(len(results))]
        
        for dim in range(results.shape[1]):
            dim_argsorted = np.argsort(results[:, dim])
            min_dim = np.max(results[:, dim])
            max_dim  = np.max(results[:, dim])

            distances[dim_argsorted[0]] = np.inf
            distances[dim_argsorted[-1]] = np.inf
            
            for i in range(1, len(dim_argsorted) - 1):
                distances[dim_argsorted[i]] += (results[dim_argsorted[i + 1], dim] - results[dim_argsorted[i - 1], dim]) / (max_dim - min_dim + eps)
        
        argsorted_distances = np.argsort(distances)[::-1]
        return argsorted_distances

def show_population(population: np.ndarray, axis: plt.Axes, **kwargs):
    assert population.shape[1] == 2
    
    axis.scatter(population[:, 0], population[:, 1], **kwargs)


class Optimizer:
    class Direction:
        MIN = "min"
        MAX = "max"

    def __init__(
        self,
        population_size: int,
        n_iterations: int,
        keep_percent: float,
        mutation_proba: float,
        mutation_sampler: Sampler = stats.norm(loc=0, scale=1).rvs,
    ):
        assert 0 < keep_percent < 1
        assert 0 < mutation_proba < 1

        self._population_size = population_size
        self._n_iterations = n_iterations
        self._keep_size = int(population_size * keep_percent)
        self._mutation_proba = mutation_proba
        self._mutation_sampler = mutation_sampler
        
    def get_frontiers(self, results: np.ndarray, comparator: DominatesComparator) -> T.List[np.ndarray]:
        dominates = [set() for _ in range(len(results))]
        dominators = np.array([0 for _ in range(len(results))])

        for i in range(len(results) - 1):
            for j in range(i + 1, len(results)):
                if comparator(results[i], results[j]):  # i dominates j
                    dominates[i].add(j)
                    dominators[j] += 1
                elif comparator(results[j], results[i]):  # j dominates i
                    dominates[j].add(i)
                    dominators[i] += 1

        frontiers = []
        used = set()
        while any(dominators > 0):
            current_frontier = np.array([i for i in range(len(dominators)) if dominators[i] == 0 and i not in used])
            used.update(current_frontier)
            
            for i in range(len(current_frontier) - 1):
                for j in range(i + 1, len(current_frontier)):
                    assert not min_dominates_comparator(results[current_frontier[i]], results[current_frontier[j]])
                    assert not min_dominates_comparator(results[current_frontier[j]], results[current_frontier[i]])
            
            frontiers.append(current_frontier)
            for i in current_frontier:
                for j in dominates[i]:
                    dominators[j] -= 1
        
        return frontiers
        
    def select_parents(self, results: np.ndarray, direction: str) -> T.Tuple[np.ndarray, np.ndarray]:
        if direction == self.Direction.MIN:
            comparator = min_dominates_comparator
        elif direction == self.Direction.MAX:
            comparator = max_dominates_comparator
        else:
            raise ValueError(f"Unknown direction {direction}")
        
        frontiers = self.get_frontiers(results, comparator)
        
        parents = []
        ranks = []

        for rank, frontier in enumerate(frontiers):
            if len(frontier) + len(parents) <= self._keep_size:
                parents.extend(frontier)
                ranks.extend([rank] * len(frontier))
            else:
                need_to_take = self._keep_size - len(parents)
                crowd_distance = CrowdDistance()
                crowd_distance_selected = crowd_distance.sort(results[frontier])

                parents.extend(frontier[crowd_distance_selected[:need_to_take]])
                ranks.extend([rank] * need_to_take)

        return np.array(parents), np.array(ranks)

    def create_offspring(self, parents: np.ndarray, ranks: np.ndarray, size: int) -> np.ndarray:
        assert len(parents) >= 4
        offspring = []
        
        for _ in range(size):
            pairs = np.random.choice(np.arange(len(parents)), size=4, replace=False)
            
            if ranks[pairs[0]] < ranks[pairs[1]]:
                first_parent = parents[pairs[0]]
            else:
                first_parent = parents[pairs[1]]
                
            if ranks[pairs[2]] < ranks[pairs[3]]:
                second_parent = parents[pairs[2]]
            else:
                second_parent = parents[pairs[3]]
                
            first_parent_genes = stats.bernoulli(p=0.5).rvs(parents.shape[1])
            second_parent_genes = np.ones_like(first_parent_genes) - first_parent_genes
            
            offspring.append(first_parent * first_parent_genes + second_parent * second_parent_genes)

        return np.vstack(offspring)
    
    def create_mutations(self, children: np.ndarray, bounds: T.List[T.Tuple[float, float]]) -> np.ndarray:
        possible_mutations = [self._mutation_sampler(len(children)) for _ in range(children.shape[1])]
        possible_mutations = np.vstack(possible_mutations).T
        
        mutations_mask = stats.bernoulli(p=self._mutation_proba).rvs(children.size).reshape(children.shape)
        
        children_mutated = children + possible_mutations * mutations_mask
        for dim in range(children_mutated.shape[1]):
            children_mutated[:, dim] = np.clip(children_mutated[:, dim], bounds[dim][0], bounds[dim][1])
            
        return children_mutated

    def optimize(
        self,
        functions: T.List[Function],
        samplers: T.List[Sampler],
        bounds: T.List[T.Tuple[float, float]],
        direction: str = Direction.MIN
    ) -> T.Tuple[np.ndarray, np.ndarray]:
        population = [sampler(self._population_size) for sampler in samplers]
        population = np.vstack(population).T
        
        all_results = []
        plots = []
        for itaration in tqdm(range(self._n_iterations)):
            population_results = np.array([[f(point) for f in functions] for point in population])
            all_results.append(population_results)
            
            parents_ids, ranks = self.select_parents(population_results, direction)
            parents = population[parents_ids]
            
            current_frontier_ids = parents_ids[ranks == 0]

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            timer = fig.canvas.new_timer(interval = 500)
            timer.add_callback(lambda: plt.close())
            show_population(population_results, ax, color="red", label="Previous population")
            show_population(population_results[current_frontier_ids], ax, color="blue", label="Current frontier")
            ax.legend()
            timer.start()
            plt.show()

            offspring = self.create_offspring(parents, ranks, self._population_size - self._keep_size)
            mutated_offspring = self.create_mutations(offspring, bounds)

            population = np.vstack([parents, mutated_offspring])

        frontier_ids = parents_ids[ranks == 0]
        frontier_results = population_results[frontier_ids]

        return frontier_results, np.vstack(all_results), plots

def main():
    args = _parse_args()
    
    optimizer = Optimizer(args.population_size, args.iterations, args.keep_percent, args.mutation_proba)
    
    bounds = [(-4, 4) for _ in range(args.variables)]
    
    sampler = stats.uniform(loc=-4, scale=4).rvs
    results, all_results, plots = optimizer.optimize(
        [FunctionA(args.variables), FunctionB(args.variables)],
        [sampler] * args.variables,
        bounds
    )

    _, ax = plt.subplots(1, 1, figsize=(12, 8))
    show_population(all_results, ax, label="All results", color="red")
    show_population(results, ax, label="Frontier results", color="blue")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
