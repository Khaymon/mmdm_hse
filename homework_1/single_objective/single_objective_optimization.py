import argparse

import numpy as np
import typing as T


def _parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--function", type=str, choices=['a', 'b'], required=True, help="Number of function to optimize")
    parser.add_argument("--population-size", type=int, default=100, required=False, help="Size of the population")
    parser.add_argument("--iterations", type=int, default=1000, required=False, help="Number of iterations")
    parser.add_argument("--elite-percent", type=float, default=0.1, required=False, help="Elite percent of population")
    
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
    def __init__(self):
        super().__init__(n_variables=2)

    def _calculate(self, point: np.ndarray) -> float:
        x, y = point
        
        result = 3 * (1 - x) ** 2 * np.exp(-x ** 2 - (y + 1) ** 2)
        result += -10 * (x / 5 - x ** 3 - y ** 5) * np.exp(-x ** 2 -y ** 2)
        result += -1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)
        
        return result
    

class FunctionB(Function):
    def __init__(self):
        super().__init__(n_variables=2)

    def _calculate(self, point: np.ndarray) -> float:
        x, y = point
        
        result = np.exp(abs(100 - np.sqrt(x ** 2 + y ** 2) / np.pi))
        result = -0.0001 * np.power(abs(np.sin(x) * np.sin(y) * result) + 1, 0.1)
        
        return result


class Optimizer:
    class Direction:
        MIN = "min"
        MAX = "max"

    def __init__(self, population_size: int, n_iterations: int, elite_percent: float):
        assert 0 < elite_percent < 1

        self._population_size = population_size
        self._n_iterations = n_iterations
        self._n_elites = int(population_size * elite_percent)

    def optimize(self, function: Function, direction: str = Direction.MIN, temperature: float = 1.0) -> T.Tuple[np.ndarray, float]:
        mean = np.zeros(function.n_variables)
        covariance = np.eye(function.n_variables) * temperature
        
        avg_points = []
        for iteration in range(self._n_iterations):
            population = np.random.multivariate_normal(mean, covariance, size=self._population_size)
            population_values = np.array([function(point) for point in population])
            if direction == self.Direction.MAX:
                population_values *= -1

            argsorted_results = np.argsort(population_values)
            
            elites = population[argsorted_results[:self._n_elites]]

            mean = np.mean(elites, axis=0)
            avg_points.append(mean)
            covariance = np.cov(elites, rowvar=False) * temperature
            
        final_point = np.mean(elites, axis=0)
        return final_point, function(final_point), avg_points


def main():
    args = _parse_args()
    
    optimizer = Optimizer(args.population_size, args.iterations, args.elite_percent)
    if args.function == 'a':
        func = FunctionA()
        direction = "max"
    else:
        func = FunctionB()
        direction = "min"

    final_point, function_value, avg_points = optimizer.optimize(func, direction)
    
    print(f"Final point: {final_point}, function value: {function_value}")


if __name__ == "__main__":
    main()
