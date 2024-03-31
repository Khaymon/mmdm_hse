from dataclasses import dataclass
import typing as T


DISTANCE_MATRIX = [
    [0, 5.48, 7.76, 6.96, 5.82, 2.74, 5.02, 1.94, 3.08, 1.94, 5.36, 5.02, 3.88, 3.54, 4.68, 7.76, 6.62],
    [5.48, 0, 6.84, 3.08, 1.94, 5.02, 7.30, 3.54, 6.96, 7.42, 10.84, 5.94, 4.80, 6.74, 10.16, 8.68, 12.10],
    [7.76, 6.84, 0, 9.92, 8.78, 5.02, 2.74, 8.10, 4.68, 7.42, 4.00, 12.78, 11.64, 11.30, 7.88, 15.52, 7.54],
    [6.96, 3.08, 9.92, 0, 1.14, 6.50, 8.78, 5.02, 8.44, 8.90, 12.32, 5.14, 6.28, 8.22, 11.64, 5.60, 13.58],
    [5.82, 1.94, 8.78, 1.14, 0, 5.36, 7.64, 3.88, 7.30, 7.76, 11.18, 4.00, 5.14, 7.08, 10.50, 6.74, 12.44],
    [2.74, 5.02, 5.02, 6.50, 5.36, 0, 2.28, 3.08, 1.94, 2.40, 5.82, 7.76, 6.62, 6.28, 5.14, 10.50, 7.08],
    [5.02, 7.30, 2.74, 8.78, 7.64, 2.28, 0, 5.36, 1.94, 4.68, 3.54, 10.04, 8.90, 8.56, 5.14, 12.78, 4.80],
    [1.94, 3.54, 8.10, 5.02, 3.88, 3.08, 5.36, 0, 3.42, 3.88, 7.30, 4.68, 3.54, 3.20, 6.62, 7.42, 8.56],
    [3.08, 6.96, 4.68, 8.44, 7.30, 1.94, 1.94, 3.42, 0, 2.74, 3.88, 8.10, 6.96, 6.62, 3.20, 10.84, 5.14],
    [1.94, 7.42, 7.42, 8.90, 7.76, 2.40, 4.68, 3.88, 2.74, 0, 3.42, 5.36, 4.22, 3.88, 2.74, 8.10, 4.68],
    [5.36, 10.84, 4.00, 12.32, 11.18, 5.82, 3.54, 7.30, 3.88, 3.42, 0, 8.78, 7.64, 7.30, 3.88, 11.52, 3.54],
    [5.02, 5.94, 12.78, 5.14, 4.00, 7.76, 10.04, 4.68, 8.10, 5.36, 8.78, 0, 1.14, 3.08, 6.50, 2.74, 8.44],
    [3.88, 4.80, 11.64, 6.28, 5.14, 6.62, 8.90, 3.54, 6.96, 4.22, 7.64, 1.14, 0, 1.94, 5.36, 3.88, 7.30],
    [3.54, 6.74, 11.30, 8.22, 7.08, 6.28, 8.56, 3.20, 6.62, 3.88, 7.30, 3.08, 1.94, 0, 3.42, 4.22, 5.36],
    [4.68, 10.16, 7.88, 11.64, 10.50, 5.14, 5.14, 6.62, 3.20, 2.74, 3.88, 6.50, 5.36, 3.42, 0, 7.64, 1.94],
    [7.76, 8.68, 15.52, 5.60, 6.74, 10.50, 12.78, 7.42, 10.84, 8.10, 11.52, 2.74, 3.88, 4.22, 7.64, 0, 7.98],
    [6.62, 12.10, 7.54, 13.58, 12.44, 7.08, 4.80, 8.56, 5.14, 4.68, 3.54, 8.44, 7.30, 5.36, 1.94, 7.98, 0]
]
FLOWER_MARKET_ID = 0
DEMANDS = [0, 100, 100, 200, 400, 200, 400, 800, 800, 100, 200, 100, 200, 400, 400, 800, 800]


@dataclass
class Task:
    distance_matrix: T.List[T.List[float]]
    starting_point: int
    demands: T.List[int]
    n_couriers: int
    salaries: T.List[int]
    max_loads: T.List[int]

    def __post_init__(self):
        assert self.n_couriers == len(self.salaries)
        assert self.n_couriers == len(self.max_loads)

        assert 0 <= self.starting_point < len(self.distance_matrix)

        assert all(len(row) == len(self.distance_matrix) for row in self.distance_matrix)
        assert len(self.demands) == len(self.distance_matrix)

        assert sum(self.max_loads) >= sum(self.demands)

        self.n_points = len(self.distance_matrix)

    def path_distance(self, path: T.List[int]) -> float:
        distance = 0
        current_point = 0
        for next_point in path + [0]:
            distance += self.distance_matrix[current_point][next_point]
            current_point = next_point
        return distance


task_a = Task(
    distance_matrix=DISTANCE_MATRIX,
    starting_point=FLOWER_MARKET_ID,
    demands=DEMANDS,
    n_couriers=4,
    salaries=[100, 100, 100, 100],
    max_loads=[2500, 1500, 1500, 500],
)

task_b = Task(
    distance_matrix=DISTANCE_MATRIX,
    starting_point=FLOWER_MARKET_ID,
    demands=DEMANDS,
    n_couriers=4,
    salaries=[100, 80, 80, 60],
    max_loads=[2500, 2000, 1000, 500],
)
