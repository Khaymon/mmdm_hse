## 8-th March flowers problem

### Formalization

Let's assume that couriers can't go back to the shop during a day when they are out of flowers for some clients. Let's also assume that there is no need for couriers to go back to the shop when there is no work for them. As our matrix satisfy the triangle inequality, the distance between each two points is shortest one. It means that couriers optimal strategy is to visit only those points, which demands still exist.

Let's $\displaystyle p_{i} =\left( p_{i}^0,p_{i}^{1} ,p_{i}^{2} ,\dotsc ,p_{i}^{k_{i}}\right) ,i\in \{1,\dotsc ,h\}$, where $\displaystyle h$ is the number of couriers and $p_{i}^0=0$ for every $i$, be the path of $\displaystyle i$-th courier. These paths also have constraint $\sum _{j=1}^{k_{i}} p_{i}^{j} \leqslant c_{i} ,$
where $\displaystyle c_{j}$ is the capacity of courier $\displaystyle c_{j}$. The every feasible solution looks like splitting $\displaystyle p_{1} ,\dotsc ,p_{h}$ of initial verticies $\displaystyle \{1,2,\dotsc ,n\}$ such that $\displaystyle \bigcup _{i=1}^{h} p_{i} \backslash \{0\} =\{1,2,\dotsc ,n\}$ and $\displaystyle \forall i,j\in \{1,2,\dotsc ,h\} ,i\neq j\hookrightarrow p_{i} \cap p_{j} =\{0\}$, which also satisfy condition written above. The cost of the solutions is $m^* = \sum_{i=1}^h\sum_{j=1}^{k_i}d_{p_{j-1}}^{p_j}\cdot s_i$, where $d_{p_{j-1}}^{p_j}$ is the distance between $p_{j-1}$ and ${p_j}$, $s_i$ is the salary of courier $i$. The optimal solution is such feasible solution which cost is minimal.

## Proposed solution

1) Iterate over all feasible solutions by bruteforce. Its time complexity is $O(2^{n+4})=O(2^n)$ because we have $n$ points and $4$ couriers.

2) Find the optimal in terms of distance Hamilton paths on the fully connected graphs $\displaystyle \left\{p_i^0,p_{i}^{1} ,\dotsc p_{i}^{k_{i}}\right\}$ for each $\displaystyle i\in \{1,\dotsc ,h\}$. We can do that by using the breadth-first search.

# Solutions

- Solution(paths=[[7, 4, 3, 15, 11, 12, 13], [8, 2, 6, 5], [9, 10, 16, 14], [1, 7]], loads=[[400, 400, 200, 800, 100, 200, 400], [800, 100, 400, 200], [100, 200, 800, 400], [100, 400]]) -- cost 6392
- Solution(paths=[[9, 14, 16, 10, 2, 6, 8], [12, 11, 15, 13], [7, 1, 4, 3, 5, 8], [7]], loads=[[100, 400, 800, 200, 100, 400, 500], [200, 100, 800, 400], [300, 100, 400, 200, 200, 300], [500]]) -- cost 6140.0
- Solution(paths=[[9, 14, 16, 10, 2, 6, 8], [3, 4, 1, 7], [12, 11, 15, 13], [8, 5]], loads=[[100, 400, 800, 200, 100, 400, 500], [200, 400, 100, 800], [200, 100, 800, 400], [300, 200]]) -- cost 6072.0