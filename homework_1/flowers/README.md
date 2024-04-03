# Formalization

Let's assume that couriers can't go back to the shop during a day when they are out of flowers for some clients and they need to go back to end up their working day. Also, each client should be visited only once. As our matrix satisfy the triangle inequality, the distance between each two points is shortest one. It means that couriers optimal strategy is to visit only those points, which demands still exist. Let's provide some definitions.



Let's $\displaystyle p_{i} =\left( p^{0} ,p_{i}^{1} ,\dotsc ,p_{i}^{t_{i}}\right) ,i\in \{1,\dotsc c\}$ be the paths of couriers, where $\displaystyle c$ is the total number of couriers, $\displaystyle p_{i}^{j}$ is the point visited by courier $\displaystyle i$ in timestamp $\displaystyle j$ and $\displaystyle p^{0} \equiv 0$. $\displaystyle m_{i}^{j}$ is the distance between points $\displaystyle i$ and $\displaystyle j$, $\displaystyle s_{i}$ is the salary of $\displaystyle i$-th courier. $\displaystyle d_{j} ,\ j\in \{1,\dotsc ,N\}$ is the demand of $\displaystyle j$-th point, $\displaystyle N$ is the total number of points and $\displaystyle l_{i}$ is the capacity of $\displaystyle i$-th courier. Let's also define the set of feasible solutions $\displaystyle S$ containing vectors $ \overline{p} =( p_{i} ,\ldots ,p_{c})$ which satisfy conditions

1. $ \sum_{j=1}^{t_{i}} d_{p_{i}^{j}} \leqslant l_{i} ,\ \forall i\in \{1,\ldots ,c\}$ which means that courier can fulfill demands of all points which he visit,

2. $ \forall j\in \{1,\ldots ,N\} \ d_{j}  >0\Rightarrow \ \exists i\in \{1,\ldots ,c\} :\ j\in p_{i}$ which means that for non-zero demand points exists courier which will visit this point.

So, our task is to find $ \min_{\overline{p} \in S}\sum _{i=1}^{c}\left(\sum _{j=1}^{t_{i}} m_{p_{i}^{j-1}}^{p_{i}^{j}} \cdotp s_{i}\right) +m_{p_{i}^{t_{i}}}^{p^{0}} \cdotp s_{i}$.

# Proposed solution
As the task is NP-complete, I propose to use genetic algorithm to find the suboptimal solution.

1. Generate population from the set of feasible solutions randomly and find cost for each of these.
2. Select "elite" set of solutions and crossover them to create offsprings,
3. Mutate offsprings and add them to elite ones to create a new population
4. Sample some random solutions once again and mix them to the new popultaion.
5. Repeat until convergence of minimal cost inside population.

# Solutions
My algorithm gave these solutions
## A
`Solution(paths=[[5, 8, 6, 2, 10, 16], [12, 11, 15, 13], [3, 4, 1, 7], [9, 14]])` -- cost 6,392

## B
`Solution(paths=[[7, 1, 4, 3, 15, 12], [9, 14, 16, 10, 2, 6], [8, 5], [11, 13]])` -- cost 5,264