import numpy as np

from Solvers.FastME.fast_me import FastMeSolver
from Solvers.Random.random_solver import RandomSolver
from Solvers.solver import Solver


class RandomFastME(Solver):
    def __init__(self, d, parallel=False, spr=True):
        super().__init__(d)
        self.fast_me = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=spr,
                                    triangular_inequality=False, logs=False)
        self.random_solver = RandomSolver(d)
        self.parallel = parallel
        self.counter = 0
        self.best_iteration = None
        self.better_solutions = []

    def solve(self, iterations, ob_init_val=10, sol_init=None, count_solutions=False):
        best_val, best_sol = 10 ** 5, sol_init
        self.better_solutions.append(sol_init)
        for i in range(iterations):
            self.random_solver.solve()

            self.fast_me.update_topology(self.random_solver.T)
            self.fast_me.solve()
            if count_solutions:
                if self.fast_me.obj_val < ob_init_val:
                    new = True
                    for sol in self.better_solutions:
                        new = not np.array_equal(self.fast_me.solution, sol)
                        if not new:
                            break
                    if new:
                        self.counter += 1
                        print(self.counter)
                        self.better_solutions.append(self.fast_me.solution)
            if self.fast_me.obj_val < best_val:
                best_val, best_sol = self.fast_me.obj_val, self.fast_me.solution
                self.best_iteration = i

        self.solution = best_sol
        self.obj_val = best_val
        self.T = self.get_tau(self.solution)
