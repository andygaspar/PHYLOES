import random
import torch
from Solvers.FastME.fast_me import FastMeSolver

from Solvers.PhyloES.PhyloEsUtils.utils import random_trees_generator, adjust_matrices
from Solvers.solver import Solver


class PhyloES2(Solver):
    def __init__(self, d, batch=10, max_iterations=25, fast_me=True, spr=True, max_non_improve_iter=10):
        super().__init__(d)

        self.d = torch.tensor(self.d, device=self.device)

        self.batch = batch
        self.counter = 0
        self.best_iteration = None
        self.better_solutions = []
        self.fast_me = fast_me
        self.max_iterations = max_iterations
        self.iterations = None
        self.fast_time = 0
        self.fast_me_solver = \
            FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=spr, triangular_inequality=False, logs=False)

        self.max_non_improve_iter = max_non_improve_iter
        self.stop_criterion = None

    def solve(self):
        self.obj_val = 10**5
        init_mats = self.initial_adj_mat(self.device, self.batch)
        obj_vals, adj_mats = random_trees_generator(3, self.d, init_mats, self.n_taxa, self.powers, self.device)
        best = torch.argmin(obj_vals)
        run_val, run_sol = obj_vals[best], adj_mats[best]
        tj = torch.zeros((self.max_iterations*self.batch, self.n_taxa - 3), device=self.device, dtype=torch.long)
        objs = torch.ones(self.max_iterations*self.batch, device=self.device)*100

        not_improved_counter = 0
        combs, i = 2, 0
        while combs > 1 and i < self.max_iterations and not_improved_counter < self.max_non_improve_iter:
            best_val, adj_mats = self.run_fast_me(adj_mats)
            best = torch.argmin(obj_vals)

            if best_val[best] < run_val:
                run_val, run_sol = best_val[best], adj_mats[best]
            if run_val < self.obj_val:
                self.obj_val, self.solution = run_val, run_sol
                not_improved_counter = 0
            else:
                not_improved_counter += 1

            trajectories = self.back_track(adj_mats, best_val)
            tj[i*self.batch: (i+1)*self.batch] = trajectories
            objs[i*self.batch: (i+1)*self.batch] = best_val
            init_mats = self.initial_adj_mat(self.device, self.batch)
            combs, obj_vals, adj_mats = self.distribution_policy(init_mats, tj, objs)
            best = torch.argmin(obj_vals)
            run_val, run_sol = obj_vals[best], adj_mats[best]
            i += 1

        best_val, adj_mats = self.run_fast_me(adj_mats)
        best = torch.argmin(obj_vals)

        if best_val[best] < run_val:
            run_val, run_sol = best_val[best], adj_mats[best]
        if run_val < self.obj_val:
            self.obj_val, self.solution = run_val, run_sol

        self.iterations = i * self.batch

        self.stop_criterion = 'convergence' if combs == 1 else \
            ('max_iterations' if i == self.max_iterations else 'local_plateau')

        self.T = self.get_tau(self.solution.to('cpu'))
        self.d = self.d.to('cpu')
        self.obj_val = self.compute_obj()

    def back_track(self, trees, objs):
        # idxs = torch.argsort(objs)
        sols = trees
        trajectories = self.tree_climb(sols)
        return trajectories

    def tree_climb(self, adj_mats):
        last_inserted_taxa = self.n_taxa - 1
        n_internals = self.m - self.n_taxa

        adj_mats = adjust_matrices(adj_mats, last_inserted_taxa, n_internals, self.n_taxa)
        adj_mats = adj_mats.unsqueeze(1).repeat(1, 2, 1, 1)
        reversed_idxs = torch.tensor([[i, i - 1] for i in range(1, adj_mats.shape[0]*2, 2)], device=adj_mats.device).flatten()
        trajectories = torch.zeros((adj_mats.shape[0], self.n_taxa - 3), dtype=torch.int)

        last_inserted_taxa = self.n_taxa - 1
        for step in range(self.m - 1, self.n_taxa, -1):

            adj_mats[:, 1, :, last_inserted_taxa] = adj_mats[:, 1, last_inserted_taxa, :] = 0
            idxs = torch.nonzero(adj_mats[:, 1, step])
            idxs = torch.column_stack([idxs, idxs[:, 1][reversed_idxs]])

            adj_mats[idxs[:, 0], 1, idxs[:, 1], idxs[:, 2]] = adj_mats[idxs[:, 0], 1, idxs[:, 2], idxs[:, 1]] = 1
            adj_mats[:, 1, :, step] = adj_mats[:, 1, step, :] = 0

            k = (last_inserted_taxa - 1)*2 - 1
            all_non_zeros = torch.nonzero(torch.triu(adj_mats[:, 1, :, :])).view(adj_mats.shape[0], k , 3)
            chosen = idxs[range(0, adj_mats.shape[0] * 2, 2)].repeat_interleave(k, dim=0).view(adj_mats.shape[0], k , 3)
            tj = torch.argwhere((all_non_zeros == chosen).prod(dim=-1))
            trajectories[:, last_inserted_taxa - 3] = tj[:, 1]

            last_inserted_taxa -= 1
        return  trajectories

    @staticmethod
    def permute(adj_mats, step, idx):
        adj_mats[:, step, :] += adj_mats[:, idx, :]
        adj_mats[:, idx, :] = adj_mats[:, step, :] - adj_mats[:, idx, :]
        adj_mats[:, step, :] -= adj_mats[:, idx, :]

        adj_mats[:, :, step] += adj_mats[:, :, idx]
        adj_mats[:, :, idx] = adj_mats[:, :, step] - adj_mats[:, :, idx]
        adj_mats[:, :, step] -= adj_mats[:, :, idx]
        return adj_mats

    def distribution_policy(self, adj_mats, trajectories, obj_vals):
        batch_size = adj_mats.shape[0]
        idxs = torch.argsort(obj_vals)
        trajectories = trajectories[idxs][:batch_size]
        combs = 1
        for step in range(3, self.n_taxa):
            c = torch.unique(trajectories[:, step - 3], return_counts=True)
            # print(c[0], c[1])
            combs *= c[1].shape[0]
            idxs = random.choices(c[0], k=batch_size)
            # idxs = trajectories[:, step - 3][idxs].to(torch.long)
            idxs_list = torch.nonzero(torch.triu(adj_mats)).reshape(batch_size, -1, 3)
            idxs_list = idxs_list[[range(batch_size)], idxs, :]
            idxs_list = (idxs_list[:, :, 0], idxs_list[:, :, 1], idxs_list[:, :, 2])
            adj_mats = self.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=self.n_taxa)

        # print('combs', combs)

        obj_vals = self.compute_obj_val_batch(adj_mats, self.d, self.powers, self.n_taxa, self.device)
        return combs, obj_vals, adj_mats

    def run_fast_me(self, adj_mats):

        tau = self.get_tau_tensors(adj_mats).to('cpu')
        self.fast_me_solver.update_topology(tau, self.batch)
        self.fast_me_solver.solve_timed()
        self.fast_time += self.fast_me_solver.solve_time

        return torch.tensor(self.fast_me_solver.obj_val, device=self.device), torch.tensor(self.fast_me_solver.solution, device=self.device)

