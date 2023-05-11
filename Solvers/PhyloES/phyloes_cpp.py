import random
import time
from typing import Union, List, Tuple

import numpy as np
import torch
from Solvers.PhyloES.PhyloEsUtils.utils import random_trees_generator_and_objs, adjust_matrices
from Solvers.fast_cpp.fast_cpp import FastCpp
from Solvers.solver import Solver

class PhyloEScpp(Solver):
    def __init__(self, d, batch: Union[int, List[Tuple]] = 16, max_iterations=25, replace=False,
                 max_non_improve_iter=None, min_tol=1e-16):
        super().__init__(d)

        self.d_np = self.d.astype(np.double)
        self.d = torch.tensor(self.d, device=self.device)
        self.replace = replace

        self.counter = 0
        self.best_iteration = None
        self.better_solutions = []
        self.max_iterations = max_iterations
        self.min_tol = min_tol
        self.n_trees = None
        self.fast_time = 0
        self.fast_me_solver = FastCpp()
        self.nni_counter, self.spr_counter = [], []
        self.adaptive = True if type(batch) == list else False
        if self.adaptive:
            self.batch = sorted(batch, key=lambda t: t[0])
        else:
            self.batch = batch
        self.max_non_improve_iter = max_non_improve_iter if max_non_improve_iter is not None else max_iterations * batch
        self.stop_criterion = None

        self.obj_vals = None

        self.best_vals = []
        self.worse_vals = []
        self.iterations = None

    def solve(self):

        iteration = 0
        batch = self.set_batch(iteration) if self.adaptive else self.batch

        init_mats = self.initial_adj_mat(self.device, batch)
        obj_vals, adj_mats = random_trees_generator_and_objs(3, self.d, init_mats, self.n_taxa, self.powers,
                                                             self.device)
        obj_vals, adj_mats = self.run_fast_me(adj_mats, batch)
        best = torch.argmin(obj_vals)
        trajectories = self.back_track(adj_mats)

        self.obj_val, self.solution = obj_vals[best], adj_mats[best]

        tj = torch.zeros((2 * batch, self.n_taxa - 3), device=self.device, dtype=torch.long)
        objs = torch.ones(2 * batch, device=self.device, dtype=torch.float64) * 1000


        tj[batch: 2 * batch] = trajectories
        objs[batch: 2 * batch] = obj_vals
        objs, sorted_idxs = torch.sort(objs)
        tj = tj[sorted_idxs]


        not_improved_counter = 0
        iteration = 1
        combs = 2
        while combs > 1 and iteration < self.max_iterations and objs[batch-1] - objs[0] > self.min_tol:
            # print(iteration, batch, [ob.item() for ob in objs[:batch]], self.nni_counter, self.spr_counter)
            # print([(objs[i] == objs[i + 1]).item() for i in range(batch - 1)])

            batch = self.set_batch(iteration) if self.adaptive else self.batch
            objs, tj = objs[:2*batch],  tj[:2*batch]

            init_mats = self.initial_adj_mat(self.device, batch)
            combs, adj_mats, objs, tj = self.distribution_policy(init_mats, tj, objs, batch)
            self.best_vals.append(objs[0].item())
            self.worse_vals.append(objs[batch-1].item())

            obj_vals, adj_mats = self.run_fast_me(adj_mats, batch)
            best = torch.argmin(obj_vals)
            trajectories = self.back_track(adj_mats)

            if obj_vals[best] < self.obj_val:
                self.obj_val, self.solution = obj_vals[best], adj_mats[best]
                not_improved_counter = 0
            else:
                not_improved_counter += 1

            tj[batch: 2 * batch] = trajectories
            objs[batch: 2 * batch] = obj_vals

            objs, sorted_idxs = torch.sort(objs)
            tj = tj[sorted_idxs]

            iteration += 1

        # print(iteration, batch, [ob.item() for ob in objs[:batch]], self.nni_counter, self.spr_counter)
        # print([(objs[i] == objs[i + 1]).item() for i in range(batch - 1)])

        self.n_trees = iteration * batch

        self.stop_criterion = 'convergence' if combs == 1 else \
            ('min_tol'if objs[batch-1] - objs[0] < self.min_tol else
             ('max_iterations' if iteration == self.max_iterations else 'local_plateau'))
        self.iterations = iteration
        self.T = self.get_tau(self.solution.to('cpu'))
        self.d = self.d.to('cpu')
        self.obj_val = self.compute_obj()

    def set_batch(self, iteration):
        for i in range(len(self.batch) - 1):
            if self.batch[i][0] <= iteration < self.batch[i + 1][0]:
                return self.batch[i][1]
        return self.batch[-1][1]

    def back_track(self, trees):
        sols = trees
        trajectories = self.tree_climb(sols)
        return trajectories

    def tree_climb(self, adj_mats):
        last_inserted_taxa = self.n_taxa - 1
        n_internals = self.m - self.n_taxa

        adj_mats = adjust_matrices(adj_mats, last_inserted_taxa, n_internals, self.n_taxa)
        adj_mats = adj_mats.unsqueeze(1).repeat(1, 2, 1, 1)
        reversed_idxs = torch.tensor([[i, i - 1] for i in range(1, adj_mats.shape[0] * 2, 2)],
                                     device=adj_mats.device).flatten()
        trajectories = torch.zeros((adj_mats.shape[0], self.n_taxa - 3), device=self.device, dtype=torch.int)

        last_inserted_taxa = self.n_taxa - 1
        for step in range(self.m - 1, self.n_taxa, -1):
            adj_mats[:, 1, :, last_inserted_taxa] = adj_mats[:, 1, last_inserted_taxa, :] = 0
            idxs = torch.nonzero(adj_mats[:, 1, step])
            idxs = torch.column_stack([idxs, idxs[:, 1][reversed_idxs]])

            adj_mats[idxs[:, 0], 1, idxs[:, 1], idxs[:, 2]] = adj_mats[idxs[:, 0], 1, idxs[:, 2], idxs[:, 1]] = 1
            adj_mats[:, 1, :, step] = adj_mats[:, 1, step, :] = 0

            k = (last_inserted_taxa - 1) * 2 - 1
            all_non_zeros = torch.nonzero(torch.triu(adj_mats[:, 1, :, :])).view(adj_mats.shape[0], k, 3)
            chosen = idxs[range(0, adj_mats.shape[0] * 2, 2)].repeat_interleave(k, dim=0).view(adj_mats.shape[0], k, 3)
            tj = torch.argwhere((all_non_zeros == chosen).prod(dim=-1))
            trajectories[:, last_inserted_taxa - 3] = tj[:, 1]

            last_inserted_taxa -= 1
        return trajectories

    @staticmethod
    def permute(adj_mats, step, idx):
        adj_mats[:, step, :] += adj_mats[:, idx, :]
        adj_mats[:, idx, :] = adj_mats[:, step, :] - adj_mats[:, idx, :]
        adj_mats[:, step, :] -= adj_mats[:, idx, :]

        adj_mats[:, :, step] += adj_mats[:, :, idx]
        adj_mats[:, :, idx] = adj_mats[:, :, step] - adj_mats[:, :, idx]
        adj_mats[:, :, step] -= adj_mats[:, :, idx]
        return adj_mats

    def distribution_policy(self, adj_mats, trajectories, obj_vals, batch):
        if self.replace:
            equals = [obj_vals[i] == obj_vals[i + 1] for i in range(batch - 1)]
            idx = batch - 2
            while idx > 0 and equals[idx]:
                idx -= 1
            if 0 < idx < batch - 2:
                # print("replacement")
                trajectories[idx + 1] = trajectories[0]
        tj = trajectories[:batch]

        combs = 1
        for step in range(3, self.n_taxa):
            c = torch.unique(tj[:, step - 3], return_counts=True)
            # print(c[0], c[1])
            combs *= c[1].shape[0]
            # idxs = random.choices(c[0], k=self.batch)
            idxs = random.choices(tj[:, step - 3], k=batch)
            idxs_list = torch.nonzero(torch.triu(adj_mats)).reshape(batch, -1, 3)
            idxs_list = idxs_list[[range(batch)], idxs, :]
            idxs_list = (idxs_list[:, :, 0], idxs_list[:, :, 1], idxs_list[:, :, 2])
            adj_mats = self.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=self.n_taxa)

        # print('combs', combs)
        # print(obj_vals[0], obj_vals[self.batch - 1])
        return combs, adj_mats, obj_vals, trajectories

    def run_fast_me(self, adj_mats, batch):
        mats = adj_mats.to('cpu').numpy().astype(dtype=np.int32)
        adjs, objs, nni_counts, spr_counts = \
            self.fast_me_solver.run_parallel(self.d_np, mats, self.n_taxa, self.m, batch)
        self.nni_counter += [nni_counts]
        self.spr_counter += [spr_counts]

        objs_tensor = torch.tensor(objs, device=self.device)
        adjs_tensor = torch.tensor(adjs, device=self.device)
        self.fast_me_solver.free_result_memory()

        return objs_tensor, adjs_tensor
