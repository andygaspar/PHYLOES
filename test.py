import ctypes
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from numpy.ctypeslib import ndpointer

from Solvers.FastME.fast_me import FastMeSolver
from Solvers.PhyloES.PhyloEsUtils.utils import random_trees_generator
from Solvers.PhyloES.phyloes_parallel import PhyloES2
from Solvers.RandomFastME.random_fast_me import RandomFastME
from Solvers.PhyloES.phyloes import PhyloES
from Data_.data_loader import DistanceData
from Solvers.fast_cpp.fast_cpp import FastCpp

distances = DistanceData()
distances.print_dataset_names()
dataset_names = distances.get_dataset_names()
data_set_list = []

result_list = []

data_set = distances.get_dataset(3)
print(data_set.name)

problems = 1
run_per_problem = 10
run_list, batch_list, max_iter_list, stop_list, n_better_solution = [], [], [], [], []

random.seed(0)
np.random.seed(0)
data_set_idx = 0

dim = 150

d = np.around(data_set.get_random_mat(dim), 20)
d = d/np.max(d)
# os.system("Solvers/fast_cpp/compile_python.sh")


# lib.test.restype = ctypes.c_void_p
batch_size = 8

phyloes = PhyloES2(d, batch=batch_size, max_iterations=100)
init_mats = phyloes.initial_adj_mat(phyloes.device, phyloes.batch)
obj_vals, adj_mats = random_trees_generator(3, phyloes.d, init_mats, phyloes.n_taxa, phyloes.powers, phyloes.device)
adj_mat_np = adj_mats.numpy().astype(dtype=np.int32)
d = d.astype(np.double)
fast = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, triangular_inequality=False,
                    logs=False)

fast_cpp = FastCpp()


fast_cpp.test_return_object()

t = time.time()
results = fast_cpp.run_parallel(fast.d, adj_mat_np, phyloes.n_taxa, phyloes.m, batch_size)
print("population size", batch_size)
print("available cores", fast_cpp.numProcs)
print('parallel time', time.time() - t)
res_obj = [fast.compute_obj_val_from_adj_mat(a, fast.d, dim) for a in results]


fast = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, triangular_inequality=False, logs=False)

tau = phyloes.get_tau_tensors(adj_mats).to('cpu')
fast.update_topology(tau, batch_size)
fast.solve_timed()

print("old fastme with parallel file time", fast.time)

# 3.3116201441751842
# for a in adj_mat_np:
#     fast = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, triangular_inequality=False,
#                         logs=False)
#     print("kkkkkkkkkkkkkk")
#     print("length initial", fast.compute_obj_val_from_adj_mat(a, fast.d, dim))
#     t = time.time()
#     res = fast_cpp.test(fast.d, a, phyloes.n_taxa)
#     print("da file ", fast.compute_obj_val_from_adj_mat(res, fast.d, dim), time.time()-t)
#     # print(res)
#
#     fast.update_topology(fast.get_tau(a))
#     fast.solve_timed()
#     print("fast normale", fast.time)
#     # print(fast.solution)
#
#     # print(np.array_equal(res, fast.solution))
#
#     print(fast.obj_val)
#
#     # print(d)
#     # print(a)
#     t = time.time()
#     res2 = fast_cpp.run(fast.d, a, phyloes.n_taxa, phyloes.m)
#     print("cpp ", fast.compute_obj_val_from_adj_mat(res2, fast.d, dim), time.time() - t)
#     print("ttttttttttttttttttttttttt\n")

# os.system("Solvers/fast_cpp/fast_me")
#
# for dim in [20, 50, 100]:
#     print('*************** ', dim)
#     for problem in range(problems):
#
#
#         d = np.around(data_set.get_random_mat(dim), 20)
#         d = d/np.max(d)
#
#         for batch in [2, 5, 10, 20]:
#             for max_iter in [50, 100, 200]:
#                 for run in range(run_per_problem):
#                     print(dim, batch, max_iter, run)
#                     result_run = [data_set_idx, dim, problem]
#                     run_list.append(run)
#                     batch_list.append(batch)
#                     max_iter_list.append(max_iter)
#
#                     phyloes = PhyloES2(d, batch=batch, max_iterations=max_iter)
#                     phyloes.solve_timed()
#                     # print("phyloes  time:", phyloes.time, '   obj:', phyloes.obj_val, '   iterations', phyloes.iterations)
#                     phyloes_tj = tuple(phyloes.tree_climb(torch.tensor(phyloes.solution).unsqueeze(0)).to('cpu').tolist()[0])
#                     stop_list.append(phyloes.stop_criterion)
#                     # print(rand_tj_tj)
#                     # print(rand_tj_tj)
#
#                     rand_fast = RandomFastME(d, parallel=False, spr=True)
#                     rand_fast.solve_timed(phyloes.iterations)
#                     # print("rand_fa  time:", rand_fast.time, '   obj:', rand_fast.obj_val, '   iterations', phyloes.iterations)
#                     rand_fast_tj = tuple(phyloes.tree_climb(torch.tensor(rand_fast.solution).unsqueeze(0)).to('cpu').tolist()[0])
#                     # print(rand_fast)
#
#                     #
#                     fast = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, triangular_inequality=False,
#                                         logs=False)
#                     fast.solve_all_flags()
#                     # print("fast  time:",  fast.time, '   obj:', fast.obj_val)
#                     fast_tj = tuple(phyloes.tree_climb(torch.tensor(fast.solution).unsqueeze(0)).to('cpu').tolist()[0])
#                     # print(fast_tj)
#                     result_run += [fast.obj_val, rand_fast.obj_val,  phyloes.obj_val, fast.time, rand_fast.time,
#                                    phyloes.time, fast.method]
#                     result_run += [fast_tj, rand_fast_tj, phyloes_tj]
#                     result_list.append(result_run)
#
#
#
#
#