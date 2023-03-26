import ctypes
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from numpy.ctypeslib import ndpointer

from Solvers.FastME.fast_me import FastMeSolver
from Solvers.PhyloES.PhyloEsUtils.utils import random_trees_generator_and_objs
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
obj_vals, adj_mats = random_trees_generator_and_objs(3, phyloes.d, init_mats, phyloes.n_taxa, phyloes.powers, phyloes.device)
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
res_obj = [fast.compute_obj_val_from_adj_mat(a, fast.d, dim) for a in results[0]]
print(res_obj)
print(results[1])


fast = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, triangular_inequality=False, logs=False)

tau = phyloes.get_tau_tensors(adj_mats).to('cpu')
fast.update_topology(tau, batch_size)
fast.solve_timed()

print("old fastme with parallel file time", fast.time)
