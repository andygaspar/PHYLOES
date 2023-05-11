import os
import random

import numpy as np
import pandas as pd
import torch

from Solvers.FastME.fast_me import FastMeSolver
from Solvers.PhyloES.phyloes import PhyloES

from Solvers.RI.ri import RI

torch.set_printoptions(precision=16)


np.random.seed(0)
random.seed(0)
seed = 0

max_iter = 10_000
batch = [(0, 64), (5, 32), (25, 16)]

files = []
for (dir_path, dir_names, file_names) in os.walk('Data_/benchmarks/matrices/experiment_mats'):
    files.extend(file_names)

files = sorted(files)
print(files)


for file in files[4:8]:
    print(file)
    data_set_list = []
    run_list, batch_list, max_iter_list, stop_list, n_better_solution = [], [], [], [], []

    result_list, best_list, worse_list, iterations = [], [], [], []

    run_per_problem = 1

    d = np.abs(np.around(np.loadtxt('Data_/benchmarks/matrices/experiment_mats/' + file), 10))

    for run in range(run_per_problem):
        print('\n', d.shape[0], batch, max_iter, run)
        result_run = [d.shape[0], file]
        run_list.append(run)
        batch_list.append(batch)
        max_iter_list.append(max_iter)

        random.seed(seed)
        phyloes = PhyloES(d, batch=batch, max_iterations=max_iter, replace=True)
        phyloes.solve_timed()
        print("phyloes CPP  time:", phyloes.time, '   obj:', phyloes.obj_val, '   n_trees',
              phyloes.n_trees, '  stop', phyloes.stop_criterion)
        phyloes_tj = tuple(phyloes.tree_climb(torch.tensor(phyloes.solution).unsqueeze(0)).to('cpu').tolist()[0])
        stop_list.append(phyloes.stop_criterion)
        best_list.append(phyloes.best_vals)
        worse_list.append(phyloes.worse_vals)
        iterations.append(phyloes.iterations)

        rand_fast = RI(d, parallel=False, spr=True)
        rand_fast.solve_timed(phyloes.n_trees)
        print("rand_fa  time:", rand_fast.time, '   obj:', rand_fast.obj_val, '   n_trees',
              phyloes.n_trees)
        rand_fast_tj = tuple(phyloes.tree_climb(torch.tensor(rand_fast.solution).unsqueeze(0)).to('cpu').tolist()[0])

        fast = FastMeSolver(d, bme=True, nni=True, digits=17, bootrstap=True, post_processing=True, triangular_inequality=False,
                            logs=False)
        fast.solve_all_flags()
        print("fast  time:",  fast.time, '   obj:', fast.obj_val)
        fast_tj = tuple(phyloes.tree_climb(torch.tensor(fast.solution).unsqueeze(0)).to('cpu').tolist()[0])
        print(fast_tj)
        result_run += [fast.obj_val, rand_fast.obj_val,  phyloes.obj_val, fast.time, rand_fast.time,
                       phyloes.time, fast.method, rand_fast.nni_counter, rand_fast.spr_counter, phyloes.nni_counter, phyloes.spr_counter]
        result_run += [fast_tj, rand_fast_tj, phyloes_tj]
        result_list.append(result_run)

        seed += 1

    df = pd.DataFrame(result_list, columns=['Taxa', 'Problem', 'fast_obj', 'random_obj', 'phyloes_obj', 'fast_time',
                                                'rand_time', 'phyloes_time', 'fastMe_init', 'rf_nni', 'rf_spr', 'p_nni', 'p_spr', 'fast_traj', 'rand_traj',
                                                'phyloes_tj'])
    df['run'] = run_list
    df['batch'] = batch_list
    df['max_iter'] = max_iter_list
    df['stop'] = stop_list
    df['iterations'] = iterations
    df['best_list'] = best_list
    df['worst_list'] = worse_list


# df.to_csv('results/results' + file + '.csv', index_label=False, index=False)
# df["fast_improvement"] = df['fast_obj']/df['td_fast_obj'] - 1
# df['random_improvement'] = df['random_obj']/df['td_fast_obj'] - 1
# df.to_csv('test_td_fast2.csv', index_label=False, index=False)
# results = np.array(results)
# df = pd.DataFrame({"mcts": results[:, 0], "lp_nj": results[:, 1], "fast": results[:, 2]})
# df.to_csv("test_new.csv", index_label=False, index=False)




