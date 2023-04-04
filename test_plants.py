import os
import random

import numpy as np
import pandas as pd
import torch

from Solvers.FastME.fast_me import FastMeSolver
from Solvers.PhyloES.phyloes_cpp import PhyloEScpp
from Solvers.PhyloES.phyloes_parallel import PhyloES2
from Solvers.RandomFastME.random_fast_me import RandomFastME
from Solvers.PhyloES.phyloes import PhyloES
from Data_.data_loader import DistanceData
from Solvers.RandomFastME.random_fast_me_cpp import RandomFastMEcpp

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


j = 0

'''
result 32

310477379381748
309760471457418

phyloes CPP  time: 3511.304529428482    obj: 9.309610894796347    n_trees 2736   stop convergence
5.350451748033435
'''



for problem in range(problems):


    d = np.around(data_set.get_minor(250), 20)
    d = d/np.max(d)

    batch = [(0, 64), (5, 32), (25, 16)]

    for max_iter in [500]:
        for run in range(run_per_problem):
            print(d.shape[0], batch, max_iter, run)
            result_run = [data_set_idx, d.shape[0], problem]
            run_list.append(run)
            batch_list.append(batch)
            max_iter_list.append(max_iter)


            random.seed(j)
            phyloes = PhyloEScpp(d, batch=batch, max_iterations=max_iter)
            phyloes.solve_timed()
            print("phyloes CPP  time:", phyloes.time, '   obj:', phyloes.obj_val, '   n_trees',
                  phyloes.n_trees, '  stop', phyloes.stop_criterion)
            phyloes_tj = tuple(phyloes.tree_climb(torch.tensor(phyloes.solution).unsqueeze(0)).to('cpu').tolist()[0])
            stop_list.append(phyloes.stop_criterion)

            # random.seed(j)
            # phyloes = PhyloEScpp(d, batch=batch, max_iterations=max_iter, replace=True)
            # phyloes.solve_timed()
            # print("phyloes CPP replace time:", phyloes.time, '   obj:', phyloes.obj_val, '   n_trees',
            #       phyloes.n_trees, '  stop', phyloes.stop_criterion)

            random.seed(j)
            rand_fast = RandomFastMEcpp(d, parallel=False, spr=True)
            rand_fast.solve_timed(phyloes.n_trees)
            print("rand_fa  time:", rand_fast.time, '   obj:', rand_fast.obj_val, '   n_trees',
                  phyloes.n_trees)
            rand_fast_tj = tuple(phyloes.tree_climb(torch.tensor(rand_fast.solution).unsqueeze(0)).to('cpu').tolist()[0])

            #
            fast = FastMeSolver(d, bme=True, nni=True, digits=17, bootrstap=True, post_processing=True, triangular_inequality=False,
                                logs=False)
            fast.solve_all_flags()
            print("fast  time:",  fast.time, '   obj:', fast.obj_val)
            fast_tj = tuple(phyloes.tree_climb(torch.tensor(fast.solution).unsqueeze(0)).to('cpu').tolist()[0])
            # print(fast_tj)
            result_run += [fast.obj_val, rand_fast.obj_val,  phyloes.obj_val, fast.time, rand_fast.time,
                           phyloes.time, fast.method, rand_fast.nni_counter, rand_fast.spr_counter, phyloes.nni_counter, phyloes.spr_counter]
            result_run += [fast_tj, rand_fast_tj, phyloes_tj]
            result_list.append(result_run)

            j += 1




df = pd.DataFrame(result_list, columns=['Dataset', 'Taxa', 'Problem', 'fast_obj', 'random_obj', 'phyloes_obj', 'fast_time',
                                            'rand_time', 'phyloes_time', 'fastMe_init', 'rf_nni', 'rf_spr', 'p_nni', 'p_spr', 'fast_traj', 'rand_traj',
                                            'phyloes_tj'])
df['run'] = run_list
df['batch'] = batch_list
df['max_iter'] = max_iter_list
df['stop'] = stop_list
df.Dataset = df.Dataset.apply(lambda el: dataset_names[el])

print(df)
# df.to_csv('plants_results_zilla.csv', index_label=False, index=False)
# df["fast_improvement"] = df['fast_obj']/df['td_fast_obj'] - 1
# df['random_improvement'] = df['random_obj']/df['td_fast_obj'] - 1
# df.to_csv('test_td_fast2.csv', index_label=False, index=False)
# results = np.array(results)
# df = pd.DataFrame({"mcts": results[:, 0], "lp_nj": results[:, 1], "fast": results[:, 2]})
# df.to_csv("test_new.csv", index_label=False, index=False)






# data_set = distances.get_dataset(3)
#
# d = data_set.get_full_dataset()
# fast = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, triangular_inequality=False,
#                     logs=False)
# fast.solve()