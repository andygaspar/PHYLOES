import os
from ast import literal_eval
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from results import utils as ut

def compute_n_trees(iterations):
    if iterations < 5:
        return 64*iterations
    if iterations < 25:
        return 64*5 + (iterations - 5) * 32
    return 64*5 + 5 * 32 + (iterations - 25) * 16

def concat_dfs():
    files = []
    df = pd.DataFrame()
    for (dir_path, dir_names, file_names) in os.walk('results/'):
        files.extend(file_names)

    files = [f for f in files if f[0] == 'r' and f[-1] == 'v']
    for file in files:
        df = pd.concat([df, pd.read_csv('results/' + file)])

    df.to_csv('results/complete.csv', index_label=False, index=False)


def compute_rf_distance():
    df = pd.read_csv('results/complete.csv')
    fp_ds_dist, fr_ds_dist, rp_ds_dist = [], [], []

    for i in range(df.shape[0]):
        n_taxa = df.Taxa.iloc[i]
        fast_tj = literal_eval(df.fast_traj.iloc[i])
        rand_tj = literal_eval(df.rand_traj.iloc[i])
        phyloes_tj = literal_eval(df.phyloes_tj.iloc[i])
        fp_ds_dist.append(ut.compute_robinson_distance(fast_tj, phyloes_tj, n_taxa))
        fr_ds_dist.append(ut.compute_robinson_distance(fast_tj, rand_tj, n_taxa))
        rp_ds_dist.append(ut.compute_robinson_distance(rand_tj, phyloes_tj, n_taxa))

    df['fast_phy_ds_dist'] = fp_ds_dist
    df['fast_rand_ds_dist'] = fr_ds_dist
    df['phy_rand_ds_dist'] = rp_ds_dist

    df.to_csv('results/complete_dist.csv', index_label=False, index=False)


def define_dataset_and_dist():
    df = pd.read_csv('results/complete_dist.csv')
    dataset, dist = [], []

    for i in range(df.shape[0]):
        pb = df.Problem.iloc[i]
        problem = 'rdpii' if pb[0] == 'r' else 'zilla'
        dataset.append(problem)

        d = df.Problem.iloc[i][-5]
        ds = 'F81' if d == '1' else ('F84' if d == '4' else ('JC69' if d == '9' else 'K2P'))
        dist.append(ds)

    df['dataset'] = dataset
    df['distance'] = dist

    df["improvement"] = (df['fast_obj'] - df['phyloes_obj']) / df['fast_obj']
    df["improvement"] = (df['fast_obj'] - df['phyloes_obj']) / df['fast_obj']
    df.sort_values(by=['Taxa', 'distance', 'run'], inplace=True)

    df.to_csv('results/complete_dist.csv', index_label=False, index=False)


# concat_dfs()
# compute_rf_distance()
# define_dataset_and_dist()

df = pd.read_csv('results/complete_dist.csv')


df['Dataset'] = df['Taxa'].astype(str) + '\_' + df.dataset + ' ' + df.distance
df['n_trees'] = df.iterations.apply(compute_n_trees)
df.sort_values(by=['Dataset'], inplace=True)

df_avg = pd.DataFrame(columns=['dataset', 'fastme init', 'fastme obj', 'fastme time'
                               'phyloes avg obj', 'phyloes mode obj', 'phyloes avg std',
                               'ri avg obj', 'ri mode obj', 'ri avg std',
                               'fastme t',
                               'phyloes avg t', 'phyloes std t',
                               'ri avg t', 'ri std t',
                               'f/p avg rf', 'f/p mode rf', 'f/p std rf',
                               'f/ri avg rf', 'f/ri mode rf', 'f/ri std rf', 'phy n sol', 'ri n sol',
                               'p nni', 'p spr', 'ri nni', 'ri spr', 'n trees', 'iterations'])


# AVERAGE DATASET
i = 0
for ds in df.Dataset.unique():
    df_dist = df[df.Dataset == ds]
    fast_init = df_dist.fastMe_init.iloc[0]
    fast_obj = df_dist.fast_obj.iloc[0]
    fast_t = df_dist.fast_time.iloc[0]

    phy_n_solutions = df_dist.phyloes_obj.unique().shape[0]
    ri_n_solutions = df_dist.random_obj.unique().shape[0]

    p_avg_obj = df_dist.phyloes_obj.mean()
    p_mode_obj = df_dist.phyloes_obj.mode().iloc[0]
    p_std_obj = df_dist.phyloes_obj.std()

    ri_avg_obj = df_dist.random_obj.mean()
    ri_mode_obj = df_dist.random_obj.mode().iloc[0]
    ri_std_obj = df_dist.random_obj.std()

    p_avg_t = df_dist.phyloes_time.mean()
    p_std_t = df_dist.phyloes_time.std()

    ri_avg_t = df_dist.rand_time.mean()
    ri_std_t = df_dist.rand_time.std()

    p_f_avg_rf = df_dist.fast_phy_ds_dist.mean()
    p_f_mode_rf = df_dist.fast_phy_ds_dist.mode().iloc[0]
    p_f_std_rf = df_dist.fast_phy_ds_dist.std()

    r_f_avg_rf = df_dist.fast_rand_ds_dist.mean()
    r_f_mode_rf = df_dist.fast_rand_ds_dist.mode().iloc[0]
    r_f_std_rf = df_dist.fast_rand_ds_dist.std()

    n_trees = df_dist.n_trees.mean()
    iterations = df_dist.iterations.mean()

    p_nni, p_spr = [], []
    for i in range(10):
        p_nni.append(sum(literal_eval(df_dist.p_nni.iloc[i])))
        p_spr.append(sum(literal_eval(df_dist.p_spr.iloc[i])))
    p_nni = np.mean(p_nni)
    p_spr = np.mean(p_spr)
    ri_nni = df_dist.rf_nni.mean()
    ri_spr = df_dist.rf_spr.mean()

    df_avg = pd.concat([df_avg, pd.DataFrame({'dataset': ds, 'fastme init': fast_init,
                                              'fastme obj': fast_obj, 'fastme time': fast_t,
                                              'phyloes avg obj': p_avg_obj, 'phyloes mode obj': p_mode_obj,
                                              'phyloes avg std': p_std_obj,
                                              'ri avg obj': ri_avg_obj, 'ri mode obj': ri_mode_obj,
                                              'ri avg std': ri_std_obj,
                                              'fastme t': fast_t,
                                              'phyloes avg t': p_avg_t, 'phyloes std t': p_std_t,
                                              'ri avg t': ri_avg_t, 'ri std t': ri_std_t,
                                              'f/p avg rf': p_f_avg_rf, 'f/p mode rf': p_f_mode_rf,
                                              'f/p std rf': p_f_std_rf,
                                              'f/ri avg rf': r_f_avg_rf, 'f/ri mode rf': r_f_mode_rf,
                                              'f/ri std rf': r_f_std_rf, 'phy n sol': phy_n_solutions,
                                              'ri n sol': ri_n_solutions, 'p nni':p_nni, 'p spr':p_spr,
                                              'ri nni': ri_nni, 'ri spr': ri_spr,
                                              'n trees': n_trees, 'iterations': iterations}, index=[i])])
    i += 1

df_avg["p improvement \%"] = (df_avg['phyloes avg obj'] - df_avg['fastme obj']) / df_avg['fastme obj']
df_avg["r improvement \%"] = (df_avg['ri avg obj'] - df_avg['fastme obj']) / df_avg['fastme obj']

df_avg.sort_values(by=['dataset'], inplace=True)

# OBJ VAL ANALYSIS

df_avg_1 = df_avg[['dataset', 'fastme init', 'fastme obj', 'phyloes avg obj', 'phyloes avg std', 'phy n sol', 'p improvement \%',
                   'ri avg obj', 'ri avg std', 'ri n sol', 'r improvement \%']]
print(df_avg_1.to_latex(index=False))



# RF DISTANCE ANALYSIS

df_avg_2 = df_avg[['dataset', 'f/p avg rf', 'f/p std rf', 'phy n sol']]
print(df_avg_2.to_latex(index=False))



# TIME ANALYSIS
df_avg3 = df_avg[['dataset', 'n trees', 'phyloes avg t',
                  'phyloes std t', 'iterations', 'p nni', 'p spr', 'ri avg t', 'ri std t', 'ri nni', 'ri spr' ,'fastme time',]]


df_avg3 = df_avg3.style.format(decimal=',', thousands='.', precision=1).hide(axis="index")

print(df_avg3.to_latex())



# PHYLOES SOLUTION RF DISTANCE ANALYSIS

for ds in df.Dataset.unique():
    df_dist = df[df.Dataset == ds]
    print(df_dist.phyloes_obj.unique())
    print(df_dist.phyloes_obj.value_counts())


avg_rf_phyloes_dist = []
std_rf_phyloes_dist = []

for dataset in df.Dataset.unique():
    df_dist = df[df.Dataset == dataset]
    # sols = df_dist.phyloes_tj.unique()
    # n_taxa = df_dist.Taxa.iloc[0]
    # print(n_taxa, dataset)
    # print(df_dist.phyloes_obj.unique().shape[0])
    # if len(sols) > 1:
    #     dists = []
    #     print(len(sols), list(combinations(range(len(sols)), 2)))
    #     for comb in combinations(range(len(sols)), 2):
    #         tree1 = literal_eval(sols[comb[0]])
    #         tree2 = literal_eval(sols[comb[1]])
    #         dists.append(rd.compute_robinson_distance(tree1, tree2, n_taxa))
    #     print(dists)
    #     avg_rf_phyloes_dist.append(np.mean(dists))
    #     std_rf_phyloes_dist.append(np.std(dists))
    # else:
    #     avg_rf_phyloes_dist.append(0)
    #     std_rf_phyloes_dist.append(0)


