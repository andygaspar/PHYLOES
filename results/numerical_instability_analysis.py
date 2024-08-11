from ast import literal_eval

import numpy as np
import pandas as pd
from Solvers.solver import Solver
from results import utils as ut


class Problem:
    def __init__(self, df, dataset, dist, n_taxa):

        self.dataset = dataset
        self.dist = dist
        self.n_taxa = n_taxa
        self.fast_objs, self.phy_objs = [], []

        self.runs = 10

        self.fastme_T, self.phy_T = [], []

        self.d = np.loadtxt('Data_/mats/' + dataset + '_' + dist + '.csv')[:n_taxa, :n_taxa]
        self.solver = Solver(self.d)

        self.error = self.compute_error()

        df_dataset = df[(df.dataset == dataset) & (df.distance == dist) & (df.Taxa == n_taxa)]
        self.df = df_dataset

        fastme_codes = df_dataset.fast_traj.values
        phy_codes = df_dataset.phyloes_tj.values

        # for i in range(fastme_codes.shape[0]):
        #     fast_tj = literal_eval(fastme_codes[i])
        #     phy_tj = literal_eval(phy_codes[i])
        #     _, fT = ut.get_adj_and_T_from_code(fast_tj, len(fast_tj) + 3)
        #     _, pT = ut.get_adj_and_T_from_code(phy_tj, len(phy_tj) + 3)
        #     self.fastme_T.append(fT)
        #     self.phy_T.append(pT)

    def compute_objs(self):
        for i in range(self.runs):
            self.solver.T = self.fastme_T[i]
            self.fast_objs.append(self.solver.compute_obj())

            self.solver.T = self.phy_T[i]
            self.phy_objs.append(self.solver.compute_obj())

    def compute_partial_objs(self, start=0, end=None):
        end = self.n_taxa if end is None else end
        bool_mats_fast = [((start <= self.fastme_T[i]) & ( self.fastme_T[i] < end)) for i in range(self.runs)]
        bool_mats_phy = [((start <= self.phy_T[i]) & ( self.phy_T[i] < end)) for i in range(self.runs)]

        fast_mat = [self.solver.d * self.solver.np_powers[self.fastme_T[i][:self.n_taxa, :self.n_taxa]] for i in range(self.runs)]
        phy_mat = [self.solver.d * self.solver.np_powers[self.phy_T[i][:self.n_taxa, :self.n_taxa]]for i in range(self.runs)]

        return [fast_mat[i][bool_mats_fast[i]].sum() for i in range(self.runs)], \
            [phy_mat[i][bool_mats_phy[i]].sum() for i in range(self.runs)]

    def compute_error(self):
        return self.n_taxa ** 2 * 2 ** (-49) * np.max(self.d)


df = pd.read_csv('results/complete_dist.csv')
df['Dataset'] = df['Taxa'].astype(str) + '\_' + df.dataset + ' ' + df.distance
df.sort_values(by=['Dataset'], inplace=True)

problems = {}
df_error = pd.DataFrame(columns=['taxa', 'dataset', 'F81', 'F84', 'JC69', 'K2P'])
i = 0
for taxa in df.Taxa.unique():
    df_t = df[df.Taxa == taxa]
    for ds in df_t.dataset.unique():
        df_ds = df_t[df_t.dataset == ds]
        errors = []
        print(df_ds.distance.unique())
        for dist in df_ds.distance.unique():
            df_dist = df_ds[df_ds.distance == dist]
            p = Problem(df, ds, dist, taxa)
            errors.append(p.error)
        df_error = pd.concat([df_error, pd.DataFrame({'taxa': taxa, 'dataset': ds ,
                                         'F81': errors[0], 'F84': errors[1], 'JC69': errors[2], 'K2P': errors[3]}, index=[i])],
                             )
        i += 1
func = lambda s: s if isinstance(s, str) or isinstance(s, int) else '{:.2E}'.format(s)
print(df_error.style.format(func).hide(axis="index").to_latex())




