import csv
import os
import time
import warnings
from typing import List

import networkx as nx
import numpy as np
import torch

from matplotlib import pyplot as plt

from Solvers.solver import Solver
from Solvers.FastME.pharser_newik.newwik_handler import get_adj_from_nwk, compute_newick, compute_multiple_newick, \
    get_multiple_adj_from_nwk

warnings.simplefilter("ignore")


class FastMeSolver(Solver):

    def __init__(self, d,
                 bme=True, nni=True, digits=None, post_processing=False, bootrstap=False, init_topology=None,
                 triangular_inequality=False, logs=False, num_topologies=1, labels: List[str] = None):
        super().__init__(d, labels=labels)
        self.path = 'Solvers/FastME/'
        self.init_topology = init_topology
        self.flags = ''
        self.method = 'b' if bme else None
        self.nni = nni
        self.post_processing = post_processing
        self.digits = digits
        self.bootstrap = bootrstap
        self.triangular_inequality = triangular_inequality
        self.logs = logs
        self.num_topologies = num_topologies
        self.solve_time = None

    def solve(self):

        self.set_flags()

        # mettere tutte flag bene e controllare taxaaddbal
        self.write_d(self.num_topologies)

        if self.init_topology is not None:
            if self.num_topologies == 1:
                with open(self.path + 'init_topology.nwk', 'w', newline='') as csvfile:
                    csvfile.write(compute_newick(self.init_topology))

            else:
                with open(self.path + 'init_topology.nwk', 'w', newline='') as csvfile:
                    csvfile.write(compute_multiple_newick(self.init_topology))
        t = time.time()
        os.system(self.path + "fastme -i " + self.path + "mat.mat " + self.flags)
        self.solve_time = time.time() - t

        if self.num_topologies < 2:
            adj_mat = get_adj_from_nwk(self.path + 'mat.mat_fastme_tree.nwk')
            self.solution = adj_mat
            self.T = self.get_tau(self.solution)
            self.obj_val = self.compute_obj()
        else:
            adj_mats = get_multiple_adj_from_nwk(self.path + 'mat.mat_fastme_tree.nwk')
            self.solution = torch.tensor(adj_mats)
            self.obj_val = self.read_objs()
            # self.obj_val = self.compute_obj_val_batch(self.solution, self.d, self.)

    def set_flags(self):
        self.flags = ''
        if self.num_topologies > 1:
            self.flags += " -D " + str(self.num_topologies) + ' '
        if self.method is not None:
            self.flags += " -m " + self.method + ' '
        if self.nni:
            self.flags += " -n "
        if self.post_processing:
            self.flags += " -s "
        if self.triangular_inequality:
            self.flags += " -q "
        if self.digits is not None:
            self.flags += " -f " + str(self.digits) + " "
        if self.bootstrap:
            self.flags += " -b 100 "
        if self.init_topology is not None:
            self.flags += ' -u ' + self.path + 'init_topology.nwk'
        if not self.logs:
            self.flags += " > /dev/null"

    def write_d(self, num_topologies):
        d_string = ''
        for _ in range(num_topologies):
            d_string += str(self.n_taxa) + '\n'
            for i, row in enumerate(self.d):
                row_string = ['{:.19f}'.format(el) for el in row]
                line = str(i) + ' ' + ' '.join(row_string)
                d_string += line + '\n'
            d_string += '\n'

        with open(self.path + 'mat.mat', 'w', newline='') as csvfile:
            csvfile.write(d_string)

    def update_topology(self, init_topology, num_topologies=1):
        self.init_topology = init_topology
        self.num_topologies = num_topologies
        self.set_flags()

    def change_flags(self,
                     bme=True, nni=True, digits=None, post_processing=False, triangular_inequality=False, logs=False):
        self.flags = ''
        self.method = 'b' if bme else None
        self.nni = nni
        self.digits = digits
        self.post_processing = post_processing
        self.triangular_inequality = triangular_inequality
        self.logs = logs

    def check_mat(self, adj_mat):
        taxa = np.array_equal(adj_mat[:self.n_taxa, self.n_taxa:].sum(axis=1), np.ones(self.n_taxa))
        internals = np.array_equal(adj_mat[self.n_taxa:, :].sum(axis=1), np.ones(self.n_taxa) * 3)
        graph = nx.from_numpy_matrix(adj_mat)
        pos = nx.spring_layout(graph)

        nx.draw(graph, pos=pos, node_color=['green' if i < self.n_taxa else 'red' for i in range(self.m)],
                with_labels=True, font_weight='bold')
        plt.show()
        return taxa * internals

    def solve_all_flags(self):
        self.time = time.time()
        best_val, best_sol, best_method = 100, None, None
        for method in ['b', 'o', 'i', 'n', 'u']:
            self.method = method
            self.solve()
            # print(method, self.obj_val)
            if self.obj_val < best_val:
                best_val, best_sol, best_method = self.obj_val, self.solution, method
        self.time = time.time() - self.time

        self.obj_val = best_val
        self.solution = best_sol
        self.method = best_method
        self.T = self.get_tau(self.solution)

    def read_objs(self):
        file = self.path + 'mat.mat_fastme_stat.txt'
        obj_vals = []
        vals = []
        with open(file, newline='') as csvfile:
            reader = list(csv.reader(csvfile, delimiter='\n'))
            for i, row in enumerate(reader):
                if len(row) > 0:
                    if row[0][:3] == '\tPe':
                        vals.append(float(reader[i - 1][0][32:]))
                    if row[0][:3] == '\tEx':
                        obj_vals.append(min(vals))
                        vals = []
        return obj_vals
