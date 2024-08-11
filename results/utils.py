import numpy as np
from ete3 import Tree

from Solvers.FastME.pharser_newik.newwik_handler import compute_newick
from Solvers.solver import Solver


def get_adj_and_T_from_code(code, n_taxa):
    solver = Solver()
    solver.n_taxa = n_taxa
    solver.m = solver.n_taxa * 2 - 2

    adj_mat = solver.initial_adj_mat().astype(int)
    for step in range(3, solver.n_taxa):
        idx = np.array(np.nonzero(np.triu(adj_mat))).T[code[step - 3]]
        adj_mat = solver.add_node(adj_mat, idx, step, solver.n_taxa)

    T = solver.get_tau(adj_mat)
    return adj_mat, T
def get_ete_tree(code, n_taxa):
    T = get_adj_and_T_from_code(code, n_taxa)
    return compute_newick(T)

def compute_robinson_distance(code1, code2, n_taxa):
    tree1 = Tree(get_ete_tree(code1, n_taxa))
    tree2 = Tree(get_ete_tree(code2, n_taxa))
    res = tree1.robinson_foulds(tree2, unrooted_trees=True)
    return res[0]








