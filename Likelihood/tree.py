import io
import random

import Bio
import ete3
import networkx as nx
import numpy as np
from Bio import Phylo
from Bio.Phylo.Newick import Tree, Clade
from matplotlib import pyplot as plt


class PhyloTree:
    def __init__(self, n_taxa, alignment_file, labels):
        self.adj = None
        self.n_taxa = n_taxa
        self.m = 2 * self.n_taxa - 2
        self.labels = labels + [str(i) for i in range(self.n_taxa, self.m)]
        self.nx_tree = None
        self.newick_tree = None
        self.alignment_file = alignment_file
        self.tree_file = "tree.newick"

    def set_random_tree(self):

        # the order of the edges matters for the recursive construction of the tree
        edges = [(self.n_taxa, 0, 1.0), (self.n_taxa, 1, 1.0), (self.n_taxa, 2, 1.0)]

        for i in range(3, self.n_taxa):
            u, v, _ = random.choice(list(edges))
            edges.remove((u, v, 1.0))
            edges.append((self.n_taxa + i - 2, i, 1.0))
            edges.append((u, self.n_taxa + i - 2, 1.0))
            edges.append((self.n_taxa + i - 2, v, 1.0))

        self.adj = np.zeros((2 * self.n_taxa - 2, 2 * self.n_taxa - 2), dtype=int)
        for idx in edges:
            self.adj[idx[0], idx[1]] = self.adj[idx[1], idx[0]] = 1

        self.set_tree(edges)

    def set_tree_from_adj(self, adj: np.ndarray):
        children = np.nonzero(adj[self.n_taxa])[0]
        edges = []
        for child in children:
            self.set_recursive(self.n_taxa, child, edges, adj)

        self.set_tree(edges)

    def set_recursive(self, parent, child, edges, adj):
        edges.append((parent, child, 1.0))
        children = np.nonzero(adj[child])[0]
        for new_child in children:
            if new_child != parent:
                self.set_recursive(child, new_child, edges, adj)

    def set_tree(self, edges):
        # nx_tree = nx.from_numpy_array(self.adj)
        self.nx_tree = nx.DiGraph()
        self.nx_tree.add_weighted_edges_from(edges)
        self.nx_tree = nx.relabel_nodes(self.nx_tree, dict(zip(range(self.m), self.labels)))
        phylo_tree = self.nx_to_phylo(self.nx_tree, self.labels[self.n_taxa])

        with open(self.tree_file, 'w') as file:
            Phylo.write(phylo_tree, file, format='newick')

    def build_clade(self, G, node):
        """Recursively build a Clade structure from a given node."""
        clade = Clade(name=node)
        children = list(G.successors(node))
        if children:
            clade.clades = [self.build_clade(G, child) for child in children]
            for child in children:
                # Set branch length from the graph edge weight
                for cl in clade.clades:
                    if cl.name == child:
                        cl.branch_length = G[node][child]['weight']
        return clade

    def nx_to_phylo(self, G, root):
        root_clade = self.build_clade(G, root)
        return Tree(root=root_clade, rooted=False)

    def show(self):
        nx.draw(self.nx_tree, with_labels=True)
        plt.show()