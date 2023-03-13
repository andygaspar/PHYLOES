import torch
@torch.jit.script
def spr_side_script(intersections, sub_to_taxon, a_side: int, neighbors, set_to_adj, adj_to_set, T, subtrees_mat, n_taxa: int, subtree_dist, powers):
    inter_neighbor = intersections * intersections[neighbors[:, a_side]]
    regrafts = torch.nonzero(inter_neighbor)

    x = regrafts[:, 0]
    b = regrafts[:, 1]
    a = neighbors[x, a_side]

    b_adj = set_to_adj[b]
    b_c = adj_to_set[b_adj[:, 1], b_adj[:, 0]]

    a_adj = set_to_adj[a]
    a_c = adj_to_set[a_adj[:, 1], a_adj[:, 0]]

    dist_ij = T[set_to_adj[x][:, 0], set_to_adj[b][:, 0]]

    h = 1 - subtrees_mat[x] - subtrees_mat[b] - subtrees_mat[a]
    h = h[:, :n_taxa]

    # LT = L(XA) + L(XB) + L(XB)
    xb = subtree_dist[x, b]
    xa = subtree_dist[x, a]
    diff_xb = xb * (1 - 1 / powers[dist_ij])
    diff_xa = xa * (1 - powers[dist_ij])

    # diff_bh = bh(1 - 1/2)
    diff_bh = (subtree_dist[b, b_c] - xb - subtree_dist[a, b]) / 2

    # diff_ah = ah(1 - 2)
    diff_ah = -(subtree_dist[a, a_c] - subtree_dist[a, b] - xa)

    p1 = powers[T[set_to_adj[x][:, 0], :n_taxa]]
    p2 = powers[T[set_to_adj[b][:, 1], :n_taxa]]

    diff_xh = (sub_to_taxon[x, :] * ((p1 - p2) / p1) * h).sum(dim=-1)

    diff_T = diff_xb + diff_xa + diff_bh + diff_xh + diff_ah
    max_side_val, max_side_idx = diff_T.max(0)

    max_side_move = regrafts[max_side_idx]


    return max_side_val, max_side_move, diff_T