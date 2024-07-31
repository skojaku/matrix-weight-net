import numpy as np
from numba import njit
from scipy import sparse


class RandomWalkNodeSampler:
    """Node Sampler based on the randomWalk."""

    def __init__(
        self,
        walk_length=5,
        p=1.0,
        q=1.0,
        restart_prob=0,
    ):
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.restart_prob = restart_prob

    def fit(self, A):
        """Initialize the random walk sampler."""
        self.num_nodes = A.shape[0]
        self.p0 = np.array(A.sum(axis=1).astype(float)).reshape(-1) / A.sum()

        is_weighted = np.max(A.data) != np.min(A.data)
        A.sort_indices()
        if is_weighted:
            data = A.data / A.sum(axis=1).A1.repeat(np.diff(A.indptr))
            A.data = _csr_row_cumsum(A.indptr, data)
        self.A = A
        self.is_weighted = is_weighted
        self.n_nodes = A.shape[0]

    def sampling(self, size=None, center_nodes=None, edge_seq=False):

        if edge_seq:
            return self.sampling_edge_seq(size, center_nodes)
        else:
            return self.sampling_node_seq(size, center_nodes)

    def sampling_node_seq(self, size=None, center_nodes=None):
        if center_nodes is None:
            center_nodes = np.random.choice(
                self.n_nodes, size=size, p=self.p0, replace=True
            )
        context_nodes = np.zeros_like(center_nodes)
        for i, center_node in enumerate(center_nodes):
            if self.is_weighted:
                walk = _random_walk_weighted(
                    self.A.indptr,
                    self.A.indices,
                    self.A.data,
                    self.walk_length,
                    self.p,
                    self.q,
                    self.restart_prob,
                    center_node,
                )
            else:
                walk = _random_walk(
                    self.A.indptr,
                    self.A.indices,
                    self.walk_length,
                    self.p,
                    self.q,
                    self.restart_prob,
                    center_node,
                )

            context_nodes[i] = walk[np.random.randint(0, len(walk))]
        return center_nodes.astype(np.int64), context_nodes.astype(np.int64)

    def sampling_edge_seq(self, size=None, center_nodes=None):
        if center_nodes is None:
            center_nodes = np.random.choice(
                self.n_nodes, size=size, p=self.p0, replace=True
            )
        retvals = []
        for i, center_node in enumerate(center_nodes):
            if self.is_weighted:
                walk = _random_walk_weighted_edge_seq(
                    self.A.indptr,
                    self.A.indices,
                    self.A.data,
                    self.walk_length,
                    self.p,
                    self.q,
                    self.restart_prob,
                    center_node,
                )
            else:
                walk = _random_walk_edge_seq(
                    self.A.indptr,
                    self.A.indices,
                    self.walk_length,
                    self.p,
                    self.q,
                    self.restart_prob,
                    center_node,
                )
            retvals.append(walk)
        return center_nodes, retvals

    def sampling_source_nodes(self, size):
        return np.random.choice(self.num_nodes, size=size, p=self.p0, replace=True)


def csr_sampling(rows, csr_mat):
    return _csr_sampling(rows, csr_mat.indptr, csr_mat.indices, csr_mat.data)


@njit(nogil=True)
def _neighbors(indptr, indices_or_data, t):
    return indices_or_data[indptr[t] : indptr[t + 1]]


@njit(nogil=True)
def _csr_sampling(rows, indptr, indices, data):
    n = len(rows)
    retval = np.empty(n, dtype=indices.dtype)
    for j in range(n):
        neighbors = _neighbors(indptr, indices, rows[j])
        neighbors_p = _neighbors(indptr, data, rows[j])
        retval[j] = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
    return retval


# Random walk utilities
@njit(nogil=True)
def _csr_row_cumsum(indptr, data):
    out = np.empty_like(data)
    for i in range(len(indptr) - 1):
        acc = 0
        for j in range(indptr[i], indptr[i + 1]):
            acc += data[j]
            out[j] = acc
        out[j] = 1.0
    return out


@njit(nogil=True)
def _neighbors(indptr, indices_or_data, t):
    return indices_or_data[indptr[t] : indptr[t + 1]]


@njit(nogil=True)
def _isin_sorted(a, x):
    return a[np.searchsorted(a, x)] == x


@njit(nogil=True)
def _random_walk(indptr, indices, walk_length, p, q, restart_prob, t):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = np.empty(walk_length, dtype=indices.dtype)
    walk[0] = t
    neighbors = _neighbors(indptr, indices, t)
    if not neighbors.size:
        return walk[:1]
    walk[0] = np.random.choice(neighbors)
    for j in range(1, walk_length):
        if np.random.rand() < restart_prob:
            neighbors = _neighbors(indptr, indices, t)
        else:
            neighbors = _neighbors(indptr, indices, walk[j - 1])

        if not neighbors.size:
            return walk[:j]

        if p == q == 1:
            # faster version
            walk[j] = np.random.choice(neighbors)
            continue

        while True:
            new_node = np.random.choice(neighbors)
            r = np.random.rand()
            if new_node == walk[j - 2]:
                if r < prob_0:
                    break
            elif _isin_sorted(_neighbors(indptr, indices, walk[j - 2]), new_node):
                if r < prob_1:
                    break
            elif r < prob_2:
                break
        walk[j] = new_node
    return walk


@njit(nogil=True)
def _random_walk_weighted(indptr, indices, data, walk_length, p, q, restart_prob, t):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = np.empty(walk_length, dtype=indices.dtype)
    walk[0] = t
    neighbors = _neighbors(indptr, indices, t)
    if not neighbors.size:
        return walk[:1]
    walk[0] = _neighbors(indptr, indices, t)[
        np.searchsorted(_neighbors(indptr, data, t), np.random.rand())
    ]
    for j in range(1, walk_length):
        if np.random.rand() < restart_prob:
            neighbors = _neighbors(indptr, indices, t)
            if not neighbors.size:
                return walk[:j]
            neighbors_p = _neighbors(indptr, data, t)
        else:
            neighbors = _neighbors(indptr, indices, walk[j - 1])
            if not neighbors.size:
                return walk[:j]
            neighbors_p = _neighbors(indptr, data, walk[j - 1])

        if p == q == 1:
            # faster version
            walk[j] = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
            continue
        while True:
            new_node = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
            r = np.random.rand()
            if new_node == walk[j - 2]:
                if r < prob_0:
                    break
            elif _isin_sorted(_neighbors(indptr, indices, walk[j - 2]), new_node):
                if r < prob_1:
                    break
            elif r < prob_2:
                break
        walk[j] = new_node
    return walk


# ==========


@njit(nogil=True)
def _random_walk_edge_seq(indptr, indices, walk_length, p, q, restart_prob, t):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk_edge_ids = np.empty(walk_length - 1, dtype=indices.dtype)
    neighbors = _neighbors(indptr, indices, t)
    if not neighbors.size:
        return walk_edge_ids
    nei_id = np.random.choice(len(neighbors))
    walk_edge_ids[0] = indptr[t] + nei_id
    prev_node = t
    current_node = neighbors[nei_id]

    for j in range(1, walk_length):
        if np.random.rand() < restart_prob:
            neighbors = _neighbors(indptr, indices, t)
        else:
            neighbors = _neighbors(indptr, indices, current_node)

        if not neighbors.size:
            return walk_edge_ids[:j]

        if p == q == 1:
            # faster version
            nei_id = np.random.choice(len(neighbors))
            walk_edge_ids[j] = indptr[current_node] + nei_id
            prev_node = current_node
            current_node = neighbors[nei_id]
            continue

        while True:
            new_node_id = np.random.choice(len(neighbors))
            new_node = neighbors[new_node_id]
            r = np.random.rand()
            if new_node == prev_node:
                if r < prob_0:
                    break
            elif _isin_sorted(_neighbors(indptr, indices, prev_node), new_node):
                if r < prob_1:
                    break
            elif r < prob_2:
                break
        walk_edge_ids[j] = indptr[current_node] + new_node_id
        prev_node = current_node
        current_node = new_node
    return walk_edge_ids


@njit(nogil=True)
def _random_walk_weighted_edge_seq(
    indptr, indices, data, walk_length, p, q, restart_prob, t
):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk_edge_ids = np.empty(walk_length - 1, dtype=indices.dtype)
    neighbors = _neighbors(indptr, indices, t)
    if not neighbors.size:
        return walk_edge_ids
    nei_id = np.searchsorted(_neighbors(indptr, data, t), np.random.rand())
    walk_edge_ids[0] = indptr[t] + nei_id
    prev_node = t
    current_node = neighbors[nei_id]

    for j in range(1, walk_length):
        if np.random.rand() < restart_prob:
            neighbors = _neighbors(indptr, indices, t)
            neighbors_p = _neighbors(indptr, data, t)
        else:
            neighbors = _neighbors(indptr, indices, current_node)
            neighbors_p = _neighbors(indptr, data, current_node)

        if not neighbors.size:
            return walk_edge_ids[:j]

        if p == q == 1:
            # faster version
            nei_id = np.searchsorted(neighbors_p, np.random.rand())
            walk_edge_ids[j] = indptr[current_node] + nei_id
            prev_node = current_node
            current_node = neighbors[nei_id]
            continue

        while True:
            new_node_id = np.searchsorted(neighbors_p, np.random.rand())
            new_node = neighbors[new_node_id]
            r = np.random.rand()
            if new_node == prev_node:
                if r < prob_0:
                    break
            elif _isin_sorted(_neighbors(indptr, indices, prev_node), new_node):
                if r < prob_1:
                    break
            elif r < prob_2:
                break
        walk_edge_ids[j] = indptr[current_node] + new_node_id
        prev_node = current_node
        current_node = new_node
    return walk_edge_ids
