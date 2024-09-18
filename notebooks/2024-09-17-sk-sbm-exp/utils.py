import numpy as np
from itertools import combinations
import networkx as nx
from scipy import sparse


def generate_random_rotation_matrix(k):
    """
    Generate a random rotation matrix.

    Parameters
    ----------
    k: int
        Dimension of the matrix-weighted SBM.

    Returns
    -------
    A: numpy.ndarray
        Random rotation matrix.

    """
    A = np.eye(k)
    planes = list(combinations(range(k), 2))
    np.random.shuffle(planes)
    for i, j in planes:
        val = np.random.uniform(-1, 1)
        th = val * np.pi
        # print("planes: ({}, {}), angle: {} pi".format(i, j, val))
        R = np.eye(k)
        R[i, i] = np.cos(th)
        R[i, j] = -np.sin(th)
        R[j, i] = np.sin(th)
        R[j, j] = np.cos(th)
        A = R @ A
    return A


def generate_rotation_noise_matrix(k, noise=0.0):
    """
    Generate a rotation matrix with noise.

    Parameters
    ----------
    k: int
        Dimension of the matrix-weighted SBM.
    noise: float
        Noise parameter.

    Returns
    -------
    A: numpy.ndarray
        Rotation matrix with noise.

    """
    A = np.eye(k)
    planes = list(combinations(range(k), 2))
    np.random.shuffle(planes)
    for i, j in planes:
        val = np.random.normal(scale=noise)
        th = val * np.pi
        # print("planes: ({}, {}), noise angle: {} pi".format(i, j, val))
        R = np.eye(k)
        R[i, i] = np.cos(th)
        R[i, j] = -np.sin(th)
        R[j, i] = np.sin(th)
        R[j, j] = np.cos(th)
        A = R @ A
    return A


def assign_rotation_matrix(
    src, trg, membership, com_com_rotation_matrix, coherence, noise, dim
):
    """
    Assign a rotation matrix to an edge.

    Parameters
    ----------
    src: int
        Source node.
    trg: int
        Target node.
    membership: numpy.ndarray
        Community assignment of each node.
    com_com_rotation_matrix: dict
        Rotation matrix for each pair of communities.
    coherence: float
        Coherence parameter.
    noise: float
        Noise parameter.
    dim: int
        Dimension of the matrix-weighted SBM.

    Returns
    -------
    rotation_matrix: numpy.ndarray
        Rotation matrix for the edge.

    """
    k, l = membership[src], membership[trg]
    if np.random.rand() < coherence:
        return (
            com_com_rotation_matrix[(k, l)]
            if (k, l) in com_com_rotation_matrix
            else np.zeros((dim, dim))
        )  # assign the community to community matrix
    else:
        return (
            generate_rotation_noise_matrix(k=dim, noise=noise).dot(
                com_com_rotation_matrix[(k, l)]
            )
            if (k, l) in com_com_rotation_matrix
            else np.zeros((dim, dim))
        )


def generate_matrix_weighted_sbm(
    n_nodes,
    n_communities,
    pin,
    pout,
    coherence,
    noise,
    dim,
):
    """
    Generate a matrix-weighted SBM.

    Parameters
    ----------
    n_nodes: int
        Number of nodes.
    n_communities: int
        Number of communities.
    pin: float
        Probability of intra-community edges.
    pout: float
        Probability of inter-community edges.
    coherence: float
        Coherence parameter.
    noise: float
        Noise parameter.
    dim: int
        Dimension of the matrix-weighted SBM.

    Returns
    -------
    A_mat: scipy.sparse.csr_matrix
        Matrix-weighted SBM.
    A: scipy.sparse.csr_matrix
        Base network
    membership: numpy.ndarray
        Community assignment of each node.
    com_com_rotation_matrix: dict
        Rotation matrix for each pair of communities.
    """
    assert n_nodes % n_communities == 0, "n_nodes must be divisible by n_communities"

    # Generate a base network using the stochastic block model
    pref_matrix = np.full((n_communities, n_communities), pout)
    pref_matrix[np.diag_indices_from(pref_matrix)] = pin
    block_sizes = [n_nodes // n_communities] * n_communities

    g = nx.stochastic_block_model(block_sizes, pref_matrix, seed=0, directed=False)
    A = nx.to_scipy_sparse_array(g)

    src, trg, _ = sparse.find(sparse.triu(A, 1))  # only edges
    membership = np.digitize(np.arange(n_nodes), np.cumsum(block_sizes))

    # intially balanced configuration
    # internal edges - identity matrix
    com_com_rotation_matrix = {(l, l): np.eye(dim) for l in range(n_communities)}
    # external edges - first path random
    com_com_rotation_matrix.update(
        {
            (l, k): generate_random_rotation_matrix(k=dim)
            for l in range(n_communities)
            for k in range(l + 1, n_communities)
        }
    )
    com_com_rotation_matrix.update(
        {
            (k, l): com_com_rotation_matrix[(l, k)].T
            for l in range(n_communities)
            for k in range(l + 1, n_communities)
        }
    )
    # external edges - others by existing ones
    # if n_communities == 2:
    #    com_com_rotation_matrix[(1, 0)] = com_com_rotation_matrix[0, 1].T

    # construct the block weight matrix
    matrix_blocks = [[None for _ in range(n_nodes)] for _ in range(n_nodes)]

    for k, l in zip(src, trg):
        rot = assign_rotation_matrix(
            k, l, membership, com_com_rotation_matrix, coherence, noise, dim
        )
        matrix_blocks[k][l] = rot
        matrix_blocks[l][k] = rot.T

    A_mat = sparse.bmat(matrix_blocks, format="csr")
    return A_mat, A, membership, com_com_rotation_matrix


def generate_initial_characteristic_vector(nodes, n_nodes, dim):
    """
    Generate an initial characteristic vector.

    Parameters
    ----------
    nodes: numpy.ndarray
        Nodes to be assigned the characteristic vector.
    n_nodes: int
        Number of nodes.
    dim: int
        Dimension of the matrix-weighted SBM.

    Returns
    -------
    y_0: scipy.sparse.csc_matrix
        Initial characteristic vector.
    """
    # inital characteristic vector for nodes
    # s1 - choose a random position, and assign to nodes in one community
    n_sel = len(nodes)

    # y_0 is a concatenation of node vectors of length dim * n_nodes.
    # A node's vector is given by the following construction.
    # y_0[node * dim:(node+1) * dim]
    col = np.zeros(n_sel * dim)
    row = np.kron(nodes * dim, np.ones(dim))
    row += np.kron(np.ones(n_sel), np.arange(dim))

    y_0_2d = np.random.rand(dim)
    y_0_2d /= np.linalg.norm(y_0_2d)
    data = np.kron(np.ones(n_sel), y_0_2d)
    y_0 = sparse.csc_matrix((data, (row, col)), shape=(n_nodes * dim, 1))
    return y_0


def calc_theoretical_results(
    y_0,
    focal_node,
    membership,
    coherence,
    noise,
    com_com_rotation_matrix,
    n_nodes,
    n_communities,
    dim,
):
    """
    Calculate the theoretical results.

    Parameters
    ----------
    y_0: scipy.sparse.csc_matrix
        Initial characteristic vector.
    focal_node: int
        Focal node.
    membership: numpy.ndarray
        Community assignment of each node.
    coherence: float
        Coherence parameter.
    noise: float
        Noise parameter.
    com_com_rotation_matrix: dict
        Rotation matrix for each pair of communities.
    n_nodes: int
        Number of nodes.
    n_communities: int
        Number of communities.
    dim: int
        Dimension of the matrix-weighted SBM.

    Returns
    -------
    y_star: list
        Theoretical results.
    """
    focal_node = 0
    focal_community = membership[focal_node]

    mat = y_0.toarray().reshape(n_nodes, dim)
    y_0_bar = np.sum(
        [
            assign_rotation_matrix(
                src=focal_node,
                trg=k,
                membership=membership,
                com_com_rotation_matrix=com_com_rotation_matrix,
                coherence=coherence,
                noise=noise,
                dim=dim,
            ).dot(mat[k, :])
            for k in range(n_nodes)
        ],
        axis=0,
    )
    y_star = []
    for c in range(n_communities):
        yc = y_0_bar / n_nodes
        yc = com_com_rotation_matrix[(focal_community, c)].dot(yc)
        y_star.append(yc)
    return y_star


def generate_matrix_weighted_ring_of_sbm(
    n_nodes,
    n_communities,
    pin,
    pout,
    coherence,
    noise,
    dim,
):
    """
    Generate a matrix-weighted ring of SBMs.

    Parameters
    ----------
    n_nodes: int
        Number of nodes.
    n_communities: int
        Number of communities.
    pin: float
        Probability of intra-community edges.
    pout: float
        Probability of inter-community edges.
    coherence: float
        Coherence parameter.
    noise: float
        Noise parameter.
    dim: int
        Dimension of the matrix-weighted SBM.

    Returns
    -------
    A_mat: scipy.sparse.csr_matrix
        Matrix-weighted SBM.
    A: scipy.sparse.csr_matrix
        Base network
    membership: numpy.ndarray
        Community assignment of each node.
    com_com_rotation_matrix: dict
        Rotation matrix for each pair of communities.
    """
    assert n_nodes % n_communities == 0, "n_nodes must be divisible by n_communities"

    # Generate a base network using the stochastic block model
    pref_matrix = np.zeros((n_communities, n_communities))
    pref_matrix[np.diag_indices_from(pref_matrix)] = pin
    for i in range(n_communities - 1):
        pref_matrix[i, i + 1] = pout
        pref_matrix[i + 1, i] = pout
    pref_matrix[n_communities - 1, 0] = pout
    pref_matrix[0, n_communities - 1] = pout
    block_sizes = [n_nodes // n_communities] * n_communities

    g = nx.stochastic_block_model(block_sizes, pref_matrix, seed=0, directed=False)
    A = nx.to_scipy_sparse_array(g)

    src, trg, _ = sparse.find(sparse.triu(A, 1))  # only edges
    membership = np.digitize(np.arange(n_nodes), np.cumsum(block_sizes))

    # intially balanced configuration
    # internal edges - identity matrix
    com_com_rotation_matrix = {(l, l): np.eye(dim) for l in range(n_communities)}
    # external edges - first path random
    for l in range(n_communities):
        R = generate_random_rotation_matrix(k=dim)
        com_com_rotation_matrix.update({(l, (l + 1) % n_communities): R})
        com_com_rotation_matrix.update({((l + 1) % n_communities, l): R.T})
    # construct the block weight matrix
    matrix_blocks = [[None for _ in range(n_nodes)] for _ in range(n_nodes)]

    for k, l in zip(src, trg):
        rot = assign_rotation_matrix(
            k, l, membership, com_com_rotation_matrix, coherence, noise, dim
        )
        matrix_blocks[k][l] = rot
        matrix_blocks[l][k] = rot.T

    A_mat = sparse.bmat(matrix_blocks, format="csr")
    return A_mat, A, membership, com_com_rotation_matrix
