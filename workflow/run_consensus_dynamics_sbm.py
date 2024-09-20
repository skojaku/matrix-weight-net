"""
This script runs the consensus dynamics of the SBM.
"""

# %%
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import utils
from scipy import sparse
from mpl_toolkits import mplot3d
from scipy.sparse.linalg import expm
import sys

# ---------------------------------
# Generate the matrix-weighted SBM
# ---------------------------------
if "snakemake" in sys.modules:
    params = snakemake.params["params"]
    n_nodes = int(params["n_nodes"])
    n_communities = int(params["n_communities"])
    pin = float(params["pin"])
    pout = float(params["pout"])
    dim = int(params["dim"])
    noise = float(params["noise"])
    coherence = float(params["coherence"])
    output_file = snakemake.output["output_file"]
else:
    n_nodes = 100
    n_communities = 3
    pin, pout = 0.3, 0.3
    dim = 5
    noise = 0.0
    coherence = 1.0
    output_file = "test_fig.pdf"

A_mat, A, membership, com_com_rotation_matrix = utils.generate_matrix_weighted_sbm(
    n_nodes=n_nodes,
    n_communities=n_communities,
    pin=pin,
    pout=pout,
    coherence=coherence,
    noise=noise,
    dim=dim,
)


# %% ---------------------------------
# Initialzie node characteristics
# ---------------------------------
# Sample non-zero characteristic nodes from the first community
non_zero_characteristic_nodes = np.random.choice(n_nodes, n_nodes // n_communities)
y_0 = utils.generate_initial_characteristic_vector(
    nodes=non_zero_characteristic_nodes, n_nodes=n_nodes, dim=dim
)


# ---------------------------------
# Consensus Dynamics
# ---------------------------------
# Theoretical results
y_star = utils.calc_theoretical_results(
    y_0=y_0,
    focal_node=0,
    membership=membership,
    coherence=coherence,
    noise=noise,
    com_com_rotation_matrix=com_com_rotation_matrix,
    n_nodes=n_nodes,
    n_communities=n_communities,
    dim=dim,
)

# Numerical results
degree = np.array(A.sum(axis=1)).reshape(-1)
D_mat = sparse.diags(np.repeat(degree, dim))
L_mat = D_mat - A_mat  # Laplacian matrix

ts = [0.08, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
res = [(expm(-L_mat * t) @ y_0).toarray().reshape(n_nodes, dim) for t in ts]
node_states = np.swapaxes(np.array(res), 0, 1)  # shape = (node x time x dim)

# %%
np.savez(
    output_file, node_states=node_states, y_star=y_star, membership=membership, ts=ts
)
