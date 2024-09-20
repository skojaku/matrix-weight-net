"""
This script runs the random walks on the SBM.
"""

# %%
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import utils
from scipy import sparse
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
    n_nodes = 120
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

# %%
# ---------------------------------
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
y_star = utils.calc_theoretical_results_for_random_walk(
    y_0=y_0,
    focal_node=0,
    membership=membership,
    coherence=coherence,
    noise=noise,
    com_com_rotation_matrix=com_com_rotation_matrix,
    n_nodes=n_nodes,
    n_communities=n_communities,
    dim=dim,
    n_edges=A.sum() / 2,
)

# Numerical results
degree = np.array(A.sum(axis=1)).reshape(-1)
D_mat = sparse.diags(np.repeat(degree, dim))
Dinv_mat = sparse.diags(1 / np.repeat(degree, dim))
P_mat = Dinv_mat @ A_mat


# %% Compute the node states

ts = np.arange(50)

traj = [y_0.copy().toarray().reshape(n_nodes, dim)]
y_t = y_0
for t in ts:
    y_t = y_t.reshape((1, -1)) @ P_mat
    traj.append(y_t.toarray().reshape(n_nodes, dim))

node_states = np.swapaxes(np.array(traj), 0, 1)  # shape = (node x time x dim)

# %%
np.savez(
    output_file, node_states=node_states, y_star=y_star, membership=membership, ts=ts
)
