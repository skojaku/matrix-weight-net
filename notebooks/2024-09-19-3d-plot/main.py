# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numba
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import utils
from scipy import sparse
from mpl_toolkits import mplot3d
from scipy.sparse.linalg import expm
import sys


def plot_sphere_trajectory_and_points(Xlist, Y=None, color_palette=plt.cm.viridis):
    # Create a figure and 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the trajectory with arrows
    for X in Xlist:
        for i in range(len(X) - 1):
            ax.plot(
                X[i : i + 2, 0],
                X[i : i + 2, 1],
                X[i : i + 2, 2],
                color=color_palette(i / len(X)),
                linewidth=1,
            )
            ax.quiver(
                X[i, 0], X[i, 1], X[i, 2],
                X[i + 1, 0] - X[i, 0], X[i + 1, 1] - X[i, 1], X[i + 1, 2] - X[i, 2],
                color=color_palette(i / len(X)),
                arrow_length_ratio=0.08
            )

    # Plot additional points if Y is provided
    if Y is not None:
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c="g", s=50)

    ax.legend().remove()

    # Create wireframe sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the wireframe sphere
    ax.plot_wireframe(x, y, z, color="b", alpha=0.1)

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory and Points on a Unit Sphere")

    # Set aspect ratio to be equal
    ax.set_box_aspect((1, 1, 1))

    # Remove grid
    ax.grid(False)

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()


def random_walk(A, n_steps, initial_node):
    @numba.njit
    def _random_walk(indptr, indices, n_steps, initial_node):
        current_node = initial_node  # Start from the initial node
        traj = np.zeros(n_steps, dtype=np.int32)  # Initialize trajectory array
        for i in range(n_steps):
            neighbors = indices[
                indptr[current_node] : indptr[current_node + 1]
            ]  # Get neighbors
            current_node = np.random.choice(neighbors)  # Choose next node randomly
            traj[i] = current_node  # Record the current node

        return traj

    return _random_walk(A.indptr, A.indices, n_steps, initial_node)


# ---------------------------------
# Random Walk
# ---------------------------------


def random_walk_traj(A, A_mat, walk_length, x0):
    src, trg, _ = sparse.find(
        sparse.triu(A, 1)
    )  # Find edges in the upper triangle of the adjacency matrix

    # Generate the random walk sequence
    node_seq = random_walk(
        A, walk_length, np.random.randint(n_nodes)
    )  # Perform random walk
    # Compute the trajectory of the walker in the state space
    x_pos = x0.copy()  # Initial position in state space
    # Convert from the node-level trajectory to the community-level trajectory
    traj = [x_pos.copy()]  # Initialize trajectory list
    for i in range(walk_length - 1):
        src, trg = node_seq[i], node_seq[i + 1]  # Get source and target nodes
        R = A_mat[
            dim * src : dim * (src + 1), dim * trg : dim * (trg + 1)
        ]  # Get rotation matrix
        x_pos = R @ x_pos  # Apply rotation
        x_pos = x_pos / np.linalg.norm(x_pos)  # Normalize the position
        traj.append(x_pos.copy())  # Append to trajectory

    traj = np.vstack(traj)  # Convert list to array
    return traj


# ---------------------------------
# Generate the matrix-weighted SBM
# ---------------------------------
if "snakemake" in sys.modules:
    n_nodes = int(snakemake.params["n_nodes"])
    n_communities = int(snakemake.params["n_communities"])
    pin = float(snakemake.params["pin"])
    pout = float(snakemake.params["pout"])
    dim = int(snakemake.params["dim"])
    noise = float(snakemake.params["noise"])
    coherence = float(snakemake.params["coherence"])
    output_file = snakemake.output["output_file"]
else:
    n_nodes = 90
    n_communities = 3
    pin, pout = 0.3, 0.1
    dim = 3
    noise = 0.1
    coherence = 0.8
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

#
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

import seaborn as sns

walk_length = 30
x0 = np.random.randn(dim)
x0 = x0 / np.linalg.norm(x0)
plot_sphere_trajectory_and_points(
    [random_walk_traj(A, A_mat, walk_length, x0)],
    Y=None,
    color_palette=sns.color_palette("plasma", as_cmap=True),
)

# %%
