# %%
import numpy as np
import pandas as pd
import igraph as ig
from scipy import sparse
import numba

# Parameters
n_nodes = 120  # Number of nodes in the network
n_communities = 1  # Number of communities in the network
pin, pout = 0.3, 0.3  # Intra-community and inter-community connection probabilities
walk_length = 1000  # Length of the random walk
theta = 2 * np.pi / n_communities  # Angle for rotation matrix

# Ensure the number of nodes is divisible by the number of communities
assert n_nodes % n_communities == 0, "n_nodes must be divisible by n_communities"

# Generate a base network using the stochastic block model
pref_matrix = np.full((n_communities, n_communities), pout)  # Preference matrix for SBM
pref_matrix[np.diag_indices_from(pref_matrix)] = (
    pin  # Set intra-community probabilities
)
block_sizes = [n_nodes // n_communities] * n_communities  # Sizes of each community
g = ig.Graph.SBM(
    n_nodes,
    pref_matrix,
    block_sizes=block_sizes,
    directed=False,
)  # Generate the graph
A = g.get_adjacency_sparse()  # Get the adjacency matrix in sparse format
membership = np.digitize(
    np.arange(n_nodes), np.cumsum(block_sizes)
)  # Assign nodes to communities

# %% Plot the network
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the adjacency matrix as a heatmap
sns.heatmap(A.todense())
plt.show()

# %% Generate random rotation matrix


def generate_rotation_matrix(theta, dim, ax1=None, ax2=None):
    """
    Generate a rotation matrix for a rotation around two axes.
    """
    if ax1 is None and ax2 is None:
        ax12 = np.random.choice(dim, 2, replace=False)  # Randomly choose two axes
        ax1, ax2 = ax12

    I = np.eye(dim)  # Identity matrix
    I[ax1, ax1] = np.cos(theta)  # Set rotation values
    I[ax1, ax2] = -np.sin(theta)
    I[ax2, ax1] = np.sin(theta)
    I[ax2, ax2] = np.cos(theta)
    return I


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


def assign_rotation_matrix(A, membership, dim, gaussian_noise_std):
    src, trg, _ = sparse.find(
        sparse.triu(A, 1)
    )  # Find edges in the upper triangle of the adjacency matrix
    n_communities = len(set(membership))  # Number of communities

    com_rotation_matrix = {}
    for k in range(n_communities):
        for l in range(k, n_communities):
            ax1, ax2 = np.random.choice(
                dim, 2, replace=False
            )  # Randomly choose two axes
            com_rotation_matrix[(k, l)] = {
                "theta": theta,
                "ax1": ax1,
                "ax2": ax2,
                "dim": dim,
            }

        ax1, ax2 = np.random.choice(
            dim, 2, replace=False
        )  # Randomly choose two axes for intra-community
        com_rotation_matrix[(k, k)] = {
            "theta": 0,
            "ax1": ax1,
            "ax2": ax2,
            "dim": dim,
        }

    R = {}
    for s, t in zip(src, trg):
        k, l = membership[s], membership[t]  # Get community memberships
        k, l = min(k, l), max(k, l)  # Ensure k <= l
        com_rot_params = com_rotation_matrix[(k, l)]  # Get rotation parameters
        com_rot_params["theta"] += (
            np.random.randn() * gaussian_noise_std
        )  # Add noise to the angle
        R[(s, t)] = generate_rotation_matrix(
            **com_rot_params
        )  # Generate rotation matrix
        R[(t, s)] = R[(s, t)].T  # Ensure symmetry
    return R


def run_random_walk(A, dim, RotationMatrix):
    src, trg, _ = sparse.find(
        sparse.triu(A, 1)
    )  # Find edges in the upper triangle of the adjacency matrix

    # Generate the random walk sequence
    node_seq = random_walk(
        A, walk_length, np.random.randint(n_nodes)
    )  # Perform random walk
    # Compute the trajectory of the walker in the state space
    x_pos = np.random.randn(dim)  # Initial position in state space
    x_pos = x_pos / np.linalg.norm(x_pos)  # Normalize the position

    # Convert from the node-level trajectory to the community-level trajectory
    traj = [x_pos.copy()]  # Initialize trajectory list
    for i in range(walk_length - 1):
        src, trg = node_seq[i], node_seq[i + 1]  # Get source and target nodes
        R = RotationMatrix[(src, trg)]  # Get rotation matrix
        x_pos = R @ x_pos  # Apply rotation
        x_pos = x_pos / np.linalg.norm(x_pos)  # Normalize the position
        traj.append(x_pos.copy())  # Append to trajectory

    traj = np.vstack(traj)  # Convert list to array
    return traj, node_seq


def calc_autocorrelation(traj, node_seq):
    sim_list = []
    dt_list = []

    for node in set(node_seq):
        ts = np.where(node_seq == node)[0]  # Get time steps for the node
        _emb = traj[ts]  # Get embeddings for the node
        S = _emb @ _emb.T  # Compute similarity matrix
        s, t = np.triu_indices(S.shape[0], k=1)  # Get upper triangle indices
        sim = S[s, t]  # Get similarities
        dt = ts[t] - ts[s]  # Get time differences

        sim_list.append(sim)  # Append similarities
        dt_list.append(dt)  # Append time differences

    sim_list = np.concatenate(sim_list)  # Concatenate similarities
    dt_list = np.concatenate(dt_list)  # Concatenate time differences

    return (
        pd.DataFrame({"dt": dt_list, "sim": sim_list}).groupby("dt").mean()
    )  # Return mean similarities


dim_list = [2, 3, 4, 8, 16, 32]  # List of dimensions to test
gaussian_noise_std_list = [0, 0.01, 0.05, 0.1]  # List of noise levels to test

import itertools
from tqdm import tqdm

results = []

# Iterate over all combinations of dimensions and noise levels
for dim, gaussian_noise_std in tqdm(
    itertools.product(dim_list, gaussian_noise_std_list),
    total=len(dim_list) * len(gaussian_noise_std_list),
):
    RotationMatrix = assign_rotation_matrix(
        A, membership, dim, gaussian_noise_std
    )  # Assign rotation matrices
    traj, node_seq = run_random_walk(A, dim, RotationMatrix)  # Run random walk
    df = calc_autocorrelation(traj, node_seq)  # Calculate autocorrelation
    results.append(
        {
            "dim": dim,
            "gaussian_noise_std": gaussian_noise_std,
            "df": df,
            "S": traj @ traj.T,
        }
    )  # Store results
n_results = len(results)  # Number of results
# %%

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

fig, axes = plt.subplots(figsize=(30, 20), nrows=6, ncols=4)  # Create subplots

cbar_ax = fig.add_axes(
    [1.05, 0.15, 0.02, 0.7]
)  # Adjusted position for the common colorbar
for ax, result in zip(axes.ravel(), results):
    sns.heatmap(
        result["S"],
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_ax=cbar_ax if ax == axes[0, 0] else None,
        cbar=ax == axes[0, 0],
    )  # Plot heatmap
    ax.set_title(
        f"dim: {result['dim']}, gaussian_noise_std: {result['gaussian_noise_std']}"
    )  # Set title
    ax.set_xticks([])  # Remove x-ticks
    ax.set_yticks([])  # Remove y-ticks
    ax.set_xlabel("Time step")  # Set x-label
    ax.set_ylabel("Time step")  # Set y-label

fig.tight_layout()  # Adjust layout
# fig.savefig(output_file, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
# %%
dflist = []
for result in results:
    df = result["df"].copy().reset_index()  # Copy and reset index
    df["dim"] = result["dim"]  # Add dimension column
    df["gaussian_noise_std"] = result["gaussian_noise_std"]  # Add noise level column
    dflist.append(df)  # Append to list
df = pd.concat(dflist)  # Concatenate all dataframes

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

g = (
    sns.FacetGrid(
        df,
        col="dim",
        hue="gaussian_noise_std",
        palette="cividis",
        height=5,
        aspect=1.0,
        col_wrap=2,
        sharex=False,
        sharey=False,
    )
    .map(sns.lineplot, "dt", "sim")
    .add_legend()
)  # Create FacetGrid and plot lineplot
g.set(xscale="log")  # Set x-axis to log scale

sns.despine()  # Remove top and right spines
