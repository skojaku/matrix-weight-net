# %%
import numpy as np
import pandas as pd
import igraph as ig
from scipy import sparse
import numba

# Parameters
n_nodes = 120
n_communities = 1
pin, pout = 0.3, 0.3
walk_length = 1000
theta = 2 * np.pi / n_communities

assert n_nodes % n_communities == 0, "n_nodes must be divisible by n_communities"

# Generate a base network using the stochastic block model
pref_matrix = np.full((n_communities, n_communities), pout)
pref_matrix[np.diag_indices_from(pref_matrix)] = pin
block_sizes = [n_nodes // n_communities] * n_communities
g = ig.Graph.SBM(
    n_nodes,
    pref_matrix,
    block_sizes=block_sizes,
    directed=False,
)
A = g.get_adjacency_sparse()
membership = np.digitize(np.arange(n_nodes), np.cumsum(block_sizes))

# %% Plot the network
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(A.todense())
plt.show()

# %% Generate random rotation matrix


def generate_rotation_matrix(theta, dim, ax1=None, ax2=None):
    """
    Generate a rotation matrix for a rotation around two axes.
    """
    if ax1 is None and ax2 is None:
        ax12 = np.random.choice(dim, 2, replace=False)
        ax1, ax2 = ax12

    I = np.eye(dim)
    I[ax1, ax1] = np.cos(theta)
    I[ax1, ax2] = -np.sin(theta)
    I[ax2, ax1] = np.sin(theta)
    I[ax2, ax2] = np.cos(theta)
    return I


def random_walk(A, n_steps, initial_node):
    @numba.njit
    def _random_walk(indptr, indices, n_steps, initial_node):
        current_node = initial_node
        traj = np.zeros(n_steps, dtype=np.int32)
        for i in range(n_steps):
            neighbors = indices[indptr[current_node] : indptr[current_node + 1]]
            current_node = np.random.choice(neighbors)
            traj[i] = current_node

        return traj

    return _random_walk(A.indptr, A.indices, n_steps, initial_node)


def assign_rotation_matrix(A, membership, dim, gaussian_noise_std):
    src, trg, _ = sparse.find(sparse.triu(A, 1))
    n_communities = len(set(membership))

    com_rotation_matrix = {}
    for k in range(n_communities):
        for l in range(k, n_communities):
            ax1, ax2 = np.random.choice(dim, 2, replace=False)
            com_rotation_matrix[(k, l)] = {
                "theta": theta,
                "ax1": ax1,
                "ax2": ax2,
                "dim": dim,
            }

        ax1, ax2 = np.random.choice(dim, 2, replace=False)
        com_rotation_matrix[(k, k)] = {
            "theta": 0,
            "ax1": ax1,
            "ax2": ax2,
            "dim": dim,
        }

    R = {}
    for s, t in zip(src, trg):
        k, l = membership[s], membership[t]
        k, l = min(k, l), max(k, l)
        com_rot_params = com_rotation_matrix[(k, l)]
        com_rot_params["theta"] += np.random.randn() * gaussian_noise_std
        R[(s, t)] = generate_rotation_matrix(**com_rot_params)
        R[(t, s)] = R[(s, t)].T
    return R


def run_random_walk(A, dim, RotationMatrix):
    src, trg, _ = sparse.find(sparse.triu(A, 1))

    # Generate the random walk sequence

    node_seq = random_walk(A, walk_length, np.random.randint(n_nodes))
    # Compute the trajectory of the walker in the state space
    x_pos = np.random.randn(dim)
    x_pos = x_pos / np.linalg.norm(x_pos)

    # Convert from the node-level trajectory to the community-level trajectory
    traj = [x_pos.copy()]
    for i in range(walk_length - 1):
        src, trg = node_seq[i], node_seq[i + 1]
        R = RotationMatrix[(src, trg)]
        x_pos = R @ x_pos
        x_pos = x_pos / np.linalg.norm(x_pos)
        traj.append(x_pos.copy())

    traj = np.vstack(traj)
    return traj, node_seq


def calc_autocorrelation(traj, node_seq):
    sim_list = []
    dt_list = []

    for node in set(node_seq):
        ts = np.where(node_seq == node)[0]
        _emb = traj[ts]
        S = _emb @ _emb.T
        s, t = np.triu_indices(S.shape[0], k=1)
        sim = S[s, t]
        dt = ts[t] - ts[s]

        sim_list.append(sim)
        dt_list.append(dt)

    sim_list = np.concatenate(sim_list)
    dt_list = np.concatenate(dt_list)

    return pd.DataFrame({"dt": dt_list, "sim": sim_list}).groupby("dt").mean()


dim_list = [2, 3, 4, 8, 16, 32]
gaussian_noise_std_list = [0, 0.01, 0.05, 0.1]

import itertools
from tqdm import tqdm

results = []


for dim, gaussian_noise_std in tqdm(
    itertools.product(dim_list, gaussian_noise_std_list),
    total=len(dim_list) * len(gaussian_noise_std_list),
):
    RotationMatrix = assign_rotation_matrix(A, membership, dim, gaussian_noise_std)
    traj, node_seq = run_random_walk(A, dim, RotationMatrix)
    df = calc_autocorrelation(traj, node_seq)
    results.append(
        {
            "dim": dim,
            "gaussian_noise_std": gaussian_noise_std,
            "df": df,
            "S": traj @ traj.T,
        }
    )
n_results = len(results)
# %%

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

fig, axes = plt.subplots(figsize=(30, 20), nrows=6, ncols=4)

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
    )
    ax.set_title(
        f"dim: {result['dim']}, gaussian_noise_std: {result['gaussian_noise_std']}"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Time step")
    ax.set_ylabel("Time step")

fig.tight_layout()
# fig.savefig(output_file, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
# %%
dflist = []
for result in results:
    df = result["df"].copy().reset_index()
    df["dim"] = result["dim"]
    df["gaussian_noise_std"] = result["gaussian_noise_std"]
    dflist.append(df)
df = pd.concat(dflist)

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
)
g.set(xscale="log")

sns.despine()

# %%
df

# %%
