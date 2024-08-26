# %%
import numpy as np
import pandas as pd
import igraph as ig

# Parameters
n_nodes = 120
n_communities = 3
pin, pout = 0.3, 0.01
dim = 10
walk_length = 1000
coherence = 0.2

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
# %% Plot the network
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(A.todense())
plt.show()

# %% Generate random rotation matrix
import numpy as np


def generate_random_rotation_matrix(sz):
    """Generate a random rotation matrix of size sz x sz using Householder reflections

    Efficient Orthogonal Parametrisation of Recurrent Neural Networks Using Householder Reflections
    Mhammedi et al. arxiv

    """
    weight = np.random.randn((sz * (sz + 1)) // 2)

    def rotate_by_householder(v, weight):
        get_uk = lambda k: weight[
            (k * (k + 1)) // 2 : (k * (k + 1)) // 2 + k + 1
        ].reshape(-1, 1)

        def apply_householder(v, uk):
            k = len(uk)
            v[-k:] = v[-k:] - 2 * (uk.T @ v[-k:]) * uk / np.linalg.norm(uk) ** 2
            return v

        for k in range(1, sz):
            uk = get_uk(k)
            v = apply_householder(v, uk)
        return v

    # Create an identity matrix
    return rotate_by_householder(np.eye(sz), weight)


# Sanity check
R = generate_random_rotation_matrix(dim)
assert np.all(np.isclose(R.T @ R, np.eye(dim))), "R is not orthonormal"

# Generate a set of random rotations for each pair of communities
from scipy import sparse

src, trg, _ = sparse.find(sparse.triu(A, 1))
membership = np.digitize(np.arange(n_nodes), np.cumsum(block_sizes))

com_com_rotation_matrix = {
    (k, l): generate_random_rotation_matrix(dim)
    for k in range(n_communities)
    for l in range(n_communities)
}


def assign_rotation_matrix(src, trg):
    k, l = membership[src], membership[trg]
    if np.random.rand() < coherence:
        if k != l:
            return com_com_rotation_matrix[(k, l)]
        else:
            return np.eye(dim)
    else:
        return generate_random_rotation_matrix(dim)


RotationMatrix = {(k, l): assign_rotation_matrix(k, l) for k, l in zip(src, trg)}

# %% Generate the random walk sequence
import numba


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


node_seq = random_walk(A, walk_length, np.random.randint(n_nodes))
# %% Compute the trajectory of the walker in the state space

# Initial random walk position
x_pos = np.random.randn(dim)
x_pos = x_pos / np.linalg.norm(x_pos)

# Convert from the node-level trajectory to the community-level trajectory
traj = [x_pos.copy()]
for i in range(walk_length - 1):
    src, trg = node_seq[i], node_seq[i + 1]
    if src <= trg:
        R = RotationMatrix[(src, trg)]
    else:
        R = RotationMatrix[(trg, src)]
        R = R.T
    x_pos = R @ x_pos
    x_pos = x_pos / np.linalg.norm(x_pos)
    traj.append(x_pos.copy())

traj = np.vstack(traj)
# %% Similarity matrix between the trajectory
# %%
sns.set_style('white')
sns.set(font_scale=1.2)
sns.set_style('ticks')
cmap = sns.color_palette("colorblind").as_hex()
row_colors = [cmap[membership[i]] for i in node_seq]
col_colors = [cmap[membership[i]] for i in node_seq]

g = sns.clustermap(
    traj @ traj.T,
    cmap="coolwarm",
    center=0,
    figsize=(7, 7),
    row_colors=row_colors,
    col_colors=col_colors,
    row_cluster=False,
    col_cluster=False,
    vmin=-1,
    vmax=1,
    cbar_pos=(0, 0.2, 0.03, 0.4),
)

# Add colorbar label
cbar = g.ax_heatmap.collections[0].colorbar
cbar.set_label("Correlation", rotation=90, labelpad=15)

g.fig.tight_layout()
g.fig.show()
# %%
# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

cmap = sns.color_palette("colorblind")

com_seq = membership[node_seq]
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
xy = PCA(n_components=2).fit_transform(traj)
sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=com_seq, palette=cmap, ax=axes[0])
# Add trajectory lines to PCA plot
for i in range(len(xy) - 1):
    axes[0].plot(
        xy[i : i + 2, 0], xy[i : i + 2, 1], color="gray", alpha=0.5, linewidth=0.5
    )

# Add an arrow to show the direction
arrow_start = xy[-2]
arrow_end = xy[-1]
axes[0].annotate(
    "",
    xy=arrow_end,
    xytext=arrow_start,
    arrowprops=dict(arrowstyle="->", color="red", lw=2),
)


xy = TSNE(n_components=2).fit_transform(traj)

sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=com_seq, palette=cmap, ax=axes[1])
for i in range(len(xy) - 1):
    axes[1].plot(
        xy[i : i + 2, 0], xy[i : i + 2, 1], color="gray", alpha=0.5, linewidth=0.5
    )
plt.tight_layout()

axes[0].set_title("PCA")
axes[1].set_title("t-SNE")
axes[0].axis("off")
axes[1].axis("off")
# %%
