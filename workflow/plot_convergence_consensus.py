"""
"""

# %%
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import sys
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "/home/skojaku/projects/matrix-weight-net/data/consensus-dynamics/sbm-n_nodes~120_dim~3_pin~0.3_pout~0.3_noise~0.1_coherence~1_n_communities~3.npz"
    output_file = "test_fig.pdf"

data = np.load(input_file)
node_states = data["node_states"]
y_star = data["y_star"]
membership = data["membership"]
ts = data["ts"]

n_nodes = len(membership)
n_communities = len(np.unique(membership))

# % Identify the principal components of the node states
X = np.vstack([node_states[:, t, :] for t in range(node_states.shape[1])])

# Compute distances between each node state and y_star
distances = np.zeros((n_nodes, len(ts)))
assignments = np.zeros((n_nodes, len(ts)), dtype=int)

for t in range(len(ts)):
    for i in range(n_nodes):
        # Calculate distances to all y_star points
        dists = np.linalg.norm(node_states[i, t] - y_star, axis=1)
        # Find the index of the nearest y_star
        # nearest_y_star = np.argmin(dists)
        nearest_y_star = membership[i]
        # Store the assignment and distance
        assignments[i, t] = nearest_y_star
        distances[i, t] = dists[nearest_y_star]

# Create a DataFrame for easier plotting
plot_data = pd.DataFrame(
    {
        "node_id": np.repeat(range(n_nodes), len(ts)),
        "time": np.tile(ts, n_nodes),
        "distance": distances.flatten(),
        "assigned_community": assignments.flatten(),
        "true_community": np.repeat(membership, len(ts)),
    }
)
sns.set_style("white")
sns.set(font_scale=1.5)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(5, 5))

for i in range(n_nodes):
    sns.lineplot(
        data=plot_data.query("node_id == @i"),
        x="time",
        y="distance",
        # marker="o",
        color=sns.color_palette()[0],
        ax=ax,
        alpha=0.1,
    )
sns.lineplot(
    data=plot_data.groupby(["time"]).mean().reset_index(),
    x="time",
    y="distance",
    # style="assigned_community"
    color=sns.color_palette()[3],
    palette=[c for i, c in enumerate(sns.color_palette("bright")) if i != 2],
    # markers={0: "o", 1: "s", 2: "D", 3: "X"},
    linewidth=3,
    markeredgecolor="k",
    # markersize=4,
    ax=ax,
    alpha=1.0,
)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xscale("log")
ax.set_yscale("log")

ax.legend(title="Node ID", loc="upper right", fontsize=12, frameon=False).remove()
sns.despine()

fig.savefig(output_file, bbox_inches="tight")
