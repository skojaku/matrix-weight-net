"""
This script plots the random walk dynamics of the SBM.
"""

# %%
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "/home/skojaku/projects/matrix-weight-net/data/random-walk/sbm-n_nodes~120_dim~2_pin~0.3_pout~0.3_noise~0.1_coherence~1_n_communities~3.npz"
    output_file = "test_fig.pdf"

data = np.load(input_file)
node_states = data["node_states"]
y_star = data["y_star"]
membership = data["membership"]
ts = data["ts"]
dim = node_states.shape[2]

n_nodes = len(membership)
n_communities = len(np.unique(membership))

# % Identify the principal components of the node states

X = np.vstack([node_states[:, t, :] for t in range(node_states.shape[1])])

# Calculate cosine similarity between each node state and its expected state
cosine_similarities = np.zeros((n_nodes, len(ts)))

for t in range(len(ts)):
    for i in range(n_nodes):
        x = node_states[i, t, :]
        y = y_star[membership[i]]
        cosine_similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        cosine_similarities[i, t] = cosine_similarity

# Create a DataFrame for plotting
plot_data = pd.DataFrame(
    {
        "node_id": np.repeat(range(n_nodes), len(ts)),
        "time": np.tile(ts, n_nodes),
        "cosine_similarity": cosine_similarities.flatten(),
        "angular_distance": np.arccos(cosine_similarities.flatten()) / np.pi,
        "community": np.repeat(membership, len(ts)),
    }
)


# Set up the plot
sns.set_style("white")
sns.set(font_scale=1.5)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(5, 5))

# Plot individual node trajectories
colormap = [c for i, c in enumerate(sns.color_palette()) if i != 2]
for i in range(n_nodes):
    sns.lineplot(
        data=plot_data[plot_data.node_id == i],
        x="time",
        y="angular_distance",
        color=sns.color_palette()[0],
        # color=colormap[membership[i]],
        alpha=0.1,
        ax=ax,
    )

# Plot average trajectories for each community
sns.lineplot(
    data=plot_data,
    x="time",
    y="angular_distance",
    color=sns.color_palette()[3],
    estimator="mean",
    palette=colormap,
    linewidth=3,
    ax=ax,
)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xscale("log")
ax.set_title("")
ax.legend(title="Community", loc="lower right").remove()
ax.set_xscale("log")
ax.set_yscale("log")
sns.despine()

plt.tight_layout()
plt.savefig(output_file, bbox_inches="tight")
# plt.close()

# %%
