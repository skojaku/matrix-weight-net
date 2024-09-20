"""
This script plots the consensus dynamics of the SBM.
"""

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
    input_file = "data/test_fig.npz"
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
t_min = 0
t_train_min = t_min
X_train = np.vstack(
    [node_states[:, t, :] for t in range(t_train_min, node_states.shape[1])]
)

# Use TruncatedSVD as a more robust alternative to PCA
pca = PCA(n_components=2).fit(X_train)
x0 = pca.transform(np.zeros((1, dim)))
xy = pca.transform(X) - np.ones((X.shape[0], 1)) @ x0

node_states_2d = np.array(
    [
        pca.transform(node_states[:, t, :]) - np.ones((n_nodes, 1)) @ x0
        for t in range(node_states.shape[1])
    ]
)
node_states_2d = np.swapaxes(node_states_2d, 0, 1)
y_star_2d = pca.transform(np.array(y_star)) - np.ones((n_communities, 1)) @ x0

plot_data = pd.DataFrame(
    {
        "x": [node_states_2d[i, t, 0] for i in range(n_nodes) for t in range(len(ts))],
        "y": [node_states_2d[i, t, 1] for i in range(n_nodes) for t in range(len(ts))],
        "node_id": np.repeat(range(n_nodes), len(ts)),
        "t": np.tile(ts, n_nodes),
        "membership": np.repeat(membership, len(ts)),
    }
)

plot_data = plot_data.query("t > @t_min")

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(6, 6))

# Sample the same number of nodes for each community
sample_node_size = 10
sampled_nodes = []
for c in range(n_communities):
    sampled_nodes.extend(
        np.random.choice(
            np.where(membership == c)[0], size=sample_node_size, replace=False
        )
    )

#  Add arrows from each node at t to the corresponding node at t+1
for t in range(t_min + 1, len(ts) - 1):
    for i in sampled_nodes:
        dx = node_states_2d[i, t + 1, 0] - node_states_2d[i, t, 0]
        dy = node_states_2d[i, t + 1, 1] - node_states_2d[i, t, 1]
        arrow_length = np.sqrt(dx**2 + dy**2)
        head_length = min(0.02, arrow_length * 0.15)  # Reduced head length
        head_width = head_length * 0.8  # Adjusted head width

        # Calculate the margin to avoid overlap
        margin = 0.005  # Adjust this value to increase/decrease the margin
        start_x = node_states_2d[i, t, 0] + margin * dx / arrow_length
        start_y = node_states_2d[i, t, 1] + margin * dy / arrow_length
        end_x = node_states_2d[i, t + 1, 0] - margin * dx / arrow_length
        end_y = node_states_2d[i, t + 1, 1] - margin * dy / arrow_length

        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(
                arrowstyle="->",
                color="#4d4d4d",
                alpha=0.5,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=10,  # Controls the size of the arrow head
                linestyle="dashed",  # Add dashed line style
            ),
        )

# Create a colorblind-friendly palette with muted and bright tones
# Use a colorblind-friendly palette
palette = sns.color_palette("colorblind")
palette_bright = sns.color_palette("bright")

# Muted colors for nodes
cmap_nodes = [
    sns.desaturate(palette[6], 0.5),  # Blue
    sns.desaturate(palette[1], 0.5),  # Orange
    sns.desaturate(palette[2], 0.5),  # Red
    sns.desaturate(palette[3], 0.5),  # Purple
]

# Bright colors for stars (slightly brighter versions of node colors)
cmap_star = [
    palette_bright[6],  # Brighter blue
    palette_bright[1],  # Brighter orange
    palette_bright[2],  # Brighter red
    palette_bright[3],  # Brighter purple
]

# Ensure we have enough colors for all communities
cmap_nodes = cmap_nodes[:n_communities]
cmap_star = cmap_star[:n_communities]

ax = sns.scatterplot(
    data=plot_data.query("node_id in @sampled_nodes"),
    x="x",
    y="y",
    hue="membership",
    palette=cmap_nodes[:n_communities],
    ax=ax,
    style="membership",
    markers=["o", "s", "D", "X"],
    edgecolor="k",
    s=50,
    zorder=10,
)

for i in range(n_communities):
    # Calculate the direction vector
    direction = y_star_2d[i] / np.linalg.norm(y_star_2d[i])
    # Extend the line to the outside of the bounding box
    ax.plot(
        [0, direction[0] * 10],
        [0, direction[1] * 10],
        color=cmap_star[i],
        linewidth=2,
        zorder=0,
    )
xmin, xmax = plot_data["x"].min(), plot_data["x"].max()
ymin, ymax = plot_data["y"].min(), plot_data["y"].max()
xmin, xmax = xmin - (xmax - xmin) * 0.01, xmax + (xmax - xmin) * 0.01
ymin, ymax = ymin - (ymax - ymin) * 0.01, ymax + (ymax - ymin) * 0.01
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
# ax.set_title("Random Walk")
ax.legend(title="Community", loc="upper right", fontsize=12, frameon=False).remove()
sns.despine()

fig.savefig(output_file, bbox_inches="tight")
