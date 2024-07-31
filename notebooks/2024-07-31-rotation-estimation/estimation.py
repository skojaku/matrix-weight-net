# %%
import torch
from random_walks import RandomWalkNodeSampler
import numpy as np
from scipy import sparse
import pandas as pd
from tqdm import tqdm


class MatrixNetwork(torch.nn.Module):
    def __init__(self, n_edges, dim):
        super().__init__()
        self.n_edges = n_edges
        self.dim = dim
        self.edge_matrix = torch.nn.Parameter(
            torch.zeros(self.n_edges, self.dim, self.dim)
        )
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, edge_ids):
        for edge_id in edge_ids:
            x = self.edge_matrix[edge_id] @ x + x  # residual connection
            x = x / x.norm(dim=0, keepdim=True)
        x = self.scale * x
        return x


def train(A, dim, n_epochs=100, lr=1e-3, walk_length=3):

    n_edges = len(A.data)
    n_nodes = A.shape[0]

    sampler = RandomWalkNodeSampler(walk_length=walk_length)
    sampler.fit(A)

    model = MatrixNetwork(n_edges, dim)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    node_vectors = torch.randn(n_nodes, dim)
    node_vectors = node_vectors / node_vectors.norm(dim=1, keepdim=True)

    pbar = tqdm(range(n_epochs))
    loss_list = []
    for epoch in pbar:
        optimizer.zero_grad()

        center_node_list, edge_seq_list = sampler.sampling(size=1, edge_seq=True)
        edge_ids = edge_seq_list[0]
        center_node = center_node_list[0]
        random_edge_ids = np.random.choice(n_edges, size=len(edge_ids), replace=True)
        # center_node = 0  # test
        v = node_vectors[center_node].unsqueeze(1)
        v_pos = model(v, edge_ids)
        v_neg = model(v, random_edge_ids)
        score = v.T @ torch.cat([v_pos, v_neg], dim=1)
        score = score.squeeze()
        loss = criterion(score, torch.tensor([1, 0], dtype=torch.float32))

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        with torch.no_grad():
            pbar.set_description(
                f"Loss: {loss.item()}, score: {(score[1] - score[0]).item()}"
            )

    return model, node_vectors, loss_list


import matplotlib.pyplot as plt
import networkx as nx

G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)
A = sparse.csr_matrix(A)
labels = np.unique([d[1]["club"] for d in G.nodes(data=True)], return_inverse=True)[1]
A.data = np.ones(len(A.data))

src, trg = [], []
for i in range(A.shape[0]):

    trg += [A.indices[A.indptr[i] : A.indptr[i + 1]]]
    src += [i] * len(trg[i])
edgeid2src, edgeid2trg = np.array(src), np.concatenate(trg)

# %% Training
model, node_vecs, loss_list = train(A, dim=5, walk_length=5, n_epochs=3000, lr=1e-3)
W = model.edge_matrix.detach().cpu().numpy()
node_vecs = node_vecs.detach().cpu().numpy()
plt.plot(loss_list)
# %%
edge_weight = []
for i in range(W.shape[0]):
    src = edgeid2src[i]
    trg = edgeid2trg[i]
    v = node_vecs[src].T @ W[i]
    v = v / np.linalg.norm(v)
    a = node_vecs[src].T @ v
    edge_weight.append(a)
edge_weight = np.array(edge_weight)

# %%
import igraph

B = A.copy()
B.data = edge_weight

src, trg, weight = sparse.find(B)

g = igraph.Graph.TupleList([(src[i], trg[i]) for i in range(len(src))], directed=True)
igraph.plot(
    g,
    edge_width=np.exp(4 * edge_weight),
    edge_color=["blue" if a < 0 else "red" for a in edge_weight],
)


# %% Walks
def walk(A, W, walk_length, start_node, v0=None):
    sampler = RandomWalkNodeSampler(walk_length=walk_length)
    sampler.fit(A)
    center_node_list, edge_seq_list = sampler.sampling(
        center_nodes=[start_node], edge_seq=True
    )
    center_node = center_node_list[0]
    edge_ids = edge_seq_list[0]

    xt = []
    if v0 is None:
        x = node_vecs[center_node].reshape((-1, 1))
    else:
        x = v0
    node_ids = edgeid2trg[edge_ids]
    for edge_id in edge_ids:
        x = W[edge_id] @ x + x  # residual connection
        x = x / np.linalg.norm(x)
        xt.append(x)
    xt = np.hstack(xt).T
    return xt


xt = walk(A, W, walk_length=100, start_node=0)

# %% Vis
import umap
from sklearn.decomposition import PCA

xyt = umap.UMAP(
    n_components=2, metric="cosine", min_dist=0.1, n_neighbors=20
).fit_transform(xt)

import seaborn as sns

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(xyt[:, 0], xyt[:, 1], alpha=0.5)
sns.scatterplot(
    x=xyt[:, 0], y=xyt[:, 1], hue=labels[node_ids], ax=ax, zorder=10, edgecolor="k"
)

# %% Distribution viz
n_samples = 1000
x_list = []
u_list = []
start_node = 0
v0 = node_vecs[start_node].reshape((-1, 1))
for i in range(n_samples):
    x = walk(A, W, walk_length=5, start_node=start_node, v0=v0)[-1]
    x_list.append(x.reshape((-1, 1)))
    u_list.append(1)

start_node = 32
# v0 = node_vecs[start_node].reshape((-1, 1))
for i in range(n_samples):
    x = walk(A, W, walk_length=5, start_node=start_node, v0=v0)[-1]
    x_list.append(x.reshape((-1, 1)))
    u_list.append(0)

start_node = 22
# v0 = node_vecs[start_node].reshape((-1, 1))
for i in range(n_samples):
    x = walk(A, W, walk_length=5, start_node=start_node, v0=v0)[-1]
    x_list.append(x.reshape((-1, 1)))
    u_list.append(3)
x_context = np.hstack(x_list).T


pca = PCA(n_components=2)
xy = pca.fit_transform(x_context)

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(7, 5))


sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=u_list, ax=ax, zorder=10, edgecolor="k")
# %%
sns.heatmap(x_context @ x_context.T, cmap="coolwarm")

# %%
