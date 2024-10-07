import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class DiscourseSheaf:
    def __init__(self, G):
        self.G = G
        self.vertex_spaces = {}
        self.edge_spaces = {}
        self.restriction_maps = {}

    def set_vertex_space(self, v, dim):
        self.vertex_spaces[v] = dim

    def set_edge_space(self, e, dim):
        self.edge_spaces[e] = dim

    def set_restriction_map(self, v, e, matrix):
        self.restriction_maps[(v, e)] = np.array(matrix)

    def coboundary(self):
        n_vertices = self.G.number_of_nodes()
        n_edges = self.G.number_of_edges()
        
        vertex_dims = [self.vertex_spaces[v] for v in self.G.nodes()]
        edge_dims = [self.edge_spaces[e] for e in self.G.edges()]
        
        total_vertex_dim = sum(vertex_dims)
        total_edge_dim = sum(edge_dims)
        
        delta = np.zeros((total_edge_dim, total_vertex_dim))
        
        v_start = 0
        for i, (u, v) in enumerate(self.G.edges()):
            e = (u, v)
            u_dim = self.vertex_spaces[u]
            v_dim = self.vertex_spaces[v]
            e_dim = self.edge_spaces[e]
            
            e_start = sum(edge_dims[:i])
            u_start = sum(vertex_dims[:list(self.G.nodes()).index(u)])
            v_start = sum(vertex_dims[:list(self.G.nodes()).index(v)])
            
            delta[e_start:e_start+e_dim, u_start:u_start+u_dim] = -self.restriction_maps[(u, e)]
            delta[e_start:e_start+e_dim, v_start:v_start+v_dim] = self.restriction_maps[(v, e)]
        
        return delta

    def sheaf_laplacian(self):
        delta = self.coboundary()
        return delta.T @ delta

    def diffuse_opinions(self, initial_opinions, alpha, steps):
        L = self.sheaf_laplacian()
        x = initial_opinions
        opinions = [x]
        for _ in range(steps):
            x = x - alpha * (L @ x)
            opinions.append(x)
        return np.array(opinions)

# Create a simple graph
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2)])

# Create a discourse sheaf
sheaf = DiscourseSheaf(G)

# Set dimensions for vertex and edge spaces
for v in G.nodes():
    sheaf.set_vertex_space(v, 2)  # 2D opinion space for each vertex
for e in G.edges():
    sheaf.set_edge_space(e, 1)  # 1D discourse space for each edge

# Set restriction maps
sheaf.set_restriction_map(0, (0, 1), [[1, 0]])
sheaf.set_restriction_map(1, (0, 1), [[1, 1]])
sheaf.set_restriction_map(1, (1, 2), [[1, -1]])
sheaf.set_restriction_map(2, (1, 2), [[0, 1]])

# Initial opinions
initial_opinions = np.array([1, 0, -1, 1, 0, -1])

# Run opinion diffusion
alpha = 0.1
steps = 100
opinion_evolution = sheaf.diffuse_opinions(initial_opinions, alpha, steps)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(len(initial_opinions)):
    plt.plot(opinion_evolution[:, i], label=f'Opinion {i+1}')
plt.xlabel('Time step')
plt.ylabel('Opinion value')
plt.title('Opinion Dynamics on Discourse Sheaf')
plt.legend()
plt.grid(True)
plt.show()
