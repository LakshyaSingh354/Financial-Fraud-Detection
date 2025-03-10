import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
import dgl.function as fn


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })
        self.relation_weights = nn.ParameterDict({
            name: nn.Parameter(torch.ones(out_size)) for name in etypes
        })

    def forward(self, G, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            if srctype in feat_dict:
                Wh = self.weight[etype](feat_dict[srctype])
                Wh = Wh * self.relation_weights[etype]  # Modulate with relation-specific weights
                G.nodes[srctype].data[f'Wh_{etype}'] = Wh
                funcs[etype] = (fn.copy_u(f'Wh_{etype}', 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes if 'h' in G.nodes[ntype].data}


def compute_rayleigh_quotient(features, laplacian):
    """
    Compute the Rayleigh Quotient for given features and Laplacian.
    Args:
        features (torch.Tensor): Node feature matrix (num_nodes, feature_dim).
        laplacian (torch.Tensor): Laplacian matrix (num_nodes, num_nodes).
    Returns:
        float: Rayleigh Quotient.
    """
    # Ensure features and Laplacian are properly aligned
    num_nodes, feature_dim = features.shape
    # assert laplacian.shape == (num_nodes, num_nodes), "Laplacian must be square and match feature size."
    
    # Convert to numpy
    x = features.cpu().numpy()
    laplacian = laplacian.cpu().numpy()

    # Compute numerator and denominator
    numerator = np.dot(x.T, np.dot(laplacian, x))  # Shape: (feature_dim, feature_dim)
    denominator = np.dot(x.T, x)  # Shape: (feature_dim, feature_dim)
    
    # Compute trace for multi-dimensional features
    return np.trace(numerator) / np.trace(denominator)

def spectral_subgraph_sampling(G, node_features, k_hop=2):
    """
    Sample subgraphs based on spectral energy using Rayleigh Quotient.
    Args:
        G (networkx.Graph): Input graph.
        node_features (torch.Tensor): Node feature matrix (num_nodes, feature_dim).
        k_hop (int): Number of hops for neighborhood sampling.
    Returns:
        dict: Subgraphs centered around each node with high spectral energy.
    """
    subgraphs = {}

    # Ensure the graph's device is known
    graph_device = G.device

    for ntype in G.ntypes:  # Loop through node types
        nodes = G.nodes(ntype)  # Get all nodes of the current type
        for node in nodes:
            # Specify the node type and ID, ensuring device compatibility
            node_dict = {ntype: torch.tensor([node.item()], dtype=torch.int64, device=graph_device)}
            
            # Extract k-hop subgraph (returns a tuple: subgraph and node mapping)
            subgraph, node_mapping = dgl.khop_in_subgraph(G, k=k_hop, nodes=node_dict)

            # Get node IDs for the subgraph of this type
            subgraph_node_ids = node_mapping[ntype]

            # Specify an edge type to compute adjacency and Laplacian matrices
            # Using the first canonical edge type for simplicity
            canonical_etypes = G.canonical_etypes
            chosen_etype = canonical_etypes[0]  # Replace with logic for selecting relevant edge type
            
            # Get adjacency and Laplacian matrices for the specified edge type
            adj = subgraph.adj(etype=chosen_etype).to_dense()  # Convert sparse to dense

            # Compute degree matrix and Laplacian
            degree = torch.diag(adj.sum(dim=1))  # Diagonal degree matrix
            laplacian = degree - adj
            
            # Ensure sub_features match the subgraph nodes
            sub_features = node_features[subgraph_node_ids]
            
            # Debug prints
            print(f"Adjacency shape: {adj.shape}")
            print(f"Laplacian shape: {laplacian.shape}")
            print(f"Subgraph features shape: {sub_features.shape}")
            
            # Ensure alignment
            assert laplacian.shape[0] == sub_features.shape[0], "Laplacian and feature sizes must match."

            # Compute Rayleigh Quotient
            rq = compute_rayleigh_quotient(torch.tensor(sub_features), laplacian)

            # Save the subgraph and its Rayleigh Quotient
            subgraphs[(ntype, node.item())] = {
                'subgraph': subgraph,
                'rayleigh_quotient': rq
            }

    # Sort subgraphs by Rayleigh Quotient
    sorted_subgraphs = sorted(subgraphs.items(), key=lambda x: x[1]['rayleigh_quotient'], reverse=True)
    return dict(sorted_subgraphs)


class HeteroRGCNUnigad(nn.Module):
    def __init__(self, ntype_dict, etypes, in_size, hidden_size, out_size, n_layers, embedding_size):
        super(HeteroRGCNUnigad, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype: nn.Parameter(torch.Tensor(num_nodes, in_size))
                      for ntype, num_nodes in ntype_dict.items() if ntype != 'target'}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        # create layers
        self.layers = nn.ModuleList()
        self.layers.append(HeteroRGCNLayer(embedding_size, hidden_size, etypes))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(HeteroRGCNLayer(hidden_size, hidden_size, etypes))

        # output layer
        self.layers.append(nn.Linear(hidden_size, out_size))

    def forward(self, g, features):
        # get embeddings for all node types. for user node type, use passed in user features
        h_dict = {ntype: emb for ntype, emb in self.embed.items()}
        # feat_para = torch.tensor(features)
        h_dict['target'] = features

        # pass through all layers
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = layer(g, h_dict)

        # get user logits
        return self.layers[-1](h_dict['target'])