import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                str(name): nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            if srctype in feat_dict:
                Wh = self.weight[etype](feat_dict[srctype])
                G.nodes[srctype].data[f'Wh_{etype}'] = Wh
                funcs[etype] = (fn.copy_u(f'Wh_{etype}', 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes if 'h' in G.nodes[ntype].data}


class HeteroRGCN(nn.Module):
    def __init__(self, ntype_dict, etypes, in_size, hidden_size, out_size, n_layers, embedding_size):
        super(HeteroRGCN, self).__init__()
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