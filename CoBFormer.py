import torch_geometric as PyG
import torch

class Linear(torch.nn.Module):
    def __init__(self, out_dim):
        super(Linear, self).__init__()
        self.out_dim = out_dim
        self.linear = PyG.nn.Linear(in_channels=-1,
                                    out_channels=self.out_dim,
                                    weight_initializer='kaiming_uniform',
                                    bias=True,
                                    bias_initializer=None)
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)


# CoB
class GCN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_linear = torch.nn.ModuleDict()
        for node_type in config.node_type_list:
            self.in_linear[node_type] = Linear(config.encoder_hidden_dim)

        self.local = PyG.nn.GCN(in_channels=config.encoder_in_dim,
                                hidden_channels=config.encoder_hidden_dim,
                                out_channels=config.encoder_out_dim,
                                num_layers=config.num_local_layer)

    def forward(self, x_dict, edge_dict):
        h_dict = dict()
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.in_linear[node_type](x)

        h, node_range_dict, edge = to_homogeneous(node_emb_dict=h_dict, edge_dict=edge_dict)
        h = self.local.forward(x=h, edge_index=edge)

        z_dict = to_heterogeneous_node_embedding(node_emb=h, node_range_dict=node_range_dict)

        return z_dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_dropout(x: torch.Tensor, p: float, training: bool):
    x = x.coalesce()
    return torch.sparse_coo_tensor(x.indices(), F.dropout(x.values(), p=p, training=training),
                                   size=x.size())


class FFN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super(FFN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x = self.dropout(x)
        x = F.relu(self.lin1(x))
        return x
        # return self.lin2(x)


class SparseFFN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super(SparseFFN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = sparse_dropout(x, 0.5, self.training)
        x = F.relu(self.lin1(x))
        return self.lin2(x) + x


class BGA(torch.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 layers: int, n_head: int, use_patch_attn=True, dropout1=0.5, dropout2=0.1, need_attn=False):
        super(BGA, self).__init__()
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout1)
        self.attribute_encoder = FFN(in_channels, hidden_channels)
        self.BGALayers = nn.ModuleList()
        for _ in range(0, layers):
            self.BGALayers.append(
                BGALayer(n_head, hidden_channels, use_patch_attn, dropout=dropout2))
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.attn=[]

    def forward(self, x: torch.Tensor, patch: torch.Tensor, need_attn=False):
        patch_mask = (patch != self.num_nodes - 1).float().unsqueeze(-1)
        attn_mask = torch.matmul(patch_mask, patch_mask.transpose(1, 2)).int()

        x = self.attribute_encoder(x)
        for i in range(0, self.layers):
            x = self.BGALayers[i](x, patch, attn_mask, need_attn)
            if need_attn:
                self.attn.append(self.BGALayers[i].attn)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
class CoBFormer(torch.nn.Module):
    def __init__(self, config,

                 dropout1=0.5,
                 dropout2=0.1,
                 alpha=0.8,
                 tau=0.5,
                 use_patch_attn=True):
        super(CoBFormer, self).__init__()
        self.alpha = alpha
        self.tau = tau
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.activation = activation
        self.dropout = nn.Dropout(dropout1)

        self.gcn = GCN(config)
        self.bga = BGA(num_nodes,
                       in_channels,
                       hidden_channels,
                       out_channels,
                       layers,
                       n_head,
                       use_patch_attn,
                       dropout1,
                       dropout2)
        self.attn = None

    def forward(self, x: torch.Tensor, patch: torch.Tensor, edge_index: torch.Tensor, need_attn=False):
        z1 = self.gcn(x, edge_index)
        z2 = self.bga(x, patch, need_attn)
        if need_attn:
            self.attn = self.beyondformer.attn

        return z1, z2

    def loss(self, pred1, pred2, label, mask):
        l1 = F.cross_entropy(pred1[mask], label[mask])
        l2 = F.cross_entropy(pred2[mask], label[mask])
        pred1 *= self.tau
        pred2 *= self.tau
        l3 = F.cross_entropy(pred1[~mask], F.softmax(pred2, dim=1)[~mask])
        l4 = F.cross_entropy(pred2[~mask], F.softmax(pred1, dim=1)[~mask])
        loss = self.alpha * (l1 + l2) + (1 - self.alpha) * (l3 + l4)
        return loss
