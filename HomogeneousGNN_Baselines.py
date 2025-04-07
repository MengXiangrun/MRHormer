import torch_geometric as PyG
import torch
import copy
import pandas as pd
from torch_scatter import scatter_mean
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import PairTensor  # noqa
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor
from torch_geometric.utils import softmax
import math
from torch.nn import Sequential, Linear
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_geometric.utils.hetero import construct_bipartite_edge_index


def to_homogeneous(node_emb_dict, edge_dict):
    # node number
    num_node_dict = dict()
    for node_type, node_emb in node_emb_dict.items():
        num_node = node_emb.shape[0]
        num_node_dict[node_type] = num_node

    # offset distance
    offset_dict = dict()
    offset = 0
    for node_type, num_node in num_node_dict.items():
        offset_dict[node_type] = offset
        offset += num_node

    # save node type mapping index range
    node_range_dict = dict()
    start_index = 0
    end_index = 0
    for node_type, num_node in num_node_dict.items():
        start_index = end_index
        end_index = end_index + num_node - 1
        node_range_dict[node_type] = (start_index, end_index)
        end_index = end_index + 1

    # stack node emb
    node_emb = torch.cat(list(node_emb_dict.values()), dim=0)

    # add offset distance to original heterogeneous subgraph edge index
    offset_edge_list = list()
    for edge_type, edge in edge_dict.items():
        source_type, target_type = edge_type[0], edge_type[-1]

        source_offset = offset_dict[source_type]
        target_offset = offset_dict[target_type]

        offset_edge = edge.clone()

        offset_edge[0] = offset_edge[0] + source_offset
        offset_edge[1] = offset_edge[1] + target_offset

        offset_edge_list.append(offset_edge)

    offset_edge = torch.cat(offset_edge_list, dim=1)

    return node_emb, node_range_dict, offset_edge


def to_heterogeneous_node_embedding(node_emb, node_range_dict):
    z_dict = dict()
    for node_type, node_range in node_range_dict.items():
        start_index, end_index = node_range
        z = node_emb[start_index:end_index + 1, :]
        z_dict[node_type] = z

    return z_dict


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


class Config():
    def __init__(self,
                 encoder_in_dim=-1, encoder_hidden_dim=128, encoder_out_dim=64,
                 decoder_in_dim=64, decoder_hidden_dim=32, decoder_out_dim=1,
                 num_local_layer=2,
                 num_global_self_layer=2, num_global_cross_layer=2,
                 num_local_head=8, num_global_head=8,
                 local_dropout_probability=0.0,
                 global_dropout_probability=0.0,

                 node_type_list=[], edge_type_list=[],
                 optimizer_learning_rate=0.001, optimizer_weight_decay=0.001):
        self.encoder_in_dim = encoder_in_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_out_dim = encoder_out_dim
        self.decoder_in_dim = decoder_in_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_out_dim = decoder_out_dim
        self.num_local_layer = num_local_layer
        self.num_global_self_layer = num_global_self_layer
        self.num_global_cross_layer = num_global_cross_layer

        self.num_local_head = num_local_head
        self.num_global_head = num_global_head

        self.local_dropout_probability = local_dropout_probability
        self.global_dropout_probability = global_dropout_probability

        self.node_type_list = node_type_list
        self.edge_type_list = edge_type_list
        self.optimizer_learning_rate = optimizer_learning_rate
        self.optimizer_weight_decay = optimizer_weight_decay

        self.num_node = None
        self.is_residual_connection = False
        self.metapath_dict

    def save_to_excel(self, file_path):
        data = {key: [value] for key, value in vars(self).items()}
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)


# Homogeneous GNN Baselines#############################################################################################
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


class GAT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_linear = torch.nn.ModuleDict()
        for node_type in config.node_type_list:
            self.in_linear[node_type] = Linear(config.encoder_hidden_dim)

        self.local = PyG.nn.GAT(in_channels=config.encoder_in_dim,
                                hidden_channels=config.encoder_hidden_dim,
                                out_channels=config.encoder_out_dim,
                                heads=config.num_local_head,
                                num_layers=config.num_local_layer)

    def forward(self, x_dict, edge_dict):
        h_dict = dict()
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.in_linear[node_type](x)

        h, node_range_dict, edge = to_homogeneous(node_emb_dict=h_dict, edge_dict=edge_dict)
        h = self.gcn.forward(x=h, edge_index=edge)

        z_dict = to_heterogeneous_node_embedding(node_emb=h, node_range_dict=node_range_dict)

        return z_dict


class GATv2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_linear = torch.nn.ModuleDict()
        for node_type in config.node_type_list:
            self.in_linear[node_type] = Linear(config.encoder_hidden_dim)

        self.local = PyG.nn.GAT(in_channels=config.encoder_in_dim,
                                hidden_channels=config.encoder_hidden_dim,
                                out_channels=config.encoder_out_dim,
                                heads=config.num_local_head,
                                v2=True,
                                num_layers=config.num_local_layer)

    def forward(self, x_dict, edge_dict):
        h_dict = dict()
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.in_linear[node_type](x)

        h, node_range_dict, edge = to_homogeneous(node_emb_dict=h_dict, edge_dict=edge_dict)
        h = self.gcn.forward(x=h, edge_index=edge)

        z_dict = to_heterogeneous_node_embedding(node_emb=h, node_range_dict=node_range_dict)

        return z_dict


class GraphSAGE(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_linear = torch.nn.ModuleDict()
        for node_type in config.node_type_list:
            self.in_linear[node_type] = Linear(config.encoder_hidden_dim)

        self.local = PyG.nn.GraphSAGE(in_channels=config.encoder_in_dim,
                                      hidden_channels=config.encoder_hidden_dim,
                                      out_channels=config.encoder_out_dim,
                                      num_layers=config.num_local_layer)

    def forward(self, x_dict, edge_dict):
        h_dict = dict()
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.in_linear[node_type](x)

        h, node_range_dict, edge = to_homogeneous(node_emb_dict=h_dict, edge_dict=edge_dict)
        h = self.gcn.forward(x=h, edge_index=edge)

        z_dict = to_heterogeneous_node_embedding(node_emb=h, node_range_dict=node_range_dict)

        return z_dict


class LightGCN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_linear = torch.nn.ModuleDict()
        for node_type in config.node_type_list:
            self.in_linear[node_type] = Linear(config.encoder_hidden_dim)

        self.local = PyG.nn.LightGCN(num_nodes=config.num_node,
                                     embedding_dim=config.encoder_hidden_dim,
                                     out_channels=config.encoder_out_dim,
                                     num_layers=config.num_local_layer)

    def forward(self, x_dict, edge_dict):
        h_dict = dict()
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.in_linear[node_type](x)

        h, node_range_dict, edge = to_homogeneous(node_emb_dict=h_dict, edge_dict=edge_dict)
        h = self.gcn.forward(x=h, edge_index=edge)

        z_dict = to_heterogeneous_node_embedding(node_emb=h, node_range_dict=node_range_dict)

        return z_dict