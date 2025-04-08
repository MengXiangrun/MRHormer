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

def create_metapath_edge_index(node_emb_dict, edge_dict, metapath_dict):
    metapath_edge_dict = {}

    for node_type, metapath_list in metapath_dict.items():
        metapath_index = 0
        for metapath in metapath_list:
            edge_type = metapath[0]
            source_type = edge_type[0]
            target_type = edge_type[-1]
            num_source_node = node_emb_dict[source_type].shape[0]
            num_target_node = node_emb_dict[target_type].shape[0]
            sparse_size = (num_source_node, num_target_node)
            adj1 = PyG.typing.SparseTensor.from_edge_index(edge_index=edge_dict[edge_type],
                                                           sparse_sizes=sparse_size)
            for edge_type in metapath[1:]:
                source_type = edge_type[0]
                target_type = edge_type[-1]
                num_source_node = node_emb_dict[source_type].shape[0]
                num_target_node = node_emb_dict[target_type].shape[0]
                sparse_size = (num_source_node, num_target_node)
                adj2 = PyG.typing.SparseTensor.from_edge_index(edge_index=edge_dict[edge_type],
                                                               sparse_sizes=sparse_size)

            adj1 = adj1 @ adj2
            source_index, target_index, edge_weight = adj1.coo()
            metapath_type = (metapath[0][0], f'{node_type}_metapath_{metapath_index}', metapath[-1][-1])
            metapath_edge_dict[metapath_type] = torch.vstack([source_index, target_index])
            metapath_index += 1

    return metapath_edge_dict

# Metapath-based Heterogeneous GNN Baselines############################################################################

# HAN[WWW 2019]
# Heterogeneous Graph Attention Network
class HANConv(MessagePassing):
    def __init__(self,
                 in_channels: Union[int, Dict[str, int]],
                 out_channels: int,
                 edge_type_list,
                 node_type_list,
                 heads: int = 1,
                 negative_slope=0.2,
                 dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in node_type_list}
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.k_lin = nn.Linear(out_channels, out_channels)
        self.q = nn.Parameter(torch.empty(1, out_channels))
        self.proj = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.proj[node_type] = Linear(in_channels, out_channels)
        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in edge_type_list:
            edge_type = '__'.join(edge_type)
            self.lin_src[edge_type] = nn.Parameter(torch.empty(1, heads, dim))
            self.lin_dst[edge_type] = nn.Parameter(torch.empty(1, heads, dim))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.proj)
        glorot(self.lin_src)
        glorot(self.lin_dst)
        self.k_lin.reset_parameters()
        glorot(self.q)

    def forward(self,
                x_dict: Dict[NodeType, Tensor],
                edge_index_dict: Dict[EdgeType, Adj],
                return_semantic_attention_weights: bool = False,
                ) -> Union[Dict[NodeType, OptTensor], Tuple[Dict[NodeType, OptTensor], Dict[NodeType, OptTensor]]]:
        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict = {}, {}

        # Iterate over node types:
        for node_type, x in x_dict.items():
            x_node_dict[node_type] = self.proj[node_type](x).view(-1, H, D)
            out_dict[node_type] = []

        # Iterate over edge types:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            lin_src = self.lin_src[edge_type]
            lin_dst = self.lin_dst[edge_type]
            x_src = x_node_dict[src_type]
            x_dst = x_node_dict[dst_type]
            alpha_src = (x_src * lin_src).sum(dim=-1)
            alpha_dst = (x_dst * lin_dst).sum(dim=-1)
            out = self.propagate(edge_index, x=(x_src, x_dst),
                                 alpha=(alpha_src, alpha_dst))
            out = F.relu(out)
            out_dict[dst_type].append(out)

        # iterate over node types:
        semantic_attn_dict = {}
        for node_type, outs in out_dict.items():
            out, attn = self.semantic(outs, self.q, self.k_lin)
            out_dict[node_type] = out
            semantic_attn_dict[node_type] = attn

        if return_semantic_attention_weights:
            return out_dict, semantic_attn_dict

        return out_dict

    def message(self, x_j: Tensor, alpha_i: Tensor, alpha_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:

        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def semantic(self, xs: List[Tensor], q: nn.Parameter, k_lin: nn.Module) -> Tuple[OptTensor, OptTensor]:
        if len(xs) == 0:
            return None, None
        else:
            num_edge_types = len(xs)
            out = torch.stack(xs)
            if out.numel() == 0:
                return out.view(0, out.size(-1)), None
            attn_score = (q * torch.tanh(k_lin(out)).mean(1)).sum(-1)
            attn = F.softmax(attn_score, dim=0)
            out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
            return out, attn


class HAN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hgnn = HANConv(in_channels=config.encoder_in_dim,
                            out_channels=config.encoder_out_dim,
                            heads=config.num_local_head,
                            edge_type_list=config.edge_type_list,
                            node_type_list=config.node_type_list)

    def forward(self, x_dict, edge_dict, metapath_dict):
        metapath_edge_dict = create_metapath_edge_index(node_emb_dict=x_dict,
                                                        edge_dict=edge_dict,
                                                        metapath_dict=metapath_dict)
        h_dict = {}
        for edge_type in metapath_edge_dict.keys():
            src_type, dst_type = edge_type[0], edge_type[-1]
            h_dict[src_type] = x_dict[src_type]
            h_dict[dst_type] = x_dict[dst_type]

        h_dict = self.hgnn.forward(x_dict=h_dict, edge_index_dict=metapath_edge_dict)

        return h_dict