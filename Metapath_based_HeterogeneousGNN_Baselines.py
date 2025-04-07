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


# SeHGNN[AAAI 2023]
# Simple and Efficient Heterogeneous Graph Neural Network
# http://arxiv.org/abs/2207.02547
# from openhgnn.models import SeHGNN
class SeHGNN_Transformer(nn.Module):
    def __init__(self, n_channels, att_drop=0., act='none', num_heads=1):
        super(SeHGNN_Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels // 4)
        self.key = nn.Linear(self.n_channels, self.n_channels // 4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

        self.reset_parameters()

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.query.weight, gain=gain)
        xavier_uniform_(self.key.weight, gain=gain)
        xavier_uniform_(self.value.weight, gain=gain)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, x, mask=None):
        # batchsize(num_node), num_metapaths, channels
        B, M, C = x.size()
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))

        f = self.query(x).view(B, M, H, -1).permute(0, 2, 1, 3)  # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0, 2, 3, 1)  # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0, 2, 1, 3)  # [B, H, M, -1]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1)  # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h)  # [B, H, M, -1]
        return o.permute(0, 2, 1, 3).reshape((B, M, C)) + x


class SeHGNN_Conv1d1x1(nn.Module):
    def __init__(self, cin, cout, groups, bias=True, cformat='channel-first'):
        super(SeHGNN_Conv1d1x1, self).__init__()
        self.cin = cin
        self.cout = cout
        self.groups = groups
        self.cformat = cformat
        if not bias:
            self.bias = None
        if self.groups == 1:  # different keypoints share same kernel
            self.W = nn.Parameter(torch.randn(self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(1, self.cout))
        else:
            self.W = nn.Parameter(torch.randn(self.groups, self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.groups, self.cout))

        self.reset_parameters()

    def reset_parameters(self):
        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.groups == 1:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,mn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,mn->bnc', x, self.W) + self.bias.T
            else:
                assert False
        else:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,cmn->bnc', x, self.W) + self.bias.T
            else:
                assert False


class SeHGNN(nn.Module):
    def __init__(self, config):
        super(SeHGNN, self).__init__()
        self.in_dim = config.encoder_in_dim
        self.hidden_dim = config.encoder_hidden_dim
        self.out_dim = config.encoder_out_dim

        self.dropout = config.local_dropout_probability
        self.in_dropout = config.local_dropout_probability

        self.node_type_list = config.node_type_list
        self.edge_type_list = config.edge_type_list
        self.metapath_dict = config.metapath_dict

        self.num_metapath_dict = {}
        for target_node_type, metapath_list in self.metapath_dict.items():
            self.num_metapath_dict[target_node_type] = len(metapath_list)

        self.feat_project_layers = torch.nn.ModuleDict()
        self.semantic_aggr_layers = torch.nn.ModuleDict()
        self.concat_project_layer = torch.nn.ModuleDict()
        for target_node_type, num_metapath in self.num_metapath_dict.items():
            mlp = nn.Sequential(
                SeHGNN_Conv1d1x1(self.in_dim[target_node_type],
                                 self.hidden_dim,
                                 num_metapath,
                                 bias=True,
                                 cformat='channel-first'),
                nn.LayerNorm(normalized_shape=[num_metapath, self.hidden_dim]),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                SeHGNN_Conv1d1x1(self.hidden_dim, self.hidden_dim, num_metapath, bias=True, cformat='channel-first'),
                nn.LayerNorm([num_metapath, self.hidden_dim]),
                nn.PReLU(),
                nn.Dropout(self.dropout))
            semantic = SeHGNN_Transformer(self.hidden_dim)
            linear = Linear(self.out_dim)
            self.feat_project_layers[target_node_type] = mlp
            self.semantic_aggr_layers[target_node_type] = semantic
            self.concat_project_layer[target_node_type] = linear

        self.PReLU = nn.PReLU()
        self.dropout = nn.Dropout(self.dropout)
        self.in_dropout = nn.Dropout(self.in_dropout)

    def forward(self, node_emb_dict, edge_dict, metapath_dict):
        metapath_node_emb_dict = self.preprocess(node_emb_dict, edge_dict, metapath_dict)

        for node_type, node_emb in metapath_node_emb_dict.items():
            num_metapath, num_node, node_dim = node_emb.reshape
            node_emb = node_emb.transpose(0, 1)
            num_node, num_metapath, node_dim = node_emb.reshape

            node_emb = self.in_dropout(node_emb)
            node_emb = self.feat_project_layers[node_type](node_emb)
            node_emb = self.semantic_aggr_layers[node_type](node_emb)

            node_emb = node_emb.contiguous().reshape(num_metapath * num_node, node_dim)
            node_emb = self.concat_project_layer[node_type](node_emb)

        return node_emb

    def preprocess(self, node_emb_dict, edge_dict, metapath_dict):
        metapath_edge_dict = create_metapath_edge_index(node_emb_dict=node_emb_dict,
                                                        edge_dict=edge_dict,
                                                        metapath_dict=metapath_dict)
        metapath_node_emb_dict = dict()
        for edge_type, edge in metapath_edge_dict:
            source_type, metapath_type, target_type = edge_type

            source_index = edge[0]
            target_index = edge[1]

            source_emb = node_emb_dict[source_type]
            target_emb = node_emb_dict[target_type]

            target_emb = scatter_mean(src=source_emb, index=target_index, dim=0, dim_size=target_emb.shape[0])

            metapath_node_emb_dict.setdefault(target_type, [])
            metapath_node_emb_dict[target_type].append(target_emb)

        # (num_metapath, num_node, node_emb_dim)
        for target_node_type, node_emb_list in metapath_node_emb_dict.items():
            metapath_node_emb_dict[target_node_type] = torch.stack(node_emb_list, dim=0)

        return metapath_node_emb_dict


