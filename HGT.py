import torch_geometric as PyG
from torch_geometric.typing import PairTensor  # noqa
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


class HGTConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int,
                 node_type_list,
                 edge_type_list,
                 heads: int = 1):
        super().__init__(aggr='add', node_dim=0)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in node_type_list}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_type_list = node_type_list
        self.edge_type_list = edge_type_list
        self.edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(edge_type_list)
        }

        self.dst_node_types = {key[-1] for key in self.edge_type_list}

        self.kqv_lin = HeteroDictLinear(self.in_channels,
                                        self.out_channels * 3)

        self.out_lin = HeteroDictLinear(self.out_channels, self.out_channels,
                                        types=self.node_type_list)

        dim = out_channels // heads
        num_types = heads * len(self.edge_type_list)

        self.k_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)
        self.v_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)

        self.skip = ParameterDict({
            node_type: Parameter(torch.empty(1))
            for node_type in self.node_type_list
        })

        self.p_rel = ParameterDict()
        for edge_type in self.edge_type_list:
            edge_type = '__'.join(edge_type)
            self.p_rel[edge_type] = Parameter(torch.empty(1, heads))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        ones(self.skip)
        ones(self.p_rel)

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        """Concatenates a dictionary of features."""
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset

    def _construct_src_node_feat(
            self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
            edge_index_dict: Dict[EdgeType, Adj]
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        """Constructs the source node representations."""
        cumsum = 0
        num_edge_types = len(self.edge_type_list)
        H, D = self.heads, self.out_channels // self.heads

        # Flatten into a single tensor with shape [num_edge_types * heads, D]:
        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[Tensor] = []
        offset: Dict[EdgeType] = {}
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N

            # construct type_vec for curr edge_type with shape [H, D]
            edge_type_offset = self.edge_types_map[edge_type]
            type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(
                1, N) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])

        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        type_vec = torch.cat(type_list, dim=1).flatten()

        k = self.k_rel(ks, type_vec).view(H, -1, D).transpose(0, 1)
        v = self.v_rel(vs, type_vec).view(H, -1, D).transpose(0, 1)

        return k, v, offset

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[EdgeType, Adj]  # Support both.
    ) -> Dict[NodeType, Optional[Tensor]]:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :class:`torch.Tensor` of
                shape :obj:`[2, num_edges]` or a
                :class:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[torch.Tensor]]` - The output node
            embeddings for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Compute K, Q, V over node types:
        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict)

        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=self.p_rel,
            num_nodes=k.size(0))

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr)

        # Reconstruct output node embeddings dict:
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        # Transform output node embeddings:
        a_dict = self.out_lin({
            k:
                torch.nn.functional.gelu(v) if v is not None else v
            for k, v in out_dict.items()
        })

        # Iterate over node types:
        for node_type, out in out_dict.items():
            out = a_dict[node_type]

            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')


class HGT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.lin_dict1 = torch.nn.ModuleDict()
        for node_type in config.node_type_list:
            self.lin_dict1[node_type] = Linear(config.encoder_hidden_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(config.num_local_layer):
            conv = HGTConv(in_channels=config.encoder_hidden_dim,
                           out_channels=config.encoder_hidden_dim,
                           node_type_list=config.node_type_list,
                           edge_type_list=config.edge_type_list,
                           heads=config.num_local_head)
            self.convs.append(conv)

        self.lin_dict2 = torch.nn.ModuleDict()
        for node_type in config.node_type_list:
            self.lin_dict2[node_type] = Linear(config.encoder_out_dim)

    def forward(self, x_dict, edge_dict):
        x_dict = {
            node_type: self.lin_dict1[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_dict)

        x_dict = {
            node_type: self.lin_dict2[node_type](x)
            for node_type, x in x_dict.items()
        }

        return x_dict
