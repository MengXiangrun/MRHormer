import torch
import torch_geometric as PyG
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import to_dense_batch
from typing import Optional
import torch
from torch import Tensor


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


class SGFormerAttention(torch.nn.Module):
    def __init__(
            self,
            channels: int,
            heads: int = 1,
            head_channels: int = 64,
            qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert channels % heads == 0
        assert heads == 1, 'The number of heads are fixed as 1.'
        if head_channels is None:
            head_channels = channels // heads

        self.heads = heads
        self.head_channels = head_channels

        inner_channels = head_channels * heads
        self.q = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.k = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.v = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        B, N, *_ = x.shape
        qs, ks, vs = self.q(x), self.k(x), self.v(x)
        # reshape and permute q, k and v to proper shape
        # (b, n, num_heads * head_channels) to (b, n, num_heads, head_channels)
        qs, ks, vs = map(
            lambda t: t.reshape(B, N, self.heads, self.head_channels),
            (qs, ks, vs))

        if mask is not None:
            mask = mask[:, :, None, None]
            vs.masked_fill_(~mask, 0.)
        # replace 0's with epsilon
        epsilon = 1e-6
        qs[qs == 0] = epsilon
        ks[ks == 0] = epsilon
        # normalize input, shape not changed
        qs, ks = map(
            lambda t: t / torch.linalg.norm(t, ord=2, dim=-1, keepdim=True),
            (qs, ks))

        # numerator
        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
        attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)
        attention_num += N * vs

        # denominator
        all_ones = torch.ones([B, N]).to(ks.device)
        ks_sum = torch.einsum("blhm,bl->bhm", ks, all_ones)
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)
        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(attention_normalizer,
                                               len(attention_normalizer.shape))
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer

        return attn_output.mean(dim=2)

    def reset_parameters(self):
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'heads={self.heads}, '
                f'head_channels={self.head_channels})')


class GraphModule(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            num_layers=2,
            dropout=0.5,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
        self.fcs.append(torch.nn.Linear(in_channels, hidden_channels))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        x = self.fcs[0](x)
        x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        last_x = x

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i + 1](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + last_x
        return x


class SGModule(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            num_layers=2,
            num_heads=1,
            dropout=0.5,
    ):
        super().__init__()

        self.attns = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
        self.fcs.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.LayerNorm(hidden_channels))
        for _ in range(num_layers):
            self.attns.append(
                SGFormerAttention(hidden_channels, num_heads, hidden_channels))
            self.bns.append(torch.nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for attn in self.attns:
            attn.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x: Tensor, batch=False):
        # to dense batch expects sorted batch
        batch, indices = batch.sort(stable=True)
        rev_perm = torch.empty_like(indices)
        rev_perm[indices] = torch.arange(len(indices), device=indices.device)
        x = x[indices]
        x, mask = to_dense_batch(x, batch)
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, attn in enumerate(self.attns):
            x = attn(x, mask)
            x = (x + layer_[i]) / 2.
            x = self.bns[i + 1](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        x_mask = x[mask]
        # reverse the sorting
        unsorted_x_mask = x_mask[rev_perm]
        return unsorted_x_mask


class SGFormer(torch.nn.Module):
    def __init__(
            self,
            config,
            trans_num_layers: int = 2,
            trans_num_heads: int = 1,
            trans_dropout: float = 0.5,
            gnn_num_layers: int = 3,
            gnn_dropout: float = 0.5,
            graph_weight: float = 0.5,
            aggregate: str = 'add',
    ):
        super().__init__()
        self.config = config
        self.in_linear = torch.nn.ModuleDict()
        for node_type in config.node_type_list:
            self.in_linear[node_type] = Linear(config.encoder_hidden_dim)

        in_channels = config.encoder_hidden_dim
        hidden_channels = config.encoder_hidden_dim
        out_channels = config.encoder_out_dim

        self.trans_conv = SGModule(
            in_channels,
            hidden_channels,
            trans_num_layers,
            trans_num_heads,
            trans_dropout,
        )
        self.graph_conv = GraphModule(
            in_channels,
            hidden_channels,
            gnn_num_layers,
            gnn_dropout,
        )
        self.graph_weight = graph_weight

        self.aggregate = aggregate

        if aggregate == 'add':
            self.fc = Linear(out_channels)
        elif aggregate == 'cat':
            self.fc = Linear(out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.graph_conv.parameters())
        self.params2.extend(list(self.fc.parameters()))

    def reset_parameters(self) -> None:
        self.trans_conv.reset_parameters()
        self.graph_conv.reset_parameters()

    def forward(self, x_dict, edge_index_dict, batch=False) -> Tensor:
        h_dict = dict()
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.in_linear[node_type](x)

        h, node_range_dict, edge_index = to_homogeneous(node_emb_dict=h_dict, edge_dict=edge_index_dict)

        x1 = self.trans_conv(x, batch)
        x2 = self.graph_conv(x, edge_index)
        if self.aggregate == 'add':
            x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
        else:
            x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)

        z_dict = to_heterogeneous_node_embedding(node_emb=x, node_range_dict=node_range_dict)

        return z_dict
