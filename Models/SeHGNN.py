import torch_geometric as PyG
from torch_scatter import scatter_mean
from torch_geometric.typing import PairTensor  # noqa
import torch.nn as nn
import torch.nn.functional as F
import math
import torch


class Linear(torch.nn.Module):
    def __init__(self, out_dim, bias=True):
        super().__init__()
        self.out_dim = out_dim
        self.linear = PyG.nn.Linear(in_channels=-1,
                                    out_channels=self.out_dim,
                                    weight_initializer='kaiming_uniform',
                                    bias=bias,
                                    bias_initializer='zeros')
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)


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


class SeHGNN_Transformer(nn.Module):
    def __init__(self, n_channels, att_drop=0., act='none', num_heads=1):
        super(SeHGNN_Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = Linear(self.n_channels // 4)
        self.key = Linear(self.n_channels // 4)
        self.value = Linear(self.n_channels)

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

    def forward(self, node_emb_dict, edge_dict):
        metapath_node_emb_dict = self.preprocess(node_emb_dict, edge_dict, self.metapath_dict)

        z_dict = {}
        for node_type, node_emb in metapath_node_emb_dict.items():
            num_metapath, num_node, node_dim = node_emb.shape
            node_emb = node_emb.transpose(0, 1)
            num_node, num_metapath, node_dim = node_emb.shape

            node_emb = self.in_dropout(node_emb)
            node_emb = self.feat_project_layers[node_type](node_emb)
            node_emb = self.semantic_aggr_layers[node_type](node_emb)

            node_emb = node_emb.contiguous().reshape(num_metapath * num_node, -1)
            node_emb = self.concat_project_layer[node_type](node_emb)

            z_dict[node_type] = node_emb

        return z_dict

    def preprocess(self, node_emb_dict, edge_dict, metapath_dict):
        metapath_edge_dict = create_metapath_edge_index(node_emb_dict=node_emb_dict,
                                                        edge_dict=edge_dict,
                                                        metapath_dict=metapath_dict)
        metapath_node_emb_dict = dict()
        for edge_type, edge in metapath_edge_dict.items():
            source_type, metapath_type, target_type = edge_type

            source_index = edge[0]
            target_index = edge[1]

            source_emb = node_emb_dict[source_type]
            target_emb = node_emb_dict[target_type]

            source_emb = source_emb[source_index]

            target_emb = scatter_mean(src=source_emb, index=target_index, dim=0, dim_size=target_emb.shape[0])

            metapath_node_emb_dict.setdefault(target_type, [])
            metapath_node_emb_dict[target_type].append(target_emb)

        # (num_metapath, num_node, node_emb_dim)
        for target_node_type, node_emb_list in metapath_node_emb_dict.items():
            metapath_node_emb_dict[target_node_type] = torch.stack(node_emb_list, dim=0)

        return metapath_node_emb_dict


#
# from Config import Config
#
# config = Config()
# # 节点嵌入字典
# node_emb_dict = {
#     'a': torch.ones((10, 2)),  # 10个类型为'a'的节点，每个节点有2维嵌入
#     'b': 2 * torch.ones((5, 2)),  # 5个类型为'b'的节点，每个节点有2维嵌入
#     'c': 3 * torch.ones((3, 2))  # 3个类型为'c'的节点，每个节点有2维嵌入
# }
# # 边字典
# edge_dict = {
#     ('a', 'a-b', 'b'): torch.stack([torch.randint(0, 10, (15,)), torch.randint(0, 5, (15,))]),  # 15条'a-b'类型的边
#     ('a', 'a-c', 'c'): torch.stack([torch.randint(0, 10, (10,)), torch.randint(0, 3, (10,))]),  # 10条'a-c'类型的边
#     ('c', 'c-b', 'b'): torch.stack([torch.randint(0, 3, (5,)), torch.randint(0, 5, (5,))]),
#     ('b', 'b-a', 'a'): torch.stack([torch.randint(0, 5, (15,)), torch.randint(0, 10, (15,))]),
# }
# config.node_type_list = list(node_emb_dict.keys())
# config.edge_type_list = list(edge_dict.keys())
# config.num_node = 0
# config.encoder_in_dim = {}
# for node_type, node_emb in node_emb_dict.items():
#     config.num_node += node_emb.shape[0]
#     config.encoder_in_dim[node_type] = node_emb.shape[1]
#
# config.metapath_dict = {'a': [[('a', 'a-b', 'b'), ('b', 'b-a', 'a')]],
#                         'b': [[('b', 'b-a', 'a'), ('a', 'a-c', 'c'), ('c', 'c-b', 'b')]]}
# print()
# cob = SeHGNN(config=config)
# zdict = cob.forward(node_emb_dict, edge_dict)
# print()
