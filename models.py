import torch_geometric as PyG
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GATv2Conv, LGConv, LightGCN
from torch_geometric.nn import HANConv, HGTConv
import torch
import copy


class GAT_T(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, node_type_list):
        super().__init__()
        self.in_linear = torch.nn.ModuleDict()
        self.in_linear2 = torch.nn.ModuleDict()
        for node_type in node_type_list:
            self.in_linear[node_type] = Linear(hidden_dim)
            self.in_linear2[node_type] = Linear(hidden_dim)

        self.conv_list = torch.nn.ModuleList()
        if num_layers == 1:
            conv = GATConv(hidden_dim, out_dim)
            self.conv_list.append(conv)
        if num_layers == 2:
            conv = GATConv(hidden_dim, hidden_dim)
            self.conv_list.append(conv)
            conv = GATConv(hidden_dim, hidden_dim)
            self.conv_list.append(conv)

        self.MultiheadAttention = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1)

    def forward(self, x_dict, edge_index_dict):
        l_h_dict = x_dict.copy()
        for node_type, h in l_h_dict.items():
            l_h_dict[node_type] = self.in_linear[node_type](h)
        l_h, node_range_dict, edge_index = to_homogeneous(node_emb_dict=l_h_dict, edge_index_dict=edge_index_dict)

        for conv in self.conv_list:
            l_h = conv.forward(x=l_h, edge_index=edge_index)

        g_h_dict = x_dict.copy()
        g_h = []
        for node_type, h in g_h_dict.items():
            h = self.in_linear2[node_type](h)
            g_h.append(h)
        g_h = torch.cat(g_h, dim=0)
        g_h = self.MultiheadAttention(g_h, g_h, g_h)[0]

        z_dict = dict()
        for node_type, node_range in node_range_dict.items():
            start_index, end_index = node_range
            l_z = l_h[start_index:end_index + 1, :]
            g_z = g_h[start_index:end_index + 1, :]
            z_dict[node_type] = torch.cat([l_z, g_z], dim=1)

        return z_dict


def to_homogeneous(node_emb_dict, edge_index_dict):
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
    offset_edge_index_list = list()
    for edge_type, edge_index in edge_index_dict.items():
        source_type, target_type = edge_type[0], edge_type[-1]

        source_offset = offset_dict[source_type]
        target_offset = offset_dict[target_type]

        offset_edge_index = edge_index.clone()

        offset_edge_index[0] = offset_edge_index[0] + source_offset
        offset_edge_index[1] = offset_edge_index[1] + target_offset

        offset_edge_index_list.append(offset_edge_index)

    offset_edge_index = torch.cat(offset_edge_index_list, dim=1)

    return node_emb, node_range_dict, offset_edge_index


class Linear(torch.nn.Module):
    def __init__(self, out_dim):
        super(Linear, self).__init__()
        self.out_dim = out_dim
        self.linear = PyG.nn.Linear(in_channels=-1,
                                    out_channels=self.out_dim,
                                    weight_initializer='kaiming_uniform',
                                    bias=True,
                                    bias_initializer=None)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)


# Homogeneous GNN Baselines
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, node_type_list):
        super().__init__()
        self.in_linear = torch.nn.ModuleDict()
        for node_type in node_type_list:
            self.in_linear[node_type] = Linear(hidden_dim)

        self.conv_list = torch.nn.ModuleList()

        if num_layers == 1:
            conv = GCNConv(hidden_dim, out_dim)
            self.conv_list.append(conv)

        if num_layers == 2:
            conv = GCNConv(hidden_dim, hidden_dim)
            self.conv_list.append(conv)
            conv = GCNConv(hidden_dim, out_dim)
            self.conv_list.append(conv)

    def forward(self, x_dict, edge_index_dict):
        h_dict = dict()
        for node_type, x in x_dict.items():
            h = self.in_linear[node_type](x)
            h_dict[node_type] = h

        h, node_range_dict, edge_index = to_homogeneous(node_emb_dict=h_dict, edge_index_dict=edge_index_dict)

        for conv in self.conv_list:
            h = conv.forward(x=h, edge_index=edge_index)

        z_dict = dict()
        for node_type, node_range in node_range_dict.items():
            start_index, end_index = node_range
            z = h[start_index:end_index + 1, :]
            z_dict[node_type] = z

        return z_dict


class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, node_type_list):
        super().__init__()
        self.in_linear = torch.nn.ModuleDict()
        for node_type in node_type_list:
            self.in_linear[node_type] = Linear(hidden_dim)
            self.in_linear[node_type].reset_parameters()

        self.conv_list = torch.nn.ModuleList()

        if num_layers == 1:
            conv = GATConv(hidden_dim, out_dim)
            self.conv_list.append(conv)

        if num_layers == 2:
            conv = GATConv(hidden_dim, hidden_dim)
            self.conv_list.append(conv)
            conv = GATConv(hidden_dim, out_dim)
            self.conv_list.append(conv)

    def forward(self, x_dict, edge_index_dict):
        h_dict = dict()
        for node_type, x in x_dict.items():
            h = self.in_linear[node_type](x)
            h_dict[node_type] = h

        h, node_range_dict, edge_index = to_homogeneous(node_emb_dict=h_dict, edge_index_dict=edge_index_dict)

        for conv in self.conv_list:
            h = conv.forward(x=h, edge_index=edge_index)

        z_dict = dict()
        for node_type, node_range in node_range_dict.items():
            start_index, end_index = node_range
            z = h[start_index:end_index + 1, :]
            z_dict[node_type] = z

        return z_dict


class GATv2(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, node_type_list):
        super().__init__()
        self.in_linear = torch.nn.ModuleDict()
        for node_type in node_type_list:
            self.in_linear[node_type] = Linear(hidden_dim)
            self.in_linear[node_type].reset_parameters()

        self.conv_list = torch.nn.ModuleList()

        if num_layers == 1:
            conv = GATv2Conv(hidden_dim, out_dim)
            self.conv_list.append(conv)

        if num_layers == 2:
            conv = GATv2Conv(hidden_dim, hidden_dim)
            self.conv_list.append(conv)
            conv = GATv2Conv(hidden_dim, out_dim)
            self.conv_list.append(conv)

    def forward(self, x_dict, edge_index_dict):
        h_dict = dict()
        for node_type, x in x_dict.items():
            h = self.in_linear[node_type](x)
            h_dict[node_type] = h

        h, node_range_dict, edge_index = to_homogeneous(node_emb_dict=h_dict, edge_index_dict=edge_index_dict)

        for conv in self.conv_list:
            h = conv.forward(x=h, edge_index=edge_index)

        z_dict = dict()
        for node_type, node_range in node_range_dict.items():
            start_index, end_index = node_range
            z = h[start_index:end_index + 1, :]
            z_dict[node_type] = z

        return z_dict


class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, node_type_list):
        super().__init__()
        self.in_linear = torch.nn.ModuleDict()
        for node_type in node_type_list:
            self.in_linear[node_type] = Linear(hidden_dim)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for i in range(num_layers):
            mlp = torch.nn.Sequential(
                PyG.nn.Linear(in_channels=-1,
                              out_channels=2 * hidden_dim,
                              bias=True,
                              weight_initializer='kaiming_uniform',
                              bias_initializer=None),
                torch.nn.BatchNorm1d(num_features=2 * hidden_dim),
                torch.nn.ReLU(),
                PyG.nn.Linear(in_channels=2 * hidden_dim,
                              out_channels=out_dim,
                              bias=True,
                              weight_initializer='kaiming_uniform',
                              bias_initializer=None))
            conv = GINConv(nn=mlp, train_eps=True)

            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        h_dict = dict()
        for node_type, x in x_dict.items():
            h = self.in_linear[node_type](x)
            h_dict[node_type] = h

        h, node_range_dict, edge_index = to_homogeneous(node_emb_dict=h_dict, edge_index_dict=edge_index_dict)

        for conv in self.convs:
            h = conv(h, edge_index)

        z_dict = dict()
        for node_type, node_range in node_range_dict.items():
            start_index, end_index = node_range
            z = h[start_index:end_index + 1, :]
            z_dict[node_type] = z

        return z_dict


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, node_type_list):
        super().__init__()
        self.in_linear = torch.nn.ModuleDict()
        for node_type in node_type_list:
            self.in_linear[node_type] = Linear(hidden_dim)
            self.in_linear[node_type].reset_parameters()

        self.conv_list = torch.nn.ModuleList()

        if num_layers == 1:
            conv = SAGEConv(hidden_dim, out_dim)
            self.conv_list.append(conv)

        if num_layers == 2:
            conv = SAGEConv(hidden_dim, hidden_dim)
            self.conv_list.append(conv)
            conv = SAGEConv(hidden_dim, out_dim)
            self.conv_list.append(conv)

    def forward(self, x_dict, edge_index_dict):
        h_dict = dict()
        for node_type, x in x_dict.items():
            h = self.in_linear[node_type](x)
            h_dict[node_type] = h

        h, node_range_dict, edge_index = to_homogeneous(node_emb_dict=h_dict, edge_index_dict=edge_index_dict)

        for conv in self.conv_list:
            h = conv.forward(x=h, edge_index=edge_index)

        z_dict = dict()
        for node_type, node_range in node_range_dict.items():
            start_index, end_index = node_range
            z = h[start_index:end_index + 1, :]
            z_dict[node_type] = z

        return z_dict


class LightGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, node_type_list):
        super().__init__()
        self.in_linear = torch.nn.ModuleDict()
        self.out_linear = torch.nn.ModuleDict()
        for node_type in node_type_list:
            self.in_linear[node_type] = Linear(hidden_dim)
            self.out_linear[node_type] = Linear(out_dim)
            self.in_linear[node_type].reset_parameters()

        self.conv_list = torch.nn.ModuleList()

        if num_layers == 1:
            conv = LGConv()
            self.conv_list.append(conv)

        if num_layers == 2:
            conv = LGConv()
            self.conv_list.append(conv)
            conv = LGConv()
            self.conv_list.append(conv)

    def forward(self, x_dict, edge_index_dict):
        h_dict = dict()
        for node_type, x in x_dict.items():
            h = self.in_linear[node_type](x)
            h_dict[node_type] = h

        h, node_range_dict, edge_index = to_homogeneous(node_emb_dict=h_dict, edge_index_dict=edge_index_dict)

        for conv in self.conv_list:
            h = conv.forward(x=h, edge_index=edge_index)

        z_dict = dict()
        for node_type, node_range in node_range_dict.items():
            start_index, end_index = node_range
            z = h[start_index:end_index + 1, :]
            z_dict[node_type] = z

        for node_type, z in z_dict.items():
            z = self.out_linear[node_type].forward(z)
            z_dict[node_type] = z

        return z_dict


########################################################################################################################

# Metapath-based Heterogeneous GNN Baselines

# HAN[WWW 2019] 
# Heterogeneous Graph Attention Network
class HAN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, node_type_list, edge_type_list,
                 num_layers: int = 1, num_heads: int = 1,
                 ):
        super().__init__()

    def forward(self, x_dict, edge_index_dict, metapath_dict):
        metapath_edge_index_dict = self.create_metapath_edge_index(node_feature_dict=x_dict,
                                                                   edge_index_dict=edge_index_dict,
                                                                   metapath_dict=metapath_dict)

        for conv in self.conv_list:
            z_dict = conv(x_dict=x_dict,
                          edge_index_dict=metapath_edge_index_dict)

        # output
        h_dict = dict()
        for node_type, z in z_dict.items():
            h = self.in_linear[node_type].forward(x=z)
            h_dict[node_type] = h

        return h_dict

    def create_metapath_edge_index(self, node_feature_dict, edge_index_dict, metapath_dict):
        metapath_edge_index_dict = {}

        for node_type, metapath_list in metapath_dict.items():
            metapath_index = 0
            for metapath in metapath_list:
                edge_type = metapath[0]
                source_type = edge_type[0]
                target_type = edge_type[-1]
                num_source_node = node_feature_dict[source_type].shape[0]
                num_target_node = node_feature_dict[target_type].shape[0]
                sparse_size = (num_source_node, num_target_node)

                adj1 = PyG.typing.SparseTensor.from_edge_index(edge_index=edge_index_dict[edge_type],
                                                               sparse_sizes=sparse_size)

                for edge_type in metapath[1:]:
                    source_type = edge_type[0]
                    target_type = edge_type[-1]
                    num_source_node = node_feature_dict[source_type].shape[0]
                    num_target_node = node_feature_dict[target_type].shape[0]
                    sparse_size = (num_source_node, num_target_node)

                    adj2 = PyG.typing.SparseTensor.from_edge_index(edge_index=edge_index_dict[edge_type],
                                                                   sparse_sizes=sparse_size)

                    adj1 = adj1 @ adj2

                source_index, target_index, edge_weight = adj1.coo()
                metapath_type = (metapath[0][0], f'{node_type}_metapath_{metapath_index}', metapath[-1][-1])
                metapath_edge_index_dict[metapath_type] = torch.vstack([source_index, target_index])
                metapath_index += 1

        return metapath_edge_index_dict


# SeHGNN[AAAI 2023]
# Simple and Efficient Heterogeneous Graph Neural Network
# http://arxiv.org/abs/2207.02547
# from openhgnn.models import SeHGNN


# Paths2Pair[KDD 2024] 
# Paths2Pair: Meta-path Based Link Prediction in Billion-Scale Commercial Heterogeneous Graphs
# https://dl.acm.org/doi/10.1145/3637528.3671563
# https://github.com/JQHang/Paths2Pair


########################################################################################################################

# Metapath-free Heterogeneous GNN Baselines

# HGT(WWW 2019) 
# Heterogeneous Graph Transformer
# https://arxiv.org/abs/2003.01332
# from openhgnn.models import HGT

# SimpleHGN[KDD 2021]
# Are we really making much progress? Revisiting, benchmarking,and refining heterogeneous graph neural networks
# https://dl.acm.org/doi/pdf/10.1145/3447548.3467350
# from openhgnn.models import SimpleHGN

########################################################################################################################

# Relation-based Heterogeneous GNN Baselines

# RGCN[ESWC 2018]
# Modeling Relational Data with Graph Convolutional Networks
# https://arxiv.org/abs/1703.06103
# from openhgnn.models import RGCN

# RSHN[ICDM 2019]
# Relation Structure-Aware Heterogeneous Graph Neural Network
# https://ieeexplore.ieee.org/abstract/document/8970828
# from openhgnn.models import RSHN

# RHGNN[TKDE 2021]
# Heterogeneous Graph Representation Learning with Relation Awareness
# https://arxiv.org/abs/2105.11122
# from openhgnn.models import RHGNN


########################################################################################################################

# Graph Inductive Link Prediction Baseline Models 

# DEAL [IJCAI2020] 
# Inductive Link Prediction for Nodes Having Only Attribute Information
# https://arxiv.org/abs/2007.08053
# https://github.com/working-yuhao/DEAL


# Graph2Feat[WWW 2023] 
# Graph2Feat: Inductive Link Prediction via Knowledge Distillation
# https://dl.acm.org/doi/pdf/10.1145/3543873.3587596
# https://github.com/AhmedESamy/Graph2Feat

########################################################################################################################
class EdgeDecoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_type_list):
        super().__init__()
        self.edge_type_list = edge_type_list

        self.bilinear = torch.nn.ModuleDict()
        self.linear1 = torch.nn.ModuleDict()
        self.linear2 = torch.nn.ModuleDict()
        for edge_type in self.edge_type_list:
            self.bilinear[str(edge_type)] = Linear(hidden_dim)
            self.linear1[str(edge_type)] = Linear(hidden_dim)
            self.linear2[str(edge_type)] = Linear(out_dim)

        self.elu = torch.nn.ELU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, node_emb_dict, edge_index_dict):
        prediction_dict = dict()
        for edge_type, edge_index in edge_index_dict.items():
            source_type, target_type = edge_type[0], edge_type[-1]

            source_index, target_index = edge_index

            source_node_emb = node_emb_dict[source_type]
            target_node_emb = node_emb_dict[target_type]

            source_node_emb = source_node_emb[source_index]
            target_node_emb = target_node_emb[target_index]

            edge_emb = torch.cat([source_node_emb, target_node_emb], dim=1)
            edge_emb = self.bilinear[str(edge_type)](edge_emb)
            edge_emb = self.elu(edge_emb)
            edge_emb = self.linear1[str(edge_type)](edge_emb)
            edge_emb = self.elu(edge_emb)
            edge_emb = self.linear2[str(edge_type)](edge_emb)

            prediction_probability = self.sigmoid(edge_emb)
            prediction_probability = prediction_probability.view(-1)
            prediction_dict[edge_type] = prediction_probability

        return prediction_dict
