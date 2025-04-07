# Graph Inductive Link Prediction Baseline Models
import torch_geometric as PyG
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F


# DEAL [IJCAI2020]
# Inductive Link Prediction for Nodes Having Only Attribute Information
# https://arxiv.org/abs/2007.08053
# https://github.com/working-yuhao/DEAL
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


class Hidden_Layer(nn.Module):  # Hidden Layer, Binary classification
    def __init__(self, emb_dim, device, BCE_mode, mode='all', dropout_p=0.3):
        super(Hidden_Layer, self).__init__()
        self.emb_dim = emb_dim
        self.mode = mode
        self.device = device
        self.BCE_mode = BCE_mode
        self.Linear1 = nn.Linear(self.emb_dim * 2, self.emb_dim).to(self.device)
        self.Linear2 = nn.Linear(self.emb_dim, 32).to(self.device)
        self.linear4 = nn.Linear(14696, 14696).to(self.device)
        x_dim = 1
        self.Linear3 = nn.Linear(32, x_dim).to(self.device)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.pdist = nn.PairwiseDistance(p=2, keepdim=True)
        self.softmax = nn.Softmax(dim=1)
        self.elu = nn.ELU()
        assert (self.mode in ['all', 'cos', 'dot', 'pdist']), "Wrong mode type"

    def forward(self, f_embs, s_embs):

        if self.mode == 'all':
            x = torch.cat([f_embs, s_embs], dim=1)
            x = F.rrelu(self.Linear1(x))
            x = F.rrelu(self.Linear2(x))
            x = F.rrelu(self.Linear3(x))
            cos_x = self.cos(f_embs, s_embs).unsqueeze(1)
            dot_x = torch.mul(f_embs, s_embs).sum(dim=1, keepdim=True)
            pdist_x = self.pdist(f_embs, s_embs)
            x = torch.cat([x, cos_x, dot_x, pdist_x], dim=1)
        elif self.mode == 'cos':
            x = self.cos(f_embs, s_embs).unsqueeze(1)
        elif self.mode == 'dot':
            x = torch.mul(f_embs, s_embs).sum(dim=1, keepdim=True)
        elif self.mode == 'pdist':
            x = self.pdist(f_embs, s_embs)

        if self.BCE_mode:
            return x.squeeze()
        else:
            x = self.linear_output(x)
            x = F.rrelu(x)
            return x

    def evaluate(self, f_embs, s_embs):
        if self.mode == 'all':
            x = torch.cat([f_embs, s_embs], dim=1)
            x = F.rrelu(self.Linear1(x))
            x = F.rrelu(self.Linear2(x))
            x = F.rrelu(self.Linear3(x))
            cos_x = self.cos(f_embs, s_embs).unsqueeze(1)
            dot_x = torch.mul(f_embs, s_embs).sum(dim=1, keepdim=True)
            pdist_x = self.pdist(f_embs, s_embs)
            x = torch.cat([x, cos_x, dot_x, pdist_x], dim=1)
        elif self.mode == 'cos':
            x = self.cos(f_embs, s_embs)
            # x = self.linear4(x)
        elif self.mode == 'dot':
            x = torch.mul(f_embs, s_embs).sum(dim=1)
        elif self.mode == 'pdist':
            x = -self.pdist(f_embs, s_embs).squeeze()
        return x


class DEAL(torch.nn.Module):
    def __init__(self, num_node, in_dim, hidden_dim, out_dim, node_type_list):
        super(DEAL, self).__init__()
        self.attr_emb = nn.Embedding(num_node, out_dim)
        self.attr_num = in_dim
        self.node_emb = nn.Embedding(num_node, out_dim)
        self.in_linear = torch.nn.ModuleDict()
        for node_type in node_type_list:
            self.in_linear[node_type] = PyG.nn.Linear(in_channels=in_dim,
                                                      out_channels=hidden_dim,
                                                      bias=True,
                                                      weight_initializer='kaiming_uniform',
                                                      bias_initializer=None)

    def forward(self, x_dict, edge_index_dict):
        h_dict = dict()
        for node_type, x in x_dict.items():
            h = self.in_linear[node_type](x)
            h_dict[node_type] = h
        h, node_range_dict, edge_index = to_homogeneous(node_emb_dict=h_dict, edge_dict=edge_index_dict)
        h2 = self.node_emb(torch.arange(0, h.size(0)).to('cuda'))
        x = torch.mm(h, self.attr_emb(torch.arange(h.size(1)).to(self.attr_emb.weight.device)))
        z_dict = dict()
        for node_type, node_range in node_range_dict.items():
            start_index, end_index = node_range
            z = x[start_index:end_index + 1, :]
            z_dict[node_type] = z

        z2_dict = dict()
        for node_type, node_range in node_range_dict.items():
            start_index, end_index = node_range
            z = h2[start_index:end_index + 1, :]
            z2_dict[node_type] = z

        return z_dict, z2_dict


# Graph2Feat[WWW 2023]
# Graph2Feat: Inductive Link Prediction via Knowledge Distillation
# https://dl.acm.org/doi/pdf/10.1145/3543873.3587596
# https://github.com/AhmedESamy/Graph2Feat
class Graph2Feat_Student(nn.Module):
    def __init__(self, config):
        super().__init__()
        output_dim = config.encoder_out_dim
        self.device = config.device
        self.em_dict = nn.ModuleDict()
        self.lin1 = nn.ModuleDict()
        self.bn1 = nn.ModuleDict()
        self.lin2 = nn.ModuleDict()
        self.bn2 = nn.ModuleDict()

        for nt in config.node_type_list:
            self.em_dict[nt] = None

            self.lin1[nt] = Linear(config.encoder_hidden_dim)
            self.bn1[nt] = nn.BatchNorm(config.encoder_hidden_dim)
            self.lin2[nt] = Linear(output_dim)
            self.bn2[nt] = nn.BatchNorm(output_dim)

    def forward(self, x_dict):
        for node_type in x_dict:
            if self.em_dict[node_type] is not None:
                x_dict[node_type] = self.em_dict[node_type](x_dict[node_type].to(self.device).squeeze())

            x_dict[node_type] = self.lin1[node_type](x_dict[node_type].to(self.device))
            x_dict[node_type] = self.bn1[node_type](x_dict[node_type])
            x_dict[node_type] = self.lin2[node_type](x_dict[node_type])
            x_dict[node_type] = self.bn2[node_type](x_dict[node_type])

        return x_dict
