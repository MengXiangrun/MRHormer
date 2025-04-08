# Graph Inductive Link Prediction Baseline Models
import torch_geometric as PyG
import torch
import torch.nn as nn
import torch.nn.functional as F

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
