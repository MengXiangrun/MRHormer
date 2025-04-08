import torch_geometric as PyG
from torch_geometric.nn.dense import Linear
from torch.nn import Sequential, Linear
import torch

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