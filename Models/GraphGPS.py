import torch
from torch.nn import (
    ModuleList,
)
import torch_geometric as PyG
from torch_geometric.nn import GATConv, GPSConv



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


# GraphGPS
class GraphGPS(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_linear = torch.nn.ModuleDict()
        for node_type in config.node_type_list:
            self.in_linear[node_type] = Linear(config.encoder_out_dim)

        self.GPSConvs = ModuleList()
        for _ in range(1):
            gnn = GATConv(in_channels=config.encoder_in_dim,
                          out_channels=config.encoder_out_dim//config.num_local_head,
                          heads=config.num_local_head)

            conv = GPSConv(channels=config.encoder_out_dim,
                           conv=gnn,
                           heads=config.num_local_head,
                           attn_type='multihead')
            self.GPSConvs.append(conv)

    def forward(self, node_emb_dict, edge_dict):
        h_dict = dict()
        for node_type, x in node_emb_dict.items():
            h_dict[node_type] = self.in_linear[node_type](x)

        h, node_range_dict, edge = to_homogeneous(node_emb_dict=h_dict, edge_dict=edge_dict)
        for conv in self.GPSConvs:
            h = conv(h, edge)
        z_dict = to_heterogeneous_node_embedding(node_emb=h, node_range_dict=node_range_dict)

        return z_dict


#
from Config import Config

config = Config()
# 节点嵌入字典
node_emb_dict = {
    'a': torch.ones((10, 2)),  # 10个类型为'a'的节点，每个节点有2维嵌入
    'b': 2 * torch.ones((5, 2)),  # 5个类型为'b'的节点，每个节点有2维嵌入
    'c': 3 * torch.ones((3, 2))  # 3个类型为'c'的节点，每个节点有2维嵌入
}
# 边字典
edge_dict = {
    ('a', 'a-b', 'b'): torch.stack([torch.randint(0, 10, (15,)), torch.randint(0, 5, (15,))]),  # 15条'a-b'类型的边
    ('a', 'a-c', 'c'): torch.stack([torch.randint(0, 10, (10,)), torch.randint(0, 3, (10,))]),  # 10条'a-c'类型的边
    ('c', 'c-b', 'b'): torch.stack([torch.randint(0, 3, (5,)), torch.randint(0, 5, (5,))])  # 5条'c-b'类型的边
}
config.node_type_list = list(node_emb_dict.keys())
config.edge_type_list = list(edge_dict.keys())
config.num_node = 0
for node_type, node_emb in node_emb_dict.items():
    config.num_node += node_emb.shape[0]

print()
cob = GraphGPS(config=config)
zdict = cob.forward(node_emb_dict, edge_dict)
print()