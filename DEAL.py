# Graph Inductive Link Prediction Baseline Models
import torch_geometric as PyG
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DEAL(nn.Module):  # Hidden Layer, Binary classification
    def __init__(self, config, ):
        super(DEAL, self).__init__()
        self.in_linear = torch.nn.ModuleDict()
        for node_type in config.node_type_list:
            self.in_linear[node_type] = Linear(config.encoder_hidden_dim)
        self.emb_dim = config.encoder_hidden_dim
        self.mode = 'all'
        self.Linear1 = Linear(self.emb_dim)
        self.Linear2 = Linear(self.emb_dim)
        self.Linear3 = Linear(config.encoder_out_dim)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.pdist = nn.PairwiseDistance(p=2, keepdim=True)

    def forward(self, x_dict, edge_dict):
        h_dict = dict()
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.in_linear[node_type](x)

        h, node_range_dict, edge_index = to_homogeneous(node_emb_dict=h_dict, edge_dict=edge_dict)

        f_embs = h.clone()
        s_embs = h.clone()
        x = torch.cat([f_embs, s_embs], dim=1)
        x = F.rrelu(self.Linear1(x))
        x = F.rrelu(self.Linear2(x))
        x = F.rrelu(self.Linear3(x))
        cos_x = self.cos(f_embs, s_embs).unsqueeze(1)
        dot_x = torch.mul(f_embs, s_embs).sum(dim=1, keepdim=True)
        pdist_x = self.pdist(f_embs, s_embs)
        z = torch.cat([x, cos_x, dot_x, pdist_x], dim=1)
        z_dict = to_heterogeneous_node_embedding(node_emb=z, node_range_dict=node_range_dict)

        return z_dict


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
#     ('c', 'c-b', 'b'): torch.stack([torch.randint(0, 3, (5,)), torch.randint(0, 5, (5,))])  # 5条'c-b'类型的边
# }
# config.node_type_list = list(node_emb_dict.keys())
# config.edge_type_list = list(edge_dict.keys())
# config.num_node = 0
# for node_type, node_emb in node_emb_dict.items():
#     config.num_node += node_emb.shape[0]
#
# print()
# m = DEAL(config=config)
# zdict = m.forward(x_dict=node_emb_dict, edge_dict=edge_dict)
# print()
