# wait for 2025 ACM SIGKDD result
# Association for Computing Machinery's Special Interest Group on Knowledge Discovery and Data Mining
# Your Active Consoles
# KDD 2025 Research Track February Authors

import torch_scatter
import torch
import torch_geometric as PyG


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


class HGANConv(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_head,
                 node_type_list,
                 edge_type_list,
                 is_self):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_head = num_head
        self.head_dim = self.hidden_dim // self.num_head
        assert self.hidden_dim % self.num_head == 0

        if isinstance(node_type_list, list) is False:
            node_type_list = list(node_type_list)
        if isinstance(edge_type_list, list) is False:
            edge_type_list = list(edge_type_list)
        self.node_type_list = node_type_list
        self.edge_type_list = edge_type_list
        self.is_self = is_self
        if self.is_self:
            for node_type in self.node_type_list:
                edge_type = (node_type, 'SelfLoop', node_type)
                if edge_type not in self.edge_type_list:
                    self.edge_type_list.append(edge_type)

        # forward
        self.in_linear = torch.nn.ModuleDict()
        self.out_linear = torch.nn.ModuleDict()
        # self.mlp = torch.nn.ModuleDict()
        for node_type in self.node_type_list:
            self.in_linear[node_type] = Linear(self.hidden_dim)
            self.out_linear[node_type] = Linear(self.out_dim)

        self.k_linear = torch.nn.ModuleDict()
        self.q_linear = torch.nn.ModuleDict()
        self.v_linear = torch.nn.ModuleDict()
        self.a_linear = torch.nn.ModuleDict()
        self.head_weight = torch.nn.ParameterDict()
        for edge_type in self.edge_type_list:
            self.k_linear[str(edge_type)] = Linear(self.hidden_dim)
            self.q_linear[str(edge_type)] = Linear(self.hidden_dim)
            self.v_linear[str(edge_type)] = Linear(self.hidden_dim)
            self.a_linear[str(edge_type)] = Linear(self.hidden_dim)
            self.head_weight[str(edge_type)] = torch.nn.Parameter(torch.empty(1, self.num_head, self.head_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for edge_type in self.edge_type_list:
            torch.nn.init.kaiming_uniform_(self.head_weight[str(edge_type)])

    def forward(self, x_dict, edge_index_dict):

        return node_emb_dict

    def to_homogeneous(self, node_emb_dict, edge_index_dict):
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
        edge_range_dict = dict()
        start = 0
        end = 0
        for edge_type, edge_index in edge_index_dict.items():
            source_type, target_type = edge_type[0], edge_type[-1]

            source_offset = offset_dict[source_type]
            target_offset = offset_dict[target_type]

            offset_edge_index = edge_index.clone()

            offset_edge_index[0] = offset_edge_index[0] + source_offset
            offset_edge_index[1] = offset_edge_index[1] + target_offset

            num_edge = offset_edge_index.shape[1]
            end = start + num_edge - 1
            edge_range = (start, end)
            edge_range_dict[edge_type] = edge_range
            start = end

            offset_edge_index_list.append(offset_edge_index)

        offset_edge_index = torch.cat(offset_edge_index_list, dim=1)

        return node_emb, node_range_dict, offset_edge_index, edge_range_dict


class GlobalAttention(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, node_type_list, edge_type_list, num_head, self_layers):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.num_head = num_head

        self.node_type_list = node_type_list
        self.edge_type_list = edge_type_list

        self.in_linear = torch.nn.ModuleDict()
        self.out_linear = torch.nn.ModuleDict()
        self.global_attention = torch.nn.ModuleDict()

    def forward(self, x_dict):
        return node_emb_dict


class MRNormer(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 node_type_list,
                 edge_type_list,
                 num_layers,
                 num_head,
                 self_layers,
                 is_self):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_head = num_head

        if isinstance(node_type_list, list) is False:
            node_type_list = list(node_type_list)
        if isinstance(edge_type_list, list) is False:
            edge_type_list = list(edge_type_list)
        self.node_type_list = node_type_list
        self.edge_type_list = edge_type_list
        self.is_self = is_self
        if self.is_self:
            for node_type in self.node_type_list:
                edge_type = (node_type, 'SelfLoop', node_type)
                if edge_type not in self.edge_type_list:
                    self.edge_type_list.append(edge_type)

        # Part 1 Heterogeneous Attention Network Message Passing GNN
        self.HGNN = torch.nn.ModuleList()
        for layer in range(num_layers):
            conv = HGANConv(in_dim=-1,
                            hidden_dim=self.hidden_dim,
                            out_dim=self.hidden_dim,
                            num_head=self.num_head,
                            is_self=self.is_self,
                            node_type_list=self.node_type_list,
                            edge_type_list=self.edge_type_list)
            self.HGNN.append(conv)

        # Part 2 Relation-based Global Attention Module
        if self_layers > 0:
            self.GATT = GlobalAttention(in_dim=-1,
                                        hidden_dim=self.hidden_dim,
                                        out_dim=self.hidden_dim,
                                        num_head=self.num_head,
                                        self_layers=self_layers,
                                        node_type_list=self.node_type_list,
                                        edge_type_list=self.edge_type_list)
        else:
            self.GATT = None



    def forward(self, x_dict, edge_index_dict):
        # local
        if len(self.HGNN) > 0:
            h_local_dict = x_dict.copy()
            for conv in self.HGNN:
                h_local_dict = conv.forward(h_local_dict, edge_index_dict)

        # global
        if self.GATT is not None:
            h_global_dict = x_dict.copy()
            h_global_dict = self.GATT.forward(x_dict=h_global_dict)


        if len(self.HGNN) > 0 and self.GATT is None:
            h_dict = h_local_dict

        if len(self.HGNN) == 0 and self.GATT is not None:
            h_dict = h_global_dict

        if len(self.HGNN) > 0 and self.GATT is not None:
            h_dict = dict()
            for node_type in x_dict.keys():
                h_local = h_local_dict[node_type]
                h_global = h_global_dict[node_type]
                h_dict[node_type] = torch.cat([h_local, h_global], dim=1)

        return h_dict




