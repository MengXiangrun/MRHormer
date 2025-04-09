import pandas as pd
import torch_scatter
import torch
import torch_geometric as PyG
import numpy as np
from transformer import myTransformer


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


class TopologyRobustLocalAttention(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_head,
                 node_type_list,
                 edge_type_list,
                 is_selfloop):
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
        self.is_selfloop = is_selfloop
        if self.is_selfloop:
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

        self.attention_dict = None
        self.edge_index_dict = None

        self.attention_type = 'sigmoid'  # softmax

    def reset_parameters(self):
        for edge_type in self.edge_type_list:
            torch.nn.init.kaiming_uniform_(self.head_weight[str(edge_type)])

    def forward(self, x_dict, edge_index_dict):
        node_emb_dict = x_dict.copy()

        if self.is_selfloop:
            for node_type, node_emb in node_emb_dict.items():
                num_node = node_emb.shape[0]
                edge_index = torch.arange(0, num_node, device=node_emb.device).view(1, -1).repeat(2, 1)
                edge_type = (node_type, 'SelfLoop', node_type)
                edge_index_dict[edge_type] = edge_index

            for edge_type in self.edge_type_list:
                assert edge_type in edge_index_dict.keys()

        for node_type, node_emb in node_emb_dict.items():
            node_emb_dict[node_type] = self.in_linear[node_type](node_emb)

        # reset edge index to homogeneous
        offset_node_emb, node_range_dict, offset_edge_index, edge_range_dict = self.to_homogeneous(
            node_emb_dict=node_emb_dict, edge_index_dict=edge_index_dict)

        # # message and aggregate
        # attention_list = list()
        # message_list = list()
        # for edge_type, edge_index in edge_index_dict.items():
        #     source_type = edge_type[0]
        #     target_type = edge_type[-1]
        #
        #     source_index = edge_index[0]
        #     target_index = edge_index[1]
        #
        #     source_emb = node_emb_dict[source_type]
        #     target_emb = node_emb_dict[target_type]
        #
        #     k_emb = self.k_linear[str(edge_type)](source_emb)[source_index]
        #     q_emb = self.q_linear[str(edge_type)](target_emb)[target_index]
        #
        #     attention = torch.cat(tensors=[k_emb, q_emb], dim=1)
        #     attention = self.a_linear[str(edge_type)](attention)
        #     attention = attention.view(-1, self.num_head, self.head_dim)
        #     attention = attention * self.head_weight[str(edge_type)]
        #     attention_list.append(attention)
        #
        #     v_emb = self.v_linear[str(edge_type)](source_emb)[source_index]
        #     v_emb = v_emb.view(-1, self.num_head, self.head_dim)
        #     message_list.append(v_emb)

        # message and aggregate
        attention_dict = dict()
        message_dict = dict()
        for edge_type, edge_index in edge_index_dict.items():
            source_type = edge_type[0]
            target_type = edge_type[-1]

            source_index = edge_index[0]
            target_index = edge_index[1]

            source_emb = node_emb_dict[source_type]
            target_emb = node_emb_dict[target_type]

            k_emb = self.k_linear[str(edge_type)](source_emb)[source_index]
            q_emb = self.q_linear[str(edge_type)](target_emb)[target_index]

            attention = torch.cat(tensors=[k_emb, q_emb], dim=1)
            attention = self.a_linear[str(edge_type)](attention)
            attention = attention.view(-1, self.num_head, self.head_dim)
            attention = attention * self.head_weight[str(edge_type)]
            attention_dict[edge_type] = attention

            v_emb = self.v_linear[str(edge_type)](source_emb)[source_index]
            v_emb = v_emb.view(-1, self.num_head, self.head_dim)
            message_dict[edge_type] = v_emb

        # aggregate
        attention = torch.cat(list(attention_dict.values()), dim=0)
        message = torch.cat(list(message_dict.values()), dim=0)

        # aggregate
        num_nodes = offset_node_emb.shape[0]
        source_index = offset_edge_index[0]
        target_index = offset_edge_index[1]

        if self.attention_type == 'sigmoid':
            attention = torch.nn.functional.sigmoid(attention)
        if self.attention_type == 'softmax':
            attention = PyG.utils.softmax(src=attention, index=target_index, num_nodes=num_nodes, dim=0)

        message = attention * message
        message = message.view(-1, self.hidden_dim)
        node_emb = torch_scatter.scatter_add(src=message, index=target_index, dim=0, dim_size=num_nodes)
        del message

        # node type
        for node_type, type_range in node_range_dict.items():
            start, end = type_range
            node_emb_dict[node_type] = node_emb[start:end + 1, :]

        for node_type, node_emb in node_emb_dict.items():
            node_emb = self.out_linear[node_type](node_emb)
            node_emb_dict[node_type] = node_emb

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

    def visualization(self, x_dict, edge_index_dict, index2node_dict, dataset_name, data_type):
        node_emb_dict = x_dict.copy()

        if self.is_selfloop:
            for node_type, node_emb in node_emb_dict.items():
                num_node = node_emb.shape[0]
                edge_index = torch.arange(0, num_node, device=node_emb.device).view(1, -1).repeat(2, 1)
                edge_type = (node_type, 'SelfLoop', node_type)
                edge_index_dict[edge_type] = edge_index

            for edge_type in self.edge_type_list:
                assert edge_type in edge_index_dict.keys()

        for node_type, node_emb in node_emb_dict.items():
            node_emb_dict[node_type] = self.in_linear[node_type](node_emb)

        # reset edge index to homogeneous
        offset_node_emb, node_range_dict, offset_edge_index, edge_range_dict = self.to_homogeneous(
            node_emb_dict=node_emb_dict, edge_index_dict=edge_index_dict)

        attention_dict = dict()
        for edge_type, edge_index in edge_index_dict.items():
            source_type = edge_type[0]
            target_type = edge_type[-1]

            source_index = edge_index[0]
            target_index = edge_index[1]

            source_emb = node_emb_dict[source_type]
            target_emb = node_emb_dict[target_type]

            k_emb = self.k_linear[str(edge_type)](source_emb)[source_index]
            q_emb = self.q_linear[str(edge_type)](target_emb)[target_index]

            attention = torch.cat(tensors=[k_emb, q_emb], dim=1)
            attention = self.a_linear[str(edge_type)](attention)
            attention = attention.view(-1, self.num_head, self.head_dim)
            attention = attention * self.head_weight[str(edge_type)]
            attention_dict[edge_type] = attention

        # aggregate
        attention = torch.cat(list(attention_dict.values()), dim=0)

        # aggregate
        num_nodes = offset_node_emb.shape[0]
        source_index = offset_edge_index[0]
        target_index = offset_edge_index[1]

        if self.attention_type == 'sigmoid':
            attention = torch.nn.functional.sigmoid(attention)
        if self.attention_type == 'softmax':
            attention = PyG.utils.softmax(src=attention, index=target_index, num_nodes=num_nodes, dim=0)

        for edge_type, edge_range in edge_range_dict.items():
            start, end = edge_range
            attention_dict[edge_type] = attention[start, end + 1]

        for edge_type in edge_index_dict.items():
            source_type, target_type = edge_type[0], edge_type[-1]
            num_source = node_emb_dict[source_type].shape[0]
            num_target = node_emb_dict[target_type].shape[0]
            num_edge = edge_index_dict[edge_type].shape[1]

            attention = attention_dict[edge_type]
            attention = attention.reshape(num_edge, -1)
            attention = torch.mean(attention, dim=1)

            edge_index = edge_index_dict[edge_type].T.detach().cpu().numpy()
            attention = attention.detach().cpu().numpy()

            edge_index = pd.DataFrame(edge_index)
            attention = pd.DataFrame(attention)

            edge_attention = pd.concat([edge_index, attention], columns=[source_type, target_type, 'attention'])

            adj_matrix = np.zeros((num_source, num_target), dtype=int)

            for index, row in edge_attention.iterrows():
                source = row[source_type]
                target = row[target_type]
                weight = row['attention']
                adj_matrix[source, target] = weight

            source_node_name = list(index2node_dict[source_type].values())
            target_node_name = list(index2node_dict[target_type].values())

            adj_matrix = pd.DataFrame(adj_matrix, index=source_node_name, columns=target_node_name)

            filename = f'{dataset_name}_{data_type}_{source_type}_{target_node_name}_{self.attention_type}_local_att.xlsx'
            adj_matrix.to_excel(filename, index=True)


class MultiRelationGlobalAttention(torch.nn.Module):
    def __init__(self, config, in_dim, hidden_dim, out_dim, node_type_list, edge_type_list, num_head, self_layers):
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
        for node_type in node_type_list:
            self.in_linear[node_type] = Linear(self.hidden_dim)
            self.out_linear[node_type] = Linear(self.out_dim)
            self.global_attention[node_type] = myTransformer(config=config)
            # self.global_attention[node_type] = torch.nn.Transformer(d_model=self.hidden_dim,
            #                                                         num_encoder_layers=self_layers,
            #                                                         num_decoder_layers=self_layers,
            #                                                         batch_first=True,)

        self.target_source_dict = {}
        for edge_type in edge_type_list:
            source_type = edge_type[0]
            target_type = edge_type[-1]
            self.target_source_dict.setdefault(target_type, [])
            self.target_source_dict[target_type].append(source_type)

    def forward(self, x_dict):
        h_dict = x_dict.copy()
        for node_type, h in h_dict.items():
            h_dict[node_type] = self.in_linear[node_type](h)

        # According to relation
        # In the meta-relation source nodes and target nodes are doing transformer interaction
        node_emb_dict = dict()
        for target_type, source_type_list in self.target_source_dict.items():
            source_emb_list = []
            for source_type in source_type_list:
                source_emb = h_dict[source_type]
                source_emb_list.append(source_emb)

            source_emb = torch.cat(source_emb_list, dim=0)
            target_emb = h_dict[target_type]

            target_emb, attention = self.global_attention[target_type](source_emb, target_emb)
            # target_emb = self.global_attention[target_type](source_emb, target_emb)
            # target_emb = self.global_attention[target_type](key=source_emb, query=target_emb, value=source_emb)[0]

            node_emb_dict[target_type] = target_emb

        for node_type, node_emb in node_emb_dict.items():
            node_emb_dict[node_type] = self.out_linear[node_type](node_emb)

        return node_emb_dict


class MRHormer(torch.nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        self.config = config
        self.node_type_list = config.node_type_list
        self.edge_type_list = config.edge_type_list
        self.is_selfloop = config.is_selfloop

        self.hidden_dim = config.encoder_hidden_dim
        self.num_local_head = config.num_local_head
        self.num_global_head = config.num_global_head

        self.num_local_layer = config.num_local_layer
        self.num_global_self_layer = config.num_global_self_layer

        if isinstance(self.node_type_list, list) is False:
            self.node_type_list = list(self.node_type_list)
        if isinstance(self.edge_type_list, list) is False:
            self.edge_type_list = list(self.edge_type_list)

        if self.is_selfloop:
            for node_type in self.node_type_list:
                edge_type = (node_type, 'SelfLoop', node_type)
                if edge_type not in self.edge_type_list:
                    self.edge_type_list.append(edge_type)

        # Part 1 Heterogeneous Attention Network Message Passing GNN
        self.TRLA = torch.nn.ModuleList()
        for layer in range(self.num_local_layer):
            conv = TopologyRobustLocalAttention(in_dim=-1,
                                                hidden_dim=self.hidden_dim,
                                                out_dim=self.hidden_dim,
                                                num_head=self.num_local_head,
                                                is_selfloop=self.is_selfloop,
                                                node_type_list=self.node_type_list,
                                                edge_type_list=self.edge_type_list)
            self.TRLA.append(conv)

        # Part 2 Relation-based Global Attention Module
        if self.num_global_self_layer > 0:
            self.MRGA = MultiRelationGlobalAttention(in_dim=-1,
                                                     hidden_dim=self.hidden_dim,
                                                     out_dim=self.hidden_dim,
                                                     num_head=self.num_global_head,
                                                     self_layers=self.num_global_self_layer,
                                                     node_type_list=self.node_type_list,
                                                     edge_type_list=self.edge_type_list,
                                                     config=config)
        else:
            self.MRGA = None

    def forward(self, x_dict, edge_index_dict):
        # local
        if len(self.TRLA) > 0:
            h_local_dict = x_dict.copy()
            for conv in self.TRLA:
                h_local_dict = conv.forward(h_local_dict, edge_index_dict)

        # global
        if self.MRGA is not None:
            h_global_dict = x_dict.copy()
            h_global_dict = self.MRGA.forward(x_dict=h_global_dict)

        if len(self.TRLA) > 0 and self.MRGA is None:
            h_dict = h_local_dict

        if len(self.TRLA) == 0 and self.MRGA is not None:
            h_dict = h_global_dict

        if len(self.TRLA) > 0 and self.MRGA is not None:
            h_dict = dict()
            for node_type in x_dict.keys():
                h_local = h_local_dict[node_type]
                h_global = h_global_dict[node_type]
                h_dict[node_type] = torch.cat([h_local, h_global], dim=1)

        return h_dict
