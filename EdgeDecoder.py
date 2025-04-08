import torch
import torch_geometric as PyG


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


class EdgeDecoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.edge_type_list = config.edge_type_list

        self.bilinear = torch.nn.ModuleDict()
        self.linear1 = torch.nn.ModuleDict()
        self.linear2 = torch.nn.ModuleDict()
        for edge_type in self.edge_type_list:
            self.bilinear[str(edge_type)] = Linear(config.decoder_hidden_dim)
            self.linear1[str(edge_type)] = Linear(config.decoder_hidden_dim)
            self.linear2[str(edge_type)] = Linear(config.decoder_out_dim)

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
