import pandas as pd

class Config():
    def __init__(self,
                 encoder_in_dim=-1, encoder_hidden_dim=128, encoder_out_dim=64,
                 decoder_in_dim=64, decoder_hidden_dim=32, decoder_out_dim=1,
                 num_local_layer=2,
                 num_global_self_layer=2, num_global_cross_layer=2,
                 num_local_head=8, num_global_head=8,
                 local_dropout_probability=0.0,
                 global_dropout_probability=0.0,

                 node_type_list=[], edge_type_list=[],
                 optimizer_learning_rate=0.001, optimizer_weight_decay=0.001):
        self.encoder_in_dim = encoder_in_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_out_dim = encoder_out_dim
        self.decoder_in_dim = decoder_in_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_out_dim = decoder_out_dim
        self.num_local_layer = num_local_layer
        self.num_global_self_layer = num_global_self_layer
        self.num_global_cross_layer = num_global_cross_layer

        self.num_local_head = num_local_head
        self.num_global_head = num_global_head

        self.local_dropout_probability = local_dropout_probability
        self.global_dropout_probability = global_dropout_probability

        self.node_type_list = node_type_list
        self.edge_type_list = edge_type_list
        self.optimizer_learning_rate = optimizer_learning_rate
        self.optimizer_weight_decay = optimizer_weight_decay

        self.num_node = None
        self.is_residual_connection = False
        self.metapath_dict

    def save_to_excel(self, file_path):
        data = {key: [value] for key, value in vars(self).items()}
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)











