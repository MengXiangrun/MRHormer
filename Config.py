import pandas as pd

class Config():
    def __init__(self):
        self.encoder_in_dim = -1 # dict
        self.encoder_hidden_dim = 128
        self.encoder_out_dim = 64
        self.decoder_in_dim = 64
        self.decoder_hidden_dim = 32
        self.decoder_out_dim = 1
        self.num_local_layer = 2
        self.num_global_self_layer = 2
        self.num_global_cross_layer = 2

        self.num_local_head = 8
        self.num_global_head = 8

        self.local_dropout_probability = 0.01
        self.global_dropout_probability = 0.01

        self.node_type_list = []
        self.edge_type_list = []
        self.optimizer_learning_rate = 0.001
        self.optimizer_weight_decay = 0.001

        self.num_node = None
        self.is_residual_connection = False
        self.metapath_dict=None
        self.is_selfloop=True
        self.dataset_name=True
        self.target_node_type_list =['herb', 'target']

        self.average_use_time_per_epoch = None
        self.use_time_dict=dict()


    def save_to_excel(self, file_path):
        data = {key: [value] for key, value in vars(self).items()}
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)











