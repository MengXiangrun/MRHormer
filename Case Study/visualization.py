import random
import numpy as np
import pandas as pd
import torch
from space4hgnn.generate_yaml import patience
from data import TCMDataset, HeterogeneousGraphData
from function import train, test, EarlyStopping, save_excel, set_seed
from EdgeDecoder import EdgeDecoder
import datetime
from Config import Config
from MRHormer import MRHormer
from time import perf_counter


class Model(torch.nn.Module):
    def __init__(self, Encoder, Decoder):
        super().__init__()
        self.encoder = Encoder
        self.decoder = Decoder


# device
device = torch.device("cpu")
set_seed()
worktype = ''  # HyPara,Ablation
# 'HeNetRW', 'TCMSP', 'HIT'
dataset_name = 'TCMSP'
dataset = TCMDataset(dataset=dataset_name)
config = Config()
config.dataset_name = dataset_name
config.num_local_layer = 2  # best 2
config.num_global_self_layer = 4  # best 4
config.encoder_hidden_dim = 128
config.optimizer_learning_rate = 0.00005
patience = 100
unseen_node_type = 'herb'
for train_val_test in dataset.data_list:
    torch.cuda.empty_cache()
    train_data, val_data, test_data = train_val_test
    if test_data.unseen_node_type == unseen_node_type:
        config.node_type_list = list(test_data.node_feature_dict.keys())
        config.edge_type_list = list(test_data.message_edge_dict.keys())
        break

encoder = MRHormer(config=config)
decoder = EdgeDecoder(config=config)
model = Model(Encoder=encoder, Decoder=decoder).to(device)
save_path = dataset_name + '_' + unseen_node_type + '_' + type(model.encoder).__name__ + '.pth'
model.load_state_dict(torch.load(save_path))
node_emb_dict, attention_dict = model.encoder.MRGA.visualization(x_dict=test_data.node_feature_dict,
                                                                 index2node_dict=test_data.index2node_dict,
                                                                 dataset_name=dataset_name, data_type='test',
                                                                 predict_edge_dict=test_data.predict_edge_dict)
node_emb_dict=model.encoder(test_data.node_feature_dict,test_data.message_edge_dict)
pred_dict = model.decoder.forward(node_emb_dict, test_data.predict_edge_dict)

edge_pred_dict = {}
for edge_type in test_data.predict_edge_dict.keys():
    source_type, target_type = edge_type[0], edge_type[-1]

    source_index2name = test_data.index2node_dict[source_type]
    target_index2name = test_data.index2node_dict[target_type]

    edge = test_data.predict_edge_dict[edge_type].detach().cpu().numpy().T
    pred = pred_dict[edge_type].reshape(-1, 1).detach().cpu().numpy()

    edge = pd.DataFrame(edge)
    pred = pd.DataFrame(pred)

    edge_pred = pd.concat([edge, pred], axis=1)
    columns = [source_type, target_type, 'score']
    edge_pred.columns = columns

    edge_pred[source_type] = edge_pred[source_type].map(source_index2name)
    edge_pred[target_type] = edge_pred[target_type].map(target_index2name)



    edge_pred_dict[edge_type] = edge_pred

source_node_list = []
target_node_list = []
for edge_type in edge_pred_dict.keys():
    source_type, target_type = edge_type[0], edge_type[-1]

    source_node = edge_pred_dict[edge_type][source_type]
    source_node_list.append(source_node)

    target_node = edge_pred_dict[edge_type][target_type]
    target_node_list.append(target_node)

source_node = pd.concat(source_node_list)
target_node = pd.concat(target_node_list)

source_node = source_node.dropna().drop_duplicates().values.tolist()
target_node = target_node.dropna().drop_duplicates().values.tolist()

df = pd.DataFrame(index=source_node, columns=target_node)
df = df.fillna(0)

for edge_type in edge_pred_dict.keys():
    source_type, target_type = edge_type[0], edge_type[-1]

    edge_pred = edge_pred_dict[edge_type]
    for index, row in edge_pred.iterrows():
        source = row[source_type]
        target = row[target_type]
        weight = row['score']
        df.at[source, target] = weight

df.to_excel('edge_pred_tcmsp.xlsx',index=True)
