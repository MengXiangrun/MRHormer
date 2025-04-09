import random
import numpy as np
import pandas as pd
import torch
from data import TCMDataset, HeterogeneousGraphData
from function import train, test, EarlyStopping, save_excel,set_seed
from EdgeDecoder import EdgeDecoder
import datetime
from Config import Config
from MRHormer import MRHormer

class Model(torch.nn.Module):
    def __init__(self, Encoder, Decoder):
        super().__init__()
        self.encoder = Encoder
        self.decoder = Decoder


# device
device = torch.device("cuda:0")
set_seed()
worktype = ''  # HyPara,Ablation
# 'HeNetRW', 'TCMSP', 'HIT'
for dataset_name in ['HeNetRW']:
    # data
    dataset = TCMDataset(dataset=dataset_name)
    config = Config()

    if dataset_name in ['HeNetRW', 'HIT']:
        config.num_local_layer = 1  # best 1
        config.num_global_self_layer = 2  # best 2
        config.encoder_hidden_dim = 128
        config.optimizer_learning_rate = 0.0001
    if 'TCMSP' in dataset_name:
        config.num_local_layer = 2  # best 2
        config.num_global_self_layer = 4  # best 4
        config.encoder_hidden_dim = 16
        config.optimizer_learning_rate = 0.00005


    result_list = []
    for train_val_test in dataset.data_list:
        train_data, val_data, test_data = train_val_test
        config.node_type_list = list(test_data.node_feature_dict.keys())
        config.edge_type_list = list(test_data.message_edge_dict.keys())

        encoder = MRHormer(config=config)
        decoder = EdgeDecoder(config=config)
        model = Model(Encoder=encoder, Decoder=decoder).to(device)
        BCELoss = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=config.optimizer_learning_rate,
                                     weight_decay=config.optimizer_weight_decay)

        train_data.to(device)
        val_data.to(device)
        stop = EarlyStopping(patience=50, threshold=0.0005)
        with torch.no_grad():  # Initialize lazy modules.
            out = model.encoder.forward(train_data.node_feature_dict, train_data.message_edge_dict)
            del out

        # train
        for epoch in range(0, 10001):
            print()
            if stop.stop:
                break
            model.train()
            optimizer.zero_grad()

            model, optimizer, loss, aucroc, auprc, accuracy, recall, precision, f1, k, hr, ndcg = train(
                model=model,
                optimizer=optimizer,
                BCELoss=BCELoss,
                device=device,
                dataset_name=dataset_name,
                node_feature_dict=train_data.node_feature_dict,
                message_edge_dict=train_data.message_edge_dict,
                positive_predict_edge_dict=train_data.predict_edge_dict,
                unseen_node_type=train_data.unseen_node_type)

            print(f'epoch {epoch} train loss {loss.item():.4f} aucroc {aucroc:.4f} auprc {auprc:.4f} '
                  f'accuracy {accuracy:.4f} recall {recall:.4f} precision {precision:.4f} f1 {f1:.4f} '
                  f'hr@{k} {hr:.4f} ndcg@{k} {ndcg:.4f}')

            loss.backward()
            optimizer.step()

            # val
            model, optimizer, loss, aucroc, auprc, accuracy, recall, precision, f1, k, hr, ndcg = test(
                model=model,
                optimizer=optimizer,
                BCELoss=BCELoss,
                device=device,
                dataset_name=dataset_name,
                node_feature_dict=val_data.node_feature_dict,
                message_edge_dict=val_data.message_edge_dict,
                positive_predict_edge_dict=val_data.predict_edge_dict,
                unseen_node_type=val_data.unseen_node_type)

            print(f'epoch {epoch} val   loss {loss.item():.4f} aucroc {aucroc:.4f} auprc {auprc:.4f} '
                  f'accuracy {accuracy:.4f} recall {recall:.4f} precision {precision:.4f} f1 {f1:.4f} '
                  f'hr@{k} {hr:.4f} ndcg@{k} {ndcg:.4f}')

            # early_stopping
            if 'HeNetRW' in dataset_name:
                stop.record(model=model, now_val_loss=-aucroc)
            if 'HIT' in dataset_name:
                stop.record(model=model, now_val_loss=-aucroc)
            if 'TCMSP' in dataset_name:
                stop.record(model=model, now_val_loss=-aucroc)

        # save model
        stop.save(dataset_name=dataset_name, model=model)

        # test
        test_data.to(device)
        print('unseen node type:', test_data.unseen_node_type)
        # model.load_state_dict(stop.best_model_parameter)
        model = stop.load(model=model)
        model, optimizer, loss, aucroc, auprc, accuracy, recall, precision, f1, k, hr, ndcg = test(
            model=model,
            optimizer=optimizer,
            BCELoss=BCELoss,
            device=device,
            dataset_name=dataset_name,
            node_feature_dict=test_data.node_feature_dict,
            message_edge_dict=test_data.message_edge_dict,
            positive_predict_edge_dict=test_data.predict_edge_dict,
            unseen_node_type=test_data.unseen_node_type)

        print(f'test loss {loss.item():.4f} aucroc {aucroc:.4f} auprc {auprc:.4f} '
              f'accuracy {accuracy:.4f} recall {recall:.4f} precision {precision:.4f} f1 {f1:.4f} '
              f'hr@{k} {hr:.4f} ndcg@{k} {ndcg:.4f}')
        print('Done')
        print()

        result = [loss.item(), aucroc, auprc, accuracy, recall, precision, f1, hr, ndcg]
        result_list.append(result)

    header = ['loss', 'aucroc', 'auprc', 'accuracy', 'recall', 'precision', 'f1', f'hr@{k}', f'ndcg@{k}']
    result_dataframe = pd.DataFrame(result_list, columns=header)
    mean_result_dataframe = result_dataframe.mean(axis=0).to_frame().T
    mean_result_dataframe.index = ['mean']  # 设置索引为 'mean'
    result_dataframe = pd.concat([result_dataframe, mean_result_dataframe], axis=0)

    now = datetime.datetime.now()
    now_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    save_excel_path = f'./{dataset_name}_{worktype}/{type(model.encoder).__name__}_{now_time}.xlsx'
    print(save_excel_path)
    save_excel(df=result_dataframe, file_path=save_excel_path, is_index=True)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(result_dataframe)

    save_excel_path = f'./{dataset_name}_{worktype}/{type(model.encoder).__name__}_{now_time}_config.xlsx'
    config.save_to_excel(file_path=save_excel_path)

    print('Done')
