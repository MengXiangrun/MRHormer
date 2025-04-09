import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import ndcg_score
import os
import random


def generate_negative_edges(num_negative: int,
                            positive_edge: torch.Tensor,
                            source_node: torch.Tensor,
                            target_node: torch.Tensor,
                            is_check=False,
                            is_undirected=True):
    device = positive_edge.device

    if positive_edge.shape[0] == 2:
        positive_edge = positive_edge.T

    assert positive_edge.shape[1] == 2

    positive_edge = positive_edge.detach().cpu().numpy()
    source_node = source_node.detach().cpu().numpy()
    source_node = np.unique(ar=source_node)
    target_node = target_node.detach().cpu().numpy()
    target_node = np.unique(ar=target_node)

    # 随机生成负边
    num_sample = int(2 * num_negative)

    source_node = np.random.choice(source_node, size=num_sample, replace=True)
    target_node = np.random.choice(target_node, size=num_sample, replace=True)

    negative_edge = np.column_stack((source_node, target_node))
    assert negative_edge.shape[1] == 2

    # remove positive edge
    negative_df = pd.DataFrame(negative_edge, columns=['source', 'target'])
    positive_df = pd.DataFrame(positive_edge, columns=['source', 'target'])
    mask = negative_df.merge(positive_df, on=['source', 'target'], how='left', indicator=True)['_merge'] == 'both'
    negative_edge = negative_df[~mask].values

    # negative_edge = negative_edge[mask, :]
    negative_edge = np.unique(negative_edge, axis=0)

    if is_undirected:
        num_negative = num_negative // 2
        negative_edge = negative_edge[:num_negative, :]
        reverse_negative_edge = negative_edge[:, [1, 0]]
        negative_edge = np.concatenate([negative_edge, reverse_negative_edge], axis=0)

    if not is_undirected:
        negative_edge = negative_edge[:num_negative, :]

    if is_check:
        pos = positive_edge.tolist()
        neg = negative_edge.tolist()
        for n in neg:
            if n in pos:
                print(n)
                assert False

    negative_edge = negative_edge.T
    negative_edge = torch.from_numpy(negative_edge)
    negative_edge = negative_edge.to(torch.int64).to(device)
    assert negative_edge.shape[0] == 2

    return negative_edge


def performance(predict_dict, label_dict, BCELoss):
    predict = list()
    label = list()
    for edge_type in predict_dict.keys():
        predict.append(predict_dict[edge_type])
        label.append(label_dict[edge_type])

    predict = torch.cat(predict, dim=0)
    label = torch.cat(label, dim=0)

    # predict 是模型的输出概率，label 是真实标签
    loss = BCELoss(input=predict, target=label)
    predict = predict.detach().cpu().numpy()
    label = label.detach().cpu().numpy()

    aucroc = roc_auc_score(y_true=label, y_score=predict, average='macro')
    auprc = average_precision_score(y_true=label, y_score=predict)

    binary_predict = (predict > 0.5).astype(int)
    # accuracy = accuracy_score(y_true=label, y_pred=binary_predict)
    # recall = recall_score(y_true=label, y_pred=binary_predict)
    # precision = precision_score(y_true=label, y_pred=binary_predict)
    f1 = f1_score(y_true=label, y_pred=binary_predict)

    def NDCG(predict, label, k):
        sorted_predict = pd.Series(predict)
        sorted_label = pd.Series(label)

        sorted_predict = sorted_predict.sort_values(ascending=False)

        top_k_index = sorted_predict.index[:k]
        top_k_label = sorted_label[top_k_index].values

        hit_rate = np.sum(top_k_label) / k

        # top_k_predict = sorted_predict.values[:k]
        # ndcg = ndcg_score(y_true=[top_k_label], y_score=[top_k_predict], k=k)
        ndcg = ndcg_score(y_true=[label], y_score=[predict], k=k)

        return hit_rate, ndcg

    # k = 500
    # hr, ndcg = NDCG(predict=predict, label=label, k=k)

    return loss, aucroc, auprc, 0, 0, 0, f1, 0, 0, 0


def train(model, optimizer, BCELoss, device, dataset_name,
          node_feature_dict, message_edge_dict, positive_predict_edge_dict, unseen_node_type):
    model.train()
    optimizer.zero_grad()

    # encode
    node_emb_dict = model.encoder.forward(node_feature_dict,
                                          message_edge_dict)

    # sample negative edges randomly
    pos_neg_edges_dict = dict()
    pos_neg_label_dict = dict()
    for edge_type in positive_predict_edge_dict.keys():
        source_type, target_type = edge_type[0], edge_type[-1]

        pos_message = message_edge_dict[edge_type]
        pos_predict = positive_predict_edge_dict[edge_type]
        num_negative = pos_predict.shape[1]

        pos = torch.cat([pos_message, pos_predict], dim=1)

        if source_type == unseen_node_type:
            source_node = pos_predict[0]
            target_node = pos[1]
        if target_type == unseen_node_type:
            source_node = pos[0]
            target_node = pos_predict[1]

        neg_predict = generate_negative_edges(num_negative=num_negative,
                                              positive_edge=pos,
                                              source_node=source_node,
                                              target_node=target_node,
                                              is_undirected=False)
        pos_neg_edges = torch.cat([pos_predict, neg_predict], dim=1)
        pos_neg_edges_dict[edge_type] = pos_neg_edges

        # label
        num_pos_predict = pos_predict.shape[1]
        pos_label = torch.ones(num_pos_predict, dtype=torch.float32, device=device, requires_grad=False)
        num_neg_predict = neg_predict.shape[1]
        neg_label = torch.zeros(num_neg_predict, dtype=torch.float32, device=device, requires_grad=False)
        pos_neg_sup_label = torch.cat([pos_label, neg_label], dim=0)
        pos_neg_label_dict[edge_type] = pos_neg_sup_label

    # decode
    pred_dict = model.decoder.forward(node_emb_dict=node_emb_dict, edge_index_dict=pos_neg_edges_dict)

    loss, aucroc, auprc, accuracy, recall, precision, f1, k, hr, ndcg = performance(
        label_dict=pos_neg_label_dict,
        predict_dict=pred_dict,
        BCELoss=BCELoss)

    return model, optimizer, loss, aucroc, auprc, accuracy, recall, precision, f1, k, hr, ndcg


def test(model, optimizer, BCELoss, device, dataset_name,
         node_feature_dict, message_edge_dict, positive_predict_edge_dict, unseen_node_type):
    with torch.no_grad():
        model.eval()
        optimizer.zero_grad()

        # encode
        node_emb_dict = model.encoder.forward(node_feature_dict,
                                              message_edge_dict)

        # sample negative edges randomly
        pos_neg_edges_dict = dict()
        pos_neg_label_dict = dict()
        for edge_type in positive_predict_edge_dict.keys():
            source_type, target_type = edge_type[0], edge_type[-1]

            pos_message = message_edge_dict[edge_type]
            pos_predict = positive_predict_edge_dict[edge_type]
            pos = torch.cat([pos_message, pos_predict], dim=1)

            num_negative = pos_predict.shape[1]

            if source_type == unseen_node_type:
                source_node = pos_predict[0]
                target_node = pos[1]
            if target_type == unseen_node_type:
                source_node = pos[0]
                target_node = pos_predict[1]

            neg_predict = generate_negative_edges(num_negative=num_negative,
                                                  positive_edge=pos,
                                                  source_node=source_node,
                                                  target_node=target_node,
                                                  is_undirected=False)
            pos_neg_edges = torch.cat([pos_predict, neg_predict], dim=1)
            pos_neg_edges_dict[edge_type] = pos_neg_edges

            # label
            num_pos_predict = pos_predict.shape[1]
            pos_label = torch.ones(num_pos_predict, dtype=torch.float32, device=device, requires_grad=False)
            num_neg_predict = neg_predict.shape[1]
            neg_label = torch.zeros(num_neg_predict, dtype=torch.float32, device=device, requires_grad=False)
            pos_neg_sup_label = torch.cat([pos_label, neg_label], dim=0)
            pos_neg_label_dict[edge_type] = pos_neg_sup_label

        # decode
        pred_dict = model.decoder.forward(node_emb_dict=node_emb_dict, edge_index_dict=pos_neg_edges_dict)

        loss, aucroc, auprc, accuracy, recall, precision, f1, k, hr, ndcg = performance(
            label_dict=pos_neg_label_dict,
            predict_dict=pred_dict,
            BCELoss=BCELoss)

    return model, optimizer, loss, aucroc, auprc, accuracy, recall, precision, f1, k, hr, ndcg


class EarlyStopping0():
    def __init__(self, patience, threshold):
        self.val_loss_list = list()
        self.best_model_parameter = 0
        self.patience = patience
        self.threshold = threshold
        self.count = 0
        self.stop = False

    def save(self, model, val_loss):
        self.val_loss_list.append(val_loss.item())
        min_val_loss = min(self.val_loss_list)
        min_val_loss_epoch = self.val_loss_list.index(min_val_loss)
        now_val_loss = self.val_loss_list[-1]
        now_val_loss_epoch = self.val_loss_list.index(now_val_loss)

        if now_val_loss <= min_val_loss:
            self.best_model_parameter = model.state_dict()

        if (now_val_loss_epoch - min_val_loss_epoch) >= self.patience:
            self.stop = True

        if len(self.val_loss_list) > 2:
            now_val_loss = self.val_loss_list[-1]
            pre_val_loss = self.val_loss_list[-2]
            difference = pre_val_loss - now_val_loss
            difference = abs(difference)
            if difference < self.threshold:
                self.count += 1
                print(self.count)
            if self.count >= self.patience:
                self.stop = True


class EarlyStopping():
    def __init__(self, patience, threshold):
        self.val_loss_list = list()
        self.best_model_parameter = 0
        self.total_patience = patience
        self.patience = patience
        self.threshold = threshold
        self.count = 0
        self.stop = False
        self.min_val_loss = None
        self.save_path = None

    def record(self, model, now_val_loss):
        if self.min_val_loss is None:
            self.min_val_loss = now_val_loss

        if now_val_loss < self.min_val_loss:
            self.best_model_parameter = model.state_dict()
            self.patience = self.total_patience
            self.min_val_loss = now_val_loss

        if now_val_loss >= self.min_val_loss:
            self.patience = self.patience - 1

        if self.patience <= 0:
            self.stop = True

        print('patience:', self.patience)

    def save(self, dataset_name, unseen_node_type, model):
        self.save_path = dataset_name + '_' + unseen_node_type + '_' + type(model.encoder).__name__
        self.save_path = f'{self.save_path}.pth'
        torch.save(self.best_model_parameter, self.save_path)
        print('best_model_parameter saved')

    def load(self, model, path=None):
        if path is None:
            path = self.save_path
        self.best_model_parameter = torch.load(path)
        model.load_state_dict(self.best_model_parameter)

        return model


def save_excel(df, file_path, is_index):
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if '.xlsx' in file_path:
        df.to_excel(file_path, index=is_index)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
