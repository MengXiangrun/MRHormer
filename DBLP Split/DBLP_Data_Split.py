import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
import pandas as pd
import copy
import random
import networkx as nx
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import os
import itertools


def DBLP_to_excel():
    path = './DBLP'
    dataset = DBLP(path, transform=T.Constant(node_types='conference'))
    data = dataset[0]

    total_node_feature_dict = {}
    for node_type, node_feature in data.x_dict.items():
        filename = f'./DBLP/total_{node_type}_feature.xlsx'

        index_name = []
        for node_index in range(node_feature.shape[0]):
            index_name.append([f'{node_type}_{node_index}'])

        column_name = []
        for column_index in range(node_feature.shape[1]):
            column_name.append(f'feature_{column_index}')

        df1 = pd.DataFrame(index_name, columns=[node_type])
        df2 = pd.DataFrame(node_feature, columns=column_name)
        node_feature_df = pd.concat([df1, df2], axis=1)

        # node_feature_df.to_excel(filename, index=True)
        total_node_feature_dict[node_type] = node_feature_df

    total_graph_edge = []
    for edge_type, edge_index in data.edge_index_dict.items():
        source_type, _, target_type = edge_type
        column_name = [source_type, target_type]

        edge_index = edge_index.T.numpy().tolist()
        edge_df = []
        for edge in edge_index:
            edge[0] = f'{source_type}_{edge[0]}'
            edge[1] = f'{target_type}_{edge[1]}'
            edge_df.append(edge)

        edge_df = pd.DataFrame(edge_df, columns=column_name)
        total_graph_edge.append(edge_df)

    # total_graph_edge = pd.concat(edge_df_list, axis=0)
    # edge_df.to_excel('./DBLP/total_graph_edge', index=True)
    return total_node_feature_dict, total_graph_edge


def graph_is_connected(edge_list):
    hg = nx.DiGraph()

    for edge_df in edge_list:
        edge_list = edge_df.values.tolist()
        hg.add_edges_from(edge_list)

    is_connected = nx.is_strongly_connected(hg)

    return is_connected


def drop_node_in_graph(message_edge_list, drop_node_type, drop_node):
    new_message_edge_list = []
    predict_edge_list = []
    for message_edge_df in message_edge_list:
        if drop_node_type not in message_edge_df.columns:
            new_message_edge_list.append(message_edge_df)
            continue

        message_edge_df = message_edge_df.drop_duplicates().reset_index(drop=True)
        mask = message_edge_df[drop_node_type] == drop_node

        new_message_edge_df = message_edge_df.loc[~mask]
        new_message_edge_df = new_message_edge_df.drop_duplicates().reset_index(drop=True)
        new_message_edge_list.append(new_message_edge_df)

        predict_edge_df = message_edge_df.loc[mask]
        predict_edge_df = predict_edge_df.drop_duplicates().reset_index(drop=True)
        predict_edge_list.append(predict_edge_df)

    return message_edge_list, new_message_edge_list, predict_edge_list


def get_node(edge_list):
    node_dict = {}
    node_df = pd.concat(edge_list, axis=0)
    for node_type in node_df.columns:
        node_list = node_df[node_type].drop_duplicates().dropna().values.tolist()
        random.shuffle(node_list)
        node_dict[node_type] = node_list

    return node_dict


def save_df(dir_path,
            total_node_feature_dict,
            message_edge_list,
            predict_edge_list,
            data_type,
            unseen_node_type):
    folder = f'{unseen_node_type} as unseen node'
    folder_path = os.path.join(dir_path, folder, data_type)
    if os.path.exists(folder_path) is False:
        os.makedirs(folder_path, exist_ok=True)

    for edge_df in message_edge_list:
        source_type, target_type = edge_df.columns
        file_name = f'{data_type}_message_{source_type}_{target_type}.xlsx'
        file_path = os.path.join(folder_path, file_name)
        edge_df.to_excel(file_path, index=False)

    for edge_df in predict_edge_list:
        source_type, target_type = edge_df.columns
        file_name = f'{data_type}_predict_{source_type}_{target_type}.xlsx'
        file_path = os.path.join(folder_path, file_name)
        edge_df.to_excel(file_path, index=False)

    node_dict = get_node(edge_list=message_edge_list + predict_edge_list)

    for node_type, node_feature in total_node_feature_dict.items():
        node_list = node_dict[node_type]
        new_feature_df = node_feature.loc[node_feature[node_type].isin(node_list)]
        for node in node_list:
            assert node in new_feature_df[node_type].values.tolist()

        file_name = f'{data_type}_{node_type}_feature.xlsx'
        file_path = os.path.join(folder_path, file_name)
        new_feature_df.to_excel(file_path, index=False)


def num_inductive_split(num, rate: float = 0.2):
    num_test_sup = int(num * rate)
    num = num - num_test_sup

    num_val_sup = int(num * rate)
    num = num - num_val_sup

    num_train_sup = int(num * rate)
    num = num - num_train_sup

    num_train_msg = num

    num_unseen_node_dict = {}

    num_unseen_node_dict['train'] = num_train_sup
    num_unseen_node_dict['valid'] = num_val_sup
    num_unseen_node_dict['test'] = num_test_sup

    return num_unseen_node_dict


def concat_different_edge_df_in_list(edge_list):
    edge_dict = {}
    for edge_df in edge_list:
        edge_type = tuple(edge_df.columns)
        edge_dict.setdefault(edge_type, [])

        edge_dict[edge_type].append(edge_df)

    for edge_type, edge_df_list in edge_dict.items():
        edge_dict[edge_type] = pd.concat(edge_df_list, axis=0)

    edge_list = list(edge_dict.values())
    return edge_list


def check(unseen_node_type,
          train_message_list, train_predict_list,
          valid_message_list, valid_predict_list,
          test_message_list, test_predict_list):
    train_node_list = get_node(edge_list=train_message_list + train_predict_list)[unseen_node_type]
    valid_node_list = get_node(edge_list=valid_predict_list)[unseen_node_type]
    test_node_list = get_node(edge_list=test_predict_list)[unseen_node_type]

    for node in test_node_list:
        if node in train_node_list: assert 0, f'wrong, {node}'
        if node in valid_node_list: assert 0, f'wrong, {node}'

    for node in valid_node_list:
        if node in train_node_list: assert 0, f'wrong, {node}'


total_node_feature_dict, total_edge_list = DBLP_to_excel()

is_connected = graph_is_connected(total_edge_list)

# split
unseen_node_type_list = ['paper']
for unseen_node_type in unseen_node_type_list:
    total_node_dict = get_node(edge_list=total_edge_list)
    num_unseen_dict = num_inductive_split(num=len(total_node_dict[unseen_node_type]), rate=0.2)
    unavailable_node = []

    # test
    test_message_list = total_edge_list.copy()
    test_predict_list = []
    for drop_node in total_node_dict[unseen_node_type]:
        print(len(unavailable_node))
        if drop_node in unavailable_node: continue
        test_message_list, new_test_message, test_predict = drop_node_in_graph(message_edge_list=test_message_list,
                                                                               drop_node_type=unseen_node_type,
                                                                               drop_node=drop_node)
        if graph_is_connected(edge_list=new_test_message):
            test_predict_list += test_predict
            test_message_list = new_test_message.copy()
            unavailable_node.append(drop_node)

        if len(unavailable_node) >= num_unseen_dict['test']:
            print('test 数量足够')
            break

    # valid
    valid_message_list = test_message_list.copy()
    valid_predict_list = []
    for drop_node in total_node_dict[unseen_node_type]:
        print(len(unavailable_node))
        if drop_node in unavailable_node: continue
        valid_message_list, new_valid_message, valid_predict = drop_node_in_graph(message_edge_list=valid_message_list,
                                                                                  drop_node_type=unseen_node_type,
                                                                                  drop_node=drop_node)
        if graph_is_connected(edge_list=new_valid_message):
            valid_predict_list += valid_predict
            valid_message_list = new_valid_message.copy()
            unavailable_node.append(drop_node)

        if len(unavailable_node) >= num_unseen_dict['valid']+num_unseen_dict['test']:
            print('valid 数量足够')
            break

    # train
    train_message_list = valid_message_list.copy()
    train_predict_list = []
    for drop_node in total_node_dict[unseen_node_type]:
        print(len(unavailable_node))
        if drop_node in unavailable_node: continue
        train_message_list, new_train_message, train_predict = drop_node_in_graph(message_edge_list=train_message_list,
                                                                                  drop_node_type=unseen_node_type,
                                                                                  drop_node=drop_node)
        if graph_is_connected(edge_list=new_train_message):
            train_predict_list += train_predict
            train_message_list = new_train_message.copy()
            unavailable_node.append(drop_node)

        if len(unavailable_node) >= num_unseen_dict['train']+num_unseen_dict['valid']+num_unseen_dict['test']:
            print('train 数量足够')
            break

    check(unseen_node_type=unseen_node_type,
          train_message_list=train_message_list, train_predict_list=train_predict_list,
          valid_message_list=valid_message_list, valid_predict_list=valid_predict_list,
          test_message_list=test_message_list, test_predict_list=test_predict_list)

    # test
    test_message_list = concat_different_edge_df_in_list(edge_list=test_message_list)
    test_predict_list = concat_different_edge_df_in_list(edge_list=test_predict_list)
    save_df(dir_path='./DBLP',
            total_node_feature_dict=total_node_feature_dict,
            message_edge_list=test_message_list,
            predict_edge_list=test_predict_list,
            data_type='test',
            unseen_node_type=unseen_node_type)

    # valid
    valid_message_list = concat_different_edge_df_in_list(edge_list=valid_message_list)
    valid_predict_list = concat_different_edge_df_in_list(edge_list=valid_predict_list)
    save_df(dir_path='./DBLP',
            total_node_feature_dict=total_node_feature_dict,
            message_edge_list=valid_message_list,
            predict_edge_list=valid_predict_list,
            data_type='valid',
            unseen_node_type=unseen_node_type)

    # test
    train_message_list = concat_different_edge_df_in_list(edge_list=train_message_list)
    train_predict_list = concat_different_edge_df_in_list(edge_list=train_predict_list)
    save_df(dir_path='./DBLP',
            total_node_feature_dict=total_node_feature_dict,
            message_edge_list=train_message_list,
            predict_edge_list=train_predict_list,
            data_type='train',
            unseen_node_type=unseen_node_type)

print()
