import torch
import zipfile
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import os
import pandas as pd
import matplotlib.pyplot as plt


class HeterogeneousGraphData():
    def __init__(self):
        super().__init__()
        self.node_feature_dict = None
        self.message_edge_dict = None
        self.predict_edge_dict = None
        self.unseen_node_type = None
        self.index2node_dict = None
        self.node2index_dict = None

    def to(self, device):
        for k, v in self.node_feature_dict.items():
            self.node_feature_dict[k] = v.to(device)
        for k, v in self.message_edge_dict.items():
            self.message_edge_dict[k] = v.to(device)
        for k, v in self.predict_edge_dict.items():
            self.predict_edge_dict[k] = v.to(device)

class TCMDataset():
    def __init__(self, dataset: str = 'HeNetRW'):
        super().__init__()
        self.metapath_dict = dict()
        self.dataset = dataset
        assert self.dataset in ['HeNetRW', 'HIT', 'TCMSP'], f'self.dataset not in [HeNetRW, HIT, TCMSP]'

        dir = f'./TCMDataset/{dataset}/'
        save_path = os.path.join(dir, f'{dataset}.pth')
        if os.path.exists(save_path):
            self.data_list = self.load(file_path=save_path)
        else:
            self.data_list = self.read(dir=dir)
            self.save(data_list=self.data_list, file_path=save_path)

        print('Done')

    def zip(self, dir):
        for zip in os.listdir(dir):
            if zip.endswith(".zip"):
                zipfile_path = os.path.join(dir, zip)
                with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
                    zip_ref.extractall(dir)

    def read(self, dir):
        self.zip(dir=dir)

        data_list = []
        for fold in os.listdir(dir):
            if fold.endswith(".zip"): continue

            if 'herb' in fold:
                unseen_node_type = 'herb'
            if 'target' in fold:
                unseen_node_type = 'target'

            fold_path = os.path.join(dir, fold)

            train_val_test = []
            for tvt in ['train', 'val', 'test']:
                node_feature_dict = {}
                message_edge_dict = {}
                predict_edge_dict = {}

                # node feature
                tvt_path = os.path.join(dir, fold, tvt)
                for file in os.listdir(tvt_path):
                    if not file.lower().endswith(('.csv', '.xlsx', '.xls')): continue
                    if 'feature' in file:
                        file_path = os.path.join(dir, fold, tvt, file)
                        node_feature = self.excel(file_name=file, file_path=file_path)
                        node_type = node_feature.columns[0]
                        node_feature_dict[node_type] = node_feature
                    if 'message' in file:
                        file_path = os.path.join(dir, fold, tvt, file)
                        message_edge = self.excel(file_name=file, file_path=file_path)
                        edge_type = tuple(message_edge.columns)
                        assert len(edge_type) == 2
                        message_edge_dict[edge_type] = message_edge
                    if 'predict' in file:
                        file_path = os.path.join(dir, fold, tvt, file)
                        predict_edge = self.excel(file_name=file, file_path=file_path)
                        edge_type = tuple(predict_edge.columns)
                        assert len(edge_type) == 2
                        predict_edge_dict[edge_type] = predict_edge

                # node index
                node2index_dict = dict()
                index2node_dict = dict()
                for node_type, node_feature in node_feature_dict.items():
                    index2node_dict[node_type] = node_feature[node_type].to_dict()
                    node2index_dict[node_type] = pd.Series(data=node_feature.index,
                                                           index=node_feature[node_type].values).to_dict()
                # node feature
                for node_type, node_feature in node_feature_dict.items():
                    node_feature_dict[node_type] = node_feature[node_feature.columns[1:]].values.astype(float)

                # message edge
                for edge_type, message_edge in message_edge_dict.items():
                    message_edge_dict = self.get_edge(edge_dict=message_edge_dict,
                                                      edge_dataframe=message_edge,
                                                      node2index_dict=node2index_dict)
                message_edge_dict = self.add_reverse(edge_dict=message_edge_dict)

                # predict edge
                for edge_type, predict_edge in predict_edge_dict.items():
                    predict_edge_dict = self.get_edge(edge_dict=predict_edge_dict,
                                                      edge_dataframe=predict_edge,
                                                      node2index_dict=node2index_dict)
                predict_edge_dict = self.add_reverse(edge_dict=predict_edge_dict)

                # hetero graph
                hetero_graph = HeterogeneousGraphData()

                node_feature_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in node_feature_dict.items()}
                message_edge_dict = {k: torch.tensor(v, dtype=torch.int64) for k, v in message_edge_dict.items()}
                predict_edge_dict = {k: torch.tensor(v, dtype=torch.int64) for k, v in predict_edge_dict.items()}

                hetero_graph.node_feature_dict = node_feature_dict
                hetero_graph.message_edge_dict = message_edge_dict
                hetero_graph.predict_edge_dict = predict_edge_dict
                hetero_graph.unseen_node_type = unseen_node_type
                hetero_graph.node2index_dict = node2index_dict
                hetero_graph.index2node_dict = index2node_dict
                train_val_test.append(hetero_graph)

            data_list.append(train_val_test)

        print('data.py Done!')
        return data_list

    def excel(self, file_name, file_path):
        print(f'Reading... {file_path}')
        if file_name.endswith('.csv'):
            try:
                df = pd.read_csv(file_path, na_filter=False)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
        elif file_name.endswith(('.xls', '.xlsx')):
            try:
                df = pd.read_excel(file_path, na_filter=False)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

        df = df.replace('', np.nan)
        print('Reading Successful')
        return df

    def add_reverse(self, edge_dict):
        out_dict = dict()
        for edge_type, edge_index in edge_dict.items():
            source_type = edge_type[0]
            target_type = edge_type[1]
            reverse_edge_type = (target_type, source_type)

            if edge_type in edge_dict.keys() and reverse_edge_type in edge_dict.keys():
                continue

            source_index = edge_index[0]
            target_index = edge_index[1]
            reverse_edge_index = [target_index, source_index]
            reverse_edge_index = np.array(reverse_edge_index)

            out_dict[edge_type] = edge_index
            out_dict[reverse_edge_type] = reverse_edge_index

        return out_dict

    def get_edge(self, edge_dict, edge_dataframe, node2index_dict):
        edge_type = tuple(edge_dataframe.columns)
        edge = edge_dataframe.copy()

        for node_type in edge.columns:
            edge[node_type] = edge[node_type].map(node2index_dict[node_type])
            is_all_int = np.issubdtype(edge[node_type].values.dtype, np.integer)
            assert is_all_int

        edge_dict[edge_type] = edge.values.astype(int).T

        return edge_dict

    def save(self, data_list, file_path):
        torch.save(data_list, file_path)

    def load(self, file_path):
        data_list = torch.load(file_path)

        return data_list

    def check(self):
        def graph_is_connected(edge_dict):
            graph = nx.Graph()
            for edge_type, edge in edge_dict.items():
                edge = edge.values.tolist()
                graph.add_edges_from(edge)
            is_connected = nx.is_connected(graph)
            assert is_connected

        def get_node_dict(edge_dict):
            node_list_dict = {}
            edge_list = list(edge_dict.values())
            graph = pd.concat(edge_list, axis=0)
            graph = graph.replace('', np.nan)
            for node_type in graph.columns:
                node_series = graph[node_type]
                if len(node_series) == 0: continue
                node_list = node_series.dropna().drop_duplicates().reset_index(drop=True).tolist()
                node_list_dict[node_type] = node_list

            return node_list_dict

        def check_node_num_error(edge_dict, feature_dict):
            all_node_dict = get_node_dict(edge_dict=edge_dict)
            for node_type, node_list in all_node_dict.items():
                feature_node_list = feature_dict[node_type][node_type].values.tolist()
                for node in node_list:
                    assert node in feature_node_list

        def degree_statistic(message_dict, predict_dict, unseen_node_type):
            message_node_dict = get_node_dict(message_dict)
            predict_node_dict = get_node_dict(predict_dict)

            graph = nx.Graph()
            for edge_type, edge in message_dict.items():
                edge = edge.values.tolist()
                graph.add_edges_from(edge)
            for edge_type, edge in predict_dict.items():
                edge = edge.values.tolist()
                graph.add_edges_from(edge)

            node_degree = []
            for node, degree in graph.degree():
                seen_type = None

                for k in message_node_dict.keys():
                    if node in message_node_dict[k]:
                        seen_type = 'seen'
                        node_type = k
                    if seen_type is not None: break

                for k in predict_node_dict.keys():
                    if node in predict_node_dict[k]:
                        node_type = k
                    if node in predict_node_dict[unseen_node_type]:
                        seen_type = 'unseen'
                    if seen_type is not None: break

                row = [node, node_type, seen_type, degree]
                node_degree.append(row)

            node_degree = pd.DataFrame(node_degree, columns=['node', 'node_type', 'seen_type', 'degree'])
            return node_degree

        def draw_degree_statistic(node_degree, save_dir):
            node_degree = node_degree.sort_values('degree', ascending=False).reset_index(drop=True)
            save_path = os.path.join(save_dir, 'Node Degree Distribution.xlsx')
            node_degree.to_excel(save_path, index=False)

            all_node_type = ['herb', 'target', 'compound', 'molecule', 'disease']
            all_color_map = {
                'herb': '#558B2F',  # 柔和的绿色
                'target': '#6495ED',  # 柔和的蓝色
                'compound': 'yellow',  # 柔和的淡黄色
                'molecule': 'yellow',  # 柔和的淡黄色
                'disease': '#A0522D'  # 柔和的棕色
            }

            node_type_list = []
            color_map = {}
            for node_type in all_node_type:
                if node_type in node_degree['node_type'].drop_duplicates().values.tolist():
                    node_type_list.append(node_type)
                    color_map[node_type] = all_color_map[node_type]
            color_map['unseen'] = 'red'

            node_degree_list = []
            for node_type in node_type_list:
                temp_node_degree = node_degree[node_degree['node_type'] == node_type]
                node_degree_list.append(temp_node_degree)

            node_degree = pd.concat(node_degree_list, axis=0)
            node_degree = node_degree.reset_index(drop=True)

            # 1
            plt.figure(figsize=(16, 9))

            if 'HeNetRW' in dir: bias = 10
            if 'HIT' in dir: bias = 50
            if 'TCMSP' in dir: bias = 100

            for i in node_degree.index:
                x = i
                y = node_degree['degree'][i]
                c = color_map[node_degree['node_type'][i]]
                plt.scatter(x, y, color=c, s=20)  # 将width减少一些，避免柱状图过于拥挤

            for i in node_degree.index:
                if node_degree['seen_type'][i] == 'unseen':
                    x = i
                    y = node_degree['degree'][i] + bias  # 调整y坐标，避免与柱状图重叠
                    plt.plot(x,
                             y,
                             marker='v',
                             markersize=8,
                             color=color_map['unseen'],
                             markeredgecolor='black',
                             markeredgewidth=0.5)

            # 控制x轴刻度密度，只显示部分刻度
            num_xticks = 10  # 设置要显示的x轴刻度数量
            xticks = [i for i in range(len(node_degree)) if i % (len(node_degree) // num_xticks) == 0]
            plt.xticks(xticks)

            # 控制y轴刻度密度，设置刻度间隔
            plt.yticks(range(0, node_degree['degree'].max() + 100, 100))  # 根据你的数据调整间隔

            handles = []
            labels = []
            for label_type, color in color_map.items():
                handles.append(plt.Rectangle((0, 0), 1, 1, fc=color))
                labels.append(label_type)
            plt.legend(handles, labels, loc='upper right')

            plt.ylabel('Node Degree')
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()
            save_path = os.path.join(save_dir, 'Node Degree Distribution 1.png')
            plt.savefig(save_path)

            # 2
            # 统计每个度值的节点数量
            degree_counts = node_degree['degree'].value_counts().sort_index()
            # 绘制散点图
            plt.figure(figsize=(16, 9))
            plt.scatter(degree_counts.index, degree_counts.values, s=20, color='#4682B4')  # 设置点大小为中等，颜色为牛仔裤蓝色
            plt.xlabel('Node Degree')
            plt.ylabel('Number of Nodes')

            plt.grid(axis='y', linestyle='--')  # 添加网格线，方便查看
            plt.tight_layout()

            save_path = os.path.join(save_dir, 'Node Degree Distribution 2.png')
            plt.savefig(save_path)

        dir = f'./TCMDataset/{self.dataset}/'
        draw_flag = True
        for fold in os.listdir(dir):
            fold_path = os.path.join(dir, fold)
            if not os.path.isdir(fold_path): continue

            if 'herb' in fold:
                unseen_node_type = 'herb'
            if 'target' in fold:
                unseen_node_type = 'target'

            for tvt in os.listdir(fold_path):
                tvt_path = os.path.join(dir, fold, tvt)
                if not os.path.isdir(tvt_path): continue

                tvt_path = os.path.join(dir, fold, tvt)
                feature_dict = {}
                message_dict = {}
                predict_dict = {}
                for file in os.listdir(tvt_path):
                    if not file.lower().endswith(('.csv', '.xlsx', '.xls')): continue

                    file_path = os.path.join(dir, fold, tvt, file)
                    print('read file path:', file_path)

                    if 'feature' in file:
                        feature = self.excel(file_name=file, file_path=file_path)
                        assert len(feature)
                        print('read ok')
                        node_type = feature.columns[0]
                        feature_dict[node_type] = feature

                    if 'message' in file:
                        message = self.excel(file_name=file, file_path=file_path)
                        assert len(message)
                        print('read ok')
                        edge_type = tuple(message.columns)
                        assert len(edge_type) == 2
                        message_dict[edge_type] = message

                    if 'predict' in file:
                        predict = self.excel(file_name=file, file_path=file_path)
                        assert len(predict)
                        print('read ok')
                        edge_type = tuple(predict.columns)
                        assert len(edge_type) == 2
                        predict_dict[edge_type] = predict

                # 检查节点
                check_node_num_error(edge_dict=message_dict, feature_dict=feature_dict)
                check_node_num_error(edge_dict=predict_dict, feature_dict=feature_dict)

                # 检查message是否是连通图
                graph_is_connected(edge_dict=message_dict)

                node_degree = degree_statistic(message_dict=message_dict,
                                               predict_dict=predict_dict,
                                               unseen_node_type=unseen_node_type)

                node_degree = node_degree.sort_values('degree', ascending=False).reset_index(drop=True)
                save_path = os.path.join(tvt_path, 'Node Degree Distribution.xlsx')
                node_degree.to_excel(save_path, index=False)

                if draw_flag:
                    draw_degree_statistic(node_degree=node_degree, save_dir=tvt_path)
                    draw_flag = False

            # for i in ['HeNetRW_isolated', 'HIT_isolated', 'HIT_nonisolated', 'TCMSP_isolated', 'TCMSP_nonisolated']:

    def get_metapath(self):
        if 'HeNetRW' in self.dataset:
            self.metapath_dict['herb'] = [[('herb', 'target'), ('target', 'herb')]]
            self.metapath_dict['target'] = [[('target', 'herb'), ('herb', 'target')]]

        if 'HIT' in self.dataset:
            self.metapath_dict['herb'] = [[('herb', 'target'), ('target', 'herb')],
                                          [('herb', 'compound'), ('compound', 'herb')]]
            self.metapath_dict['target'] = [[('target', 'herb'), ('herb', 'target')],
                                            [('target', 'compound'), ('compound', 'target')]]

        if 'TCMSP' in self.dataset:
            self.metapath_dict['herb'] = [[('herb', 'target'), ('target', 'herb')],
                                          [('herb', 'molecule'), ('molecule', 'herb')]]
            self.metapath_dict['target'] = [[('target', 'herb'), ('herb', 'target')],
                                            [('target', 'molecule'), ('molecule', 'target')],
                                            [('target', 'disease'), ('disease', 'target')]]

        return self.metapath_dict

# device = torch.device("cuda")
# for i in ['HeNetRW', 'HIT', 'TCMSP']:
#     tcm = TCMDataset(dataset=i)
#     print()
