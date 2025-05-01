'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv
import pickle
import os
import networkx as nx
from sklearn.preprocessing import StandardScaler


class DatasetLoader(dataset):
    c = 0.15
    k = 5
    data = None
    batch_size = None

    dataset_source_folder_path = None
    dataset_name = None

    load_all_tag = False
    compute_s = False
    use_rope = True  # 启用RoPE
    use_grqc_dataset = True  # 新增：控制是否使用ca-GrQc数据集

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(DatasetLoader, self).__init__(dName, dDescription)

    def load_hop_wl_batch(self):
        print('Load WL Dictionary')
        f = open('./result/WL/' + self.dataset_name, 'rb')
        wl_dict = pickle.load(f)
        f.close()

        print('Load Hop Distance Dictionary')
        f = open('./result/Hop/hop_' + self.dataset_name + '_' + str(self.k), 'rb')
        hop_dict = pickle.load(f)
        f.close()

        print('Load Subgraph Batches')
        f = open('./result/Batch/' + self.dataset_name + '_' + str(self.k), 'rb')
        batch_dict = pickle.load(f)
        f.close()

        # 加载最短路径距离用于RoPE
        shortest_paths = None
        if self.use_rope:
            try:
                print('Load Shortest Path Distances for RoPE')
                rope_path = './result/RoPE/shortest_paths_' + self.dataset_name
                if os.path.exists(rope_path):
                    f = open(rope_path, 'rb')
                    shortest_paths = pickle.load(f)
                    f.close()
                else:
                    print("Warning: RoPE data not found at", rope_path)
            except Exception as e:
                print("Error loading RoPE data:", e)
                shortest_paths = None

        return hop_dict, wl_dict, batch_dict, shortest_paths

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def preprocess_ca_grqc(self):
        """
        预处理ca-GrQc数据集，创建node和link文件
        """
        print("正在预处理ca-GrQc数据集，但仍将其作为'cora'使用...")

        # 边列表文件的路径
        edge_file = os.path.join(self.dataset_source_folder_path, 'ca-GrQc.txt')

        # 检查边列表文件是否存在
        if not os.path.exists(edge_file):
            raise FileNotFoundError(f"边列表文件不存在：{edge_file}。请从SNAP下载。")

        # 从边列表创建NetworkX图
        G = nx.Graph()

        # 读取边列表，跳过注释行
        with open(edge_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    src, dst = int(parts[0]), int(parts[1])
                    G.add_edge(src, dst)

        # 获取所有节点并重新标记为从0开始的连续整数
        all_nodes = sorted(G.nodes())
        node_map = {node: i for i, node in enumerate(all_nodes)}
        G = nx.relabel_nodes(G, node_map)

        # 创建节点特征：使用度、中心性和聚类系数
        degrees = dict(G.degree())
        betweenness = nx.betweenness_centrality(G, k=min(100, len(G.nodes)))  # 使用采样以加快计算
        clustering = nx.clustering(G)

        # 组合特征
        features = []
        for node in range(len(G.nodes())):
            node_features = [
                degrees.get(node, 0),
                betweenness.get(node, 0),
                clustering.get(node, 0)
            ]
            features.append(node_features)

        # 标准化特征
        features = np.array(features, dtype=np.float32)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # 扩展特征维度，匹配cora的1433维
        expanded_features = np.zeros((len(G.nodes()), 1433))

        # 将计算的特征放入前3列
        expanded_features[:, :features.shape[1]] = features

        # 使用度信息生成一些随机特征填充剩余列
        for i in range(len(G.nodes())):
            deg = degrees.get(i, 1)
            np.random.seed(i * 42)  # 确保每个节点的随机特征是确定性的
            expanded_features[i, features.shape[1]:] = np.random.normal(
                scale=0.01 * deg, size=1433 - features.shape[1]
            )

        # 使用社区检测为节点分配类别标签（7类，与cora相同）
        communities = list(nx.algorithms.community.greedy_modularity_communities(G))

        # 将每个节点映射到其社区
        node_to_community = {}
        for i, community in enumerate(communities):
            for node in community:
                # 限制为7个类别（与cora相同）
                community_id = min(i, 6)
                node_to_community[node] = community_id

        # 创建node文件（ID，特征，标签）
        node_file_path = os.path.join(self.dataset_source_folder_path, 'node')
        with open(node_file_path, 'w') as f:
            for node in range(len(G.nodes())):
                # 获取节点特征
                feature_str = ' '.join([str(val) for val in expanded_features[node]])
                # 获取或分配社区（类别）ID
                class_id = node_to_community.get(node, 0)
                # 写入文件：ID，特征，类别
                f.write(f"{node} {feature_str} {class_id}\n")

        # 创建link文件
        link_file_path = os.path.join(self.dataset_source_folder_path, 'link')
        with open(link_file_path, 'w') as f:
            for src, dst in G.edges():
                f.write(f"{src} {dst}\n")

        print(f"预处理完成。节点文件：{node_file_path}，链接文件：{link_file_path}")
        return True

    def load(self):
        """加载引用网络数据集"""
        print(f'加载{self.dataset_name}数据集...')

        # 如果启用了ca-GrQc数据集，但使用cora名称
        if self.use_grqc_dataset and self.dataset_name.lower() == 'cora':
            node_file = os.path.join(self.dataset_source_folder_path, 'node')
            if not os.path.exists(node_file):
                self.preprocess_ca_grqc()

        # 加载数据集
        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

        one_hot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # 构建图
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        index_id_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(one_hot_labels.shape[0], one_hot_labels.shape[0]),
                            dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        eigen_adj = None
        if self.compute_s:
            eigen_adj = self.c * inv((sp.eye(adj.shape[0]) - (1 - self.c) * self.adj_normalize(adj)).toarray())

        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # 为ca-GrQc设置不同的训练/验证/测试索引
        if self.use_grqc_dataset and self.dataset_name.lower() == 'cora':
            num_nodes = adj.shape[0]
            # 使用5%的节点用于训练，10%用于验证，其余用于测试
            train_size = int(0.05 * num_nodes)
            val_size = int(0.1 * num_nodes)

            # 生成随机索引
            np.random.seed(42)  # 为了可重复性
            all_indices = np.random.permutation(num_nodes)

            idx_train = all_indices[:train_size]
            idx_val = all_indices[train_size:train_size + val_size]
            idx_test = all_indices[train_size + val_size:]
        elif self.dataset_name == 'cora':
            idx_train = range(140)
            idx_test = range(200, 1200)
            idx_val = range(1200, 1500)
        elif self.dataset_name == 'citeseer':
            idx_train = range(120)
            idx_test = range(200, 1200)
            idx_val = range(1200, 1500)
        elif self.dataset_name == 'pubmed':
            idx_train = range(60)
            idx_test = range(6300, 7300)
            idx_val = range(6000, 6300)
        elif self.dataset_name == 'cora-small':
            idx_train = range(5)
            idx_val = range(5, 10)
            idx_test = range(5, 10)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(one_hot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        if self.load_all_tag:
            hop_dict, wl_dict, batch_dict, shortest_paths = self.load_hop_wl_batch()
            raw_feature_list = []
            role_ids_list = []
            position_ids_list = []
            hop_ids_list = []
            rope_distances_list = []  # RoPE距离列表

            for node in idx:
                node_index = idx_map[node]
                neighbors_list = batch_dict[node]

                raw_feature = [features[node_index].tolist()]
                role_ids = [wl_dict[node]]
                position_ids = range(len(neighbors_list) + 1)
                hop_ids = [0]
                rope_distances = [0]  # 到自身的距离为0

                for neighbor, intimacy_score in neighbors_list:
                    neighbor_index = idx_map[neighbor]
                    raw_feature.append(features[neighbor_index].tolist())
                    role_ids.append(wl_dict[neighbor])

                    if neighbor in hop_dict[node]:
                        hop_ids.append(hop_dict[node][neighbor])
                    else:
                        hop_ids.append(99)

                    # 添加最短路径距离用于RoPE
                    if shortest_paths and node in shortest_paths and neighbor in shortest_paths[node]:
                        rope_distances.append(shortest_paths[node][neighbor])
                    else:
                        rope_distances.append(99)  # 对不可达节点使用较大的值

                raw_feature_list.append(raw_feature)
                role_ids_list.append(role_ids)
                position_ids_list.append(position_ids)
                hop_ids_list.append(hop_ids)
                rope_distances_list.append(rope_distances)

            raw_embeddings = torch.FloatTensor(raw_feature_list)
            wl_embedding = torch.LongTensor(role_ids_list)
            hop_embeddings = torch.LongTensor(hop_ids_list)
            int_embeddings = torch.LongTensor(position_ids_list)
            rope_distances = torch.LongTensor(rope_distances_list) if shortest_paths else None
        else:
            raw_embeddings, wl_embedding, hop_embeddings, int_embeddings, rope_distances = None, None, None, None, None

        print(f"*** 数据集统计：节点数={features.shape[0]}，边数={edges.shape[0]}，特征维度={features.shape[1]} ***")

        return {
            'X': features,
            'A': adj,
            'S': eigen_adj,
            'index_id_map': index_id_map,
            'edges': edges_unordered,
            'raw_embeddings': raw_embeddings,
            'wl_embedding': wl_embedding,
            'hop_embeddings': hop_embeddings,
            'int_embeddings': int_embeddings,
            'rope_distances': rope_distances,  # RoPE距离字段
            'y': labels,
            'idx': idx,
            'idx_train': idx_train,
            'idx_test': idx_test,
            'idx_val': idx_val
        }