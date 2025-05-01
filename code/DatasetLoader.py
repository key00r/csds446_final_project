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


class DatasetLoader(dataset):
    c = 0.15
    k = 5
    data = None
    batch_size = None

    dataset_source_folder_path = None
    dataset_name = None

    load_all_tag = False
    compute_s = False
    use_rope = True  # RoPE标志
    use_multi_scale_rope = True  # 多尺度RoPE标志

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

        # 加载最短路径距离，用于RoPE或全局RoPE
        shortest_paths = None
        if self.use_rope or self.use_multi_scale_rope:
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

    def load(self):
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(self.dataset_name))

        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

        one_hot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # build graph
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

        if self.dataset_name == 'cora':
            idx_train = range(140)
            idx_test = range(200, 1200)
            idx_val = range(1200, 1500)
        elif self.dataset_name == 'citeseer':
            idx_train = range(120)
            idx_test = range(200, 1200)
            idx_val = range(1200, 1500)
            # features = self.normalize(features)
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

        raw_embeddings = None
        wl_embedding = None
        hop_embeddings = None
        int_embeddings = None
        global_distances = None

        if self.load_all_tag:
            hop_dict, wl_dict, batch_dict, shortest_paths = self.load_hop_wl_batch()
            raw_feature_list = []
            role_ids_list = []
            position_ids_list = []
            hop_ids_list = []  # 本地跳数距离（局部RoPE）
            global_distances_list = []  # 全局距离（全局RoPE）

            for node in idx:
                node_index = idx_map[node]
                neighbors_list = batch_dict[node]

                raw_feature = [features[node_index].tolist()]
                role_ids = [wl_dict[node]]
                position_ids = list(range(len(neighbors_list) + 1))
                hop_ids = [0]  # 自身到自身的距离为0
                global_distance = [0]  # 自身到自身的距离为0

                for neighbor, intimacy_score in neighbors_list:
                    neighbor_index = idx_map[neighbor]
                    raw_feature.append(features[neighbor_index].tolist())
                    role_ids.append(wl_dict[neighbor])

                    # 本地跳数距离（用于局部RoPE）
                    if neighbor in hop_dict[node]:
                        hop_ids.append(hop_dict[node][neighbor])
                    else:
                        hop_ids.append(99)

                    # 全局距离（用于全局RoPE）
                    if shortest_paths and node in shortest_paths and neighbor in shortest_paths[node]:
                        global_distance.append(shortest_paths[node][neighbor])
                    else:
                        global_distance.append(99)  # 对于不可达的节点使用较大的值

                raw_feature_list.append(raw_feature)
                role_ids_list.append(role_ids)
                position_ids_list.append(position_ids)
                hop_ids_list.append(hop_ids)
                global_distances_list.append(global_distance)

            raw_embeddings = torch.FloatTensor(raw_feature_list)
            wl_embedding = torch.LongTensor(role_ids_list)
            hop_embeddings = torch.LongTensor(hop_ids_list)  # 本地距离
            int_embeddings = torch.LongTensor(position_ids_list)
            global_distances = torch.LongTensor(global_distances_list)  # 全局距离

        return {
            'X': features,
            'A': adj,
            'S': eigen_adj,
            'index_id_map': index_id_map,
            'edges': edges_unordered,
            'raw_embeddings': raw_embeddings,
            'wl_embedding': wl_embedding,
            'hop_embeddings': hop_embeddings,  # 本地距离
            'int_embeddings': int_embeddings,
            'global_distances': global_distances,  # 全局距离
            'y': labels,
            'idx': idx,
            'idx_train': idx_train,
            'idx_test': idx_test,
            'idx_val': idx_val
        }