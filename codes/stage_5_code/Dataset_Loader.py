'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from codes.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = sorted(set(labels))
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))

        # load node data from file
        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)
        edges = torch.LongTensor(edges.transpose((1,0)))

        # the following part, you can either put them into the setting class or you can leave them in the dataset loader
        # the following train, test, val index are just examples, sample the train, test according to project requirements
        if self.dataset_name == 'cora':
            idx_train, idx_val, idx_test = self.balanced_split(list(labels.numpy()), train_size=140, test_size=1050)

        elif self.dataset_name == 'citeseer':
            idx_train, idx_val, idx_test = self.balanced_split(list(labels.numpy()), train_size=120, test_size=1200)

        elif self.dataset_name == 'pubmed':
            idx_train, idx_val, idx_test = self.balanced_split(list(labels.numpy()), train_size=60, test_size=600)

        #---- cora-small is a toy dataset I hand crafted for debugging purposes ---
        elif self.dataset_name == 'cora-small':
            idx_train = range(5)
            idx_val = range(5, 10)
            idx_test = range(5, 10)
        else:
            assert 0

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        # get the training nodes/testing nodes
        train_x = features[idx_train]
        val_x = features[idx_val]
        test_x = features[idx_test]
        print(train_x.shape, val_x.shape, test_x.shape)
        print('node:', len(idx_map), 'edge:', edges.shape, 'X:', features.shape, 'y:', labels.shape, 'adj:', adj.shape)

        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test_val': train_test_val}

    def balanced_split(self, y, train_size, test_size):

        def split_class(y, label, train_size, test_size):
            indices = np.flatnonzero(y == label)
            n_train = train_size // (max(y) + 1)
            n_test = test_size // (max(y) + 1)
            idx_train, idx_test = train_test_split(range(indices.shape[0]), train_size=n_train, random_state=42)
            idx_val, idx_test = train_test_split(idx_test, test_size=n_test, random_state=42)
            train_index = indices[idx_train]
            val_index = indices[idx_val]
            test_index = indices[idx_test]
            return (train_index, val_index, test_index)

        idx = [split_class(y, label, train_size, test_size) for label in np.unique(y)]
        train_index = np.concatenate([train for train, _, _ in idx])
        val_index = np.concatenate([val for _, val, _ in idx])
        test_index = np.concatenate([test for _, _, test in idx])
        return train_index, val_index, test_index



