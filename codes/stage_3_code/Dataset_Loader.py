'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from codes.base_class.dataset import dataset
import pickle
import numpy as np
import torch

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X = []
        y = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        print('training set size:', len(data['train']), 'testing set size:', len(data['test']))
        if self.dataset_source_file_name in ["MNIST"]:
            X_train = [[d['image']] for d in data['train']]
            y_train = [d['label'] for d in data['train']]
            X_test = [[d['image']] for d in data['test']]
            y_test = [d['label'] for d in data['test']]
        elif self.dataset_source_file_name in ["CIFAR"]:
            X_train = np.array([d['image'] for d in data['train']])
            y_train = np.array([d['label'] for d in data['train']])
            X_test = np.array([d['image'] for d in data['test']])
            y_test = np.array([d['label'] for d in data['test']])
            X_train = X_train / 255
            X_test = X_test / 255
            X_train = np.transpose(X_train, (0, 3, 1, 2))
            X_test = np.transpose(X_test, (0, 3, 1, 2))
        elif self.dataset_source_file_name == "ORL":
            X_train = np.array([d['image'] for d in data['train']])
            y_train = np.array([d['label'] for d in data['train']])
            X_test = np.array([d['image'] for d in data['test']])
            y_test = np.array([d['label'] for d in data['test']])
            X_train = np.transpose(X_train, (0, 3, 1, 2))
            X_test = np.transpose(X_test, (0, 3, 1, 2))
            y_train = y_train - 1
            y_test = y_test - 1
        return X_train, X_test, y_train, y_test

    def create_mini_batches(method_n, X, y, batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        index = np.arange(X.shape[0])
        np.random.shuffle(index)
        n_minibatches = X.shape[0] // batch_size + 1
        res_flag = (X.shape[0] % batch_size) > 0
        i = 0
        mini_batches = []
        for i in range(n_minibatches):
            X_mini = X[index[i * batch_size:(i + 1) * batch_size]]
            Y_mini = y[index[i * batch_size:(i + 1) * batch_size]]
            X_mini = torch.FloatTensor(np.array(X_mini))
            Y_mini = torch.LongTensor(np.array(Y_mini))
            mini_batches.append((X_mini, Y_mini))
        if res_flag:
            n_minibatches += 1
            X_mini = X[index[i * batch_size:index.shape[0]]]
            Y_mini = y[index[i * batch_size:index.shape[0]]]
            X_mini = torch.FloatTensor(np.array(X_mini))
            Y_mini = torch.LongTensor(np.array(Y_mini))
            mini_batches.append((X_mini, Y_mini))

        return mini_batches

