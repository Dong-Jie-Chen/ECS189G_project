'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from codes.base_class.dataset import dataset
import pickle
import numpy as np

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
        X_train = [[d['image']] for d in data['train']]
        y_train = [d['label'] for d in data['train']]
        X_test = [[d['image']] for d in data['test']]
        y_test = [d['label'] for d in data['test']]
        return X_train, X_test, y_train, y_test

    def create_mini_batches(X, y, batch_size):
        X, y = np.array(X), np.array(y)
        index = np.arange(X.shape[0])
        np.random.shuffle(index)
        n_minibatches = X.shape[0] // batch_size + 1
        res_flag = (X.shape[0] % batch_size) > 0
        i = 0
        mini_batches = []
        for i in range(n_minibatches):
            X_mini = X[index[i * batch_size:(i + 1) * batch_size]]
            Y_mini = y[index[i * batch_size:(i + 1) * batch_size]]
            mini_batches.append((X_mini, Y_mini))
        if res_flag:
            n_minibatches += 1
            X_mini = X[index[i * batch_size:index.shape[0]]]
            Y_mini = y[index[i * batch_size:index.shape[0]]]
            mini_batches.append((X_mini, Y_mini))

        print(n_minibatches)
        return mini_batches