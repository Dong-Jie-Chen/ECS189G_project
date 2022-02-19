'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from codes.base_class.dataset import dataset
import pickle
import numpy as np
import torch
import os
import re
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.legacy import data

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load(self):
        print('loading data...')
        train_dir = os.path.join(self.dataset_source_folder_path, 'train')
        test_dir = os.path.join(self.dataset_source_folder_path, 'test')

        X_train, y_train = self.load_data(train_dir)
        X_test, y_test = self.load_data(test_dir)

        TEXT = data.Field(tokenize='spacy',
                          tokenizer_language='en_core_web_sm',
                          include_lengths=True)

        LABEL = data.LabelField(dtype=torch.float)

        self.tokenizer = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(self.yield_tokens(X_train), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        self.vocab_size = len(vocab)

        self.text_pipeline = lambda x: vocab(self.tokenizer(x))
        self.label_pipeline = lambda x: int(x)

        return X_train, X_test, y_train, y_test

    def yield_tokens(self, data_iter):
        for text in data_iter:
            yield self.tokenizer(text)

    def text_clean(self, text):
        text = text.lower()
        text = re.sub("\\s", " ", text)
        text = re.sub("[^a-zA-Z' ]", "", text)
        text = text.split(' ')
        return text

    def load_data(self, dir):
        X, y = [], []
        for y_val, y_label in enumerate(['neg', 'pos']):
            y_dir = os.path.join(dir, y_label)
            for f_name in os.listdir(y_dir):
                with open(os.path.join(y_dir, f_name)) as f:
                    x_val = f.read()
                X.append(x_val)
                y.append(y_val)
        return X, y

    def collate_batch(self, batch_X, batch_y):
        label_list, text_list, offsets = [], [], [0]
        for _label, _text in zip(batch_y, batch_X):
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return text_list.to(self.device), label_list.to(self.device), offsets.to(self.device)

    def create_mini_batches(self, method_n, X, y, batch_size, shuffle=True):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        index = np.arange(len(X))
        if shuffle:
          np.random.shuffle(index)
        index = list(index)
        n_minibatches = len(X) // batch_size
        res_flag = (len(X) % batch_size) > 0
        i = 0
        mini_batches = []
        for i in range(n_minibatches):
            X_mini = [X[idx] for idx in index[i * batch_size:(i + 1) * batch_size]]
            Y_mini = [y[idx]for idx in index[i * batch_size:(i + 1) * batch_size]]
            X_mini, Y_mini, offsets_mini = self.collate_batch(X_mini, Y_mini)
            #Y_mini = torch.unsqueeze(Y_mini, 1)
            mini_batches.append((X_mini, Y_mini, offsets_mini))
        if res_flag:
            i += 1
            X_mini = [X[idx] for idx in index[i * batch_size:(i + 1) * batch_size]]
            Y_mini = [y[idx] for idx in index[i * batch_size:(i + 1) * batch_size]]
            X_mini, Y_mini, offsets_mini = self.collate_batch(X_mini, Y_mini)
            #Y_mini = torch.unsqueeze(Y_mini, 1)
            mini_batches.append((X_mini, Y_mini, offsets_mini))

        return mini_batches

