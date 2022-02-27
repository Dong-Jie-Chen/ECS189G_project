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
import json
import string
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.legacy import data
import nltk
from nltk.corpus import stopwords


class Dataset_Loader_IMDB(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load(self):
        print('loading data...')
        nltk.download('stopwords')
        train_dir = os.path.join(self.dataset_source_folder_path, 'train')
        test_dir = os.path.join(self.dataset_source_folder_path, 'test')
        if os.path.exists(os.path.join(self.dataset_source_folder_path, 'train.json')):
            X_train, y_train = self.load_json('train')
            X_test, y_test = self.load_json('test')
        else:
            X_train, y_train = self.load_data(train_dir)
            X_test, y_test = self.load_data(test_dir)

        print("max length: ", len(max(X_train, key=lambda i: len(i))))

        self.save_json(X_train, y_train, 'train')
        self.save_json(X_test, y_test, 'test')

        self.TEXT = data.Field(tokenize='spacy',
                               tokenizer_language='en_core_web_sm',
                               include_lengths=True)

        self.LABEL = data.LabelField(dtype=torch.float)

        fields = {'text': ('text', self.TEXT), 'label': ('label', self.LABEL)}
        train_data, test_data = data.TabularDataset.splits(
            path=self.dataset_source_folder_path,
            train='train.json',
            test='test.json',
            format='json',
            fields=fields
        )

        MAX_VOCAB_SIZE = 25_000

        self.TEXT.build_vocab(train_data,
                              max_size=MAX_VOCAB_SIZE,
                              vectors="glove.6B.100d",
                              unk_init=torch.Tensor.normal_)

        self.LABEL.build_vocab(train_data)

        # No. of unique tokens in text
        print("Size of TEXT vocabulary:", len(self.TEXT.vocab))

        # No. of unique tokens in label
        print("Size of LABEL vocabulary:", len(self.LABEL.vocab))

        self.train_iterator, self.test_iterator = data.BucketIterator.splits(
            (train_data, test_data),
            batch_size=64,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            device=self.device)

        self.vocab_size = len(self.TEXT.vocab)

        return X_train, X_test, y_train, y_test

    def save_json(self, X, y, mode):

        json_lst = []
        for text, label in zip(X, y):
            if label == 0:
                json_lst.append({'text': text, 'label': 'neg'})
            elif label == 1:
                json_lst.append({'text': text, 'label': 'pos'})
        with open(os.path.join(self.dataset_source_folder_path, f'{mode}.json'), 'w') as outfile:
            for item in json_lst:
                outfile.write(json.dumps(item) + "\n")

    def load_json(self, mode):

        X, y = [], []
        with open(os.path.join(self.dataset_source_folder_path, f'{mode}.json')) as file:
            for line in file:
                dic = json.loads(line)
                X.append(dic['text'])
                if dic['label'] == 'neg':
                    y.append(0)
                elif dic['label'] == 'pos':
                    y.append(1)

        return X, y

    def text_clean(self, text):
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"www\S+", "", text)
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('[''"",,,]', '', text)
        text = text.lower()
        text = re.sub('\n', '', text)
        text = text.split(' ')
        return text

    # turn a doc into clean tokens
    def clean_doc(self, doc):
        # split into tokens by white space
        tokens = doc.split()
        # remove punctuation from each token
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        return tokens

    def load_data(self, dir):
        X, y = [], []
        for y_val, y_label in enumerate(['neg', 'pos']):
            y_dir = os.path.join(dir, y_label)
            for f_name in os.listdir(y_dir):
                with open(os.path.join(y_dir, f_name)) as f:
                    x_val = self.clean_doc(f.read())
                X.append(x_val)
                y.append(y_val)
        return X, y

    def create_mini_batches(self, method_n, batch_size, train=True):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if train:
            return self.train_iterator
        else:
            return self.test_iterator
