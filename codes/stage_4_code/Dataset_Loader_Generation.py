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
from collections import Counter
import pandas as pd
import re

class Dataset_Loader_Generation(dataset, torch.utils.data.Dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load(self):
        print('loading data...')
        train_dir = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        train_df = pd.read_csv(train_dir)
        train_df['Joke'] = train_df['Joke'].astype(str)

        text = train_df['Joke']
        text = [self.text_clean(i) + ['<EOS>'] for i in text]
        maxlength = max(len(x) for x in text)
        #for i in range(len(text)):
        #    text[i] = text[i] + ['<EOS>'] * (maxlength - len(text[i]))
        text_all = []
        for i in range(len(text)):
            text_all += text[i]
        return text_all


    def text_clean(self, text):
        clean = re.compile('<.*>')
        text = re.sub(clean, '', text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"www\S+", "", text)
        text = re.sub('[''"",,,]', '', text)
        text = re.sub('\n', '', text)
        #text = text.lower()
        text = text.split(' ')
        return text

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.sequence_length - 1

    def __getitem__(self, index):
        #n_batch = index // self.batch_size
        #total_size = len(self.words_indexes) - self.sequence_length - 1
        #index = (index % self.batch_size) * (total_size // self.batch_size) + n_batch
        return (
            torch.tensor(self.words_indexes[index:index+self.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.sequence_length+1]),
        )
