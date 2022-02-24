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
        train_df['Joke'] = train_df['Joke'].astype(str) + ' <EOS>'
        text = train_df['Joke'].str.cat(sep=' ').split(' ')
        return text

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.sequence_length+1]),
        )
