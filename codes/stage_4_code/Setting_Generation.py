'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from codes.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader

class Setting_Generation(setting):

    def print_setup_summary(self):
        print('dataset:', self.dataset.dataset_name, ', method:', self.method.method_name,
              ', setting:', self.setting_name, ', result:', self.result.result_name, ', evaluation:', self.evaluate.evaluate_name)

    def load_run_save_evaluate(self):
        
        # load dataset
        self.dataset.index_to_word = {index: word for index, word in enumerate(self.dataset.uniq_words)}
        self.dataset.word_to_index = {word: index for index, word in enumerate(self.dataset.uniq_words)}
        self.dataset.words_indexes = [self.dataset.word_to_index[w] for w in self.dataset.words]
        self.dataset.sequence_length = self.method.sequence_length

        # run MethodModule
        self.method.data = DataLoader(self.dataset, batch_size=self.method.batch_size)
        learned_result = self.method.run(self.dataset)
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        #self.evaluate.data = learned_result
        
        #return self.evaluate.evaluate(), None

        