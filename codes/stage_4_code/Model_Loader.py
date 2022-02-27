'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from codes.base_class.result import result
import pickle
import torch


class Model_Loader(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None
    
    def load(self):
        print('loading results...')
        f_name = self.result_destination_folder_path + self.result_destination_file_name + '.pth'
        self.data = torch.load(f_name)
