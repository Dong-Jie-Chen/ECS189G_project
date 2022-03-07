'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from codes.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

class Setting_Train_Test_Split(setting):

    def print_setup_summary(self):
        print('dataset:', self.dataset.dataset_name, ', method:', self.method.method_name,
              ', setting:', self.setting_name, ', result:', self.result.result_name, ', evaluation:', self.evaluate.evaluate_name)

    def load_run_save_evaluate(self):

        # load dataset
        # load dataset
        loaded_data = self.dataset.load()

        X_train, y_train = loaded_data['graph']['X'][loaded_data['train_test_val']['idx_train']], \
                           loaded_data['graph']['y'][loaded_data['train_test_val']['idx_train']]
        X_val, y_val = loaded_data['graph']['X'][loaded_data['train_test_val']['idx_val']], \
                           loaded_data['graph']['y'][loaded_data['train_test_val']['idx_val']]
        X_test, y_test = loaded_data['graph']['X'][loaded_data['train_test_val']['idx_test']], \
                           loaded_data['graph']['y'][loaded_data['train_test_val']['idx_test']]

        print(Counter(np.array(loaded_data['graph']['y'])))
        print("Training set:", np.array(X_train).shape, np.array(y_train).max())
        print(Counter(np.array(y_train)))
        print("Testing set:", np.array(X_test).shape, np.array(y_test).max())
        print(Counter(np.array(y_test)))

        self.method.num_features = np.array(X_train).shape[1]
        self.method.num_classes = np.array(y_train).max() + 1
        # run MethodModule
        self.method.data = loaded_data
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None
        