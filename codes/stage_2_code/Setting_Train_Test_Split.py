'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from codes.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test_Split(setting):
    fold = 3

    def print_setup_summary(self):
        print('dataset:', self.dataset['train'].dataset_name, ', method:', self.method.method_name,
              ', setting:', self.setting_name, ', result:', self.result.result_name, ', evaluation:', self.evaluate.evaluate_name)

    def load_run_save_evaluate(self):
        
        # load dataset
        train_data = self.dataset['train'].load()
        test_data = self.dataset['test'].load()


        X_train, X_test, y_train, y_test = train_data['X'], test_data['X'], train_data['y'], test_data['y']

        print("Training set:", np.array(X_train).shape)
        print("Testing set:", np.array(X_test).shape)

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None

        