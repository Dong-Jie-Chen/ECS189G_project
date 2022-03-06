'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from codes.base_class.method import method
from codes.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
from torch_geometric.nn import GCNConv



class Method_GNN(method, nn.Module):
    data = None
    max_epoch = 200
    learning_rate = 1e-3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_features = 0
    hidden_channels = 128
    num_classes = 0
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.conv1 = GCNConv(self.num_features, self.hidden_channels).to(self.device)
        self.conv2 = GCNConv(self.hidden_channels, self.num_classes).to(self.device)

    def forward(self, x, edge_index):
        '''Forward propagation'''
        x = self.conv1(x, edge_index)
        x = x.relu().to(self.device)
        x = nn.Dropout(0.5).to(self.device)(x)
        x = self.conv2(x, edge_index)
        return x

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss().to(self.device)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        X = torch.FloatTensor(np.array(X)).to(self.device)
        y = torch.LongTensor(np.array(y)).to(self.device)
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            optimizer.zero_grad()
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(X.to(self.device), self.data['graph']['edge'].to(self.device))[self.data['train_test_val']['idx_train']]
            # convert y to torch.tensor as well
            y_true = y[self.data['train_test_val']['idx_train']]
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch%50 == 0:
                accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)).to(self.device), self.data['graph']['edge'].to(self.device))[self.data['train_test_val']['idx_test']]
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def run(self):
        self.__init__("", "")
        print('method running...')
        print('--start training...')
        self.train(self.data['graph']['X'], self.data['graph']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['graph']['X'])
        return {'pred_y': pred_y.cpu(), 'true_y': self.data['graph']['y'][self.data['train_test_val']['idx_test']]}
            