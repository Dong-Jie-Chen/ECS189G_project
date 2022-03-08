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
#from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import copy
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class Method_GNN(method, nn.Module):
    data = None
    max_epoch = 200
    learning_rate = 0.01
    weight_decay = 5e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_features = 0
    hidden_channels = 128
    num_classes = 0
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.conv1 = GraphConvolution(self.num_features, self.hidden_channels).to(self.device)
        self.conv2 = GraphConvolution(self.hidden_channels, self.num_classes).to(self.device)

    def forward(self, x, edge_index):
        '''Forward propagation'''
        x = self.conv1(x, edge_index)
        x = x.relu().to(self.device)
        x = nn.Dropout(0.5).to(self.device)(x)
        x = self.conv2(x, edge_index)
        return x

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_function = nn.CrossEntropyLoss().to(self.device)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        X = torch.FloatTensor(np.array(X)).to(self.device)
        y = torch.LongTensor(np.array(y)).to(self.device)
        train_acc_hist = []
        train_loss_hist = []
        val_acc_hist = []
        val_loss_hist = []
        test_acc_hist = []
        best_model = None
        best_val_acc = 0
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            optimizer.zero_grad()
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(X.to(self.device), self.data['graph']['utility']['A'].to(self.device))[self.data['train_test_val']['idx_train']]
            y_pred_val = self.forward(X.to(self.device), self.data['graph']['utility']['A'].to(self.device))[
                self.data['train_test_val']['idx_val']]
            y_pred_test = self.forward(X.to(self.device), self.data['graph']['utility']['A'].to(self.device))[
                self.data['train_test_val']['idx_test']]
            # convert y to torch.tensor as well
            y_true = y[self.data['train_test_val']['idx_train']]
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            accuracy_evaluator.data = {'true_y': y[self.data['train_test_val']['idx_val']].cpu(), 'pred_y': y_pred_val.max(1)[1].cpu()}
            val_acc = accuracy_evaluator.evaluate()
            val_acc_hist.append(val_acc)
            val_loss_hist.append(train_loss.item())

            accuracy_evaluator.data = {'true_y': y[self.data['train_test_val']['idx_test']].cpu(),
                                       'pred_y': y_pred_test.max(1)[1].cpu()}
            test_acc = accuracy_evaluator.evaluate()
            test_acc_hist.append(test_acc)

            if val_acc > best_val_acc:
                best_model = copy.deepcopy(self.state_dict())
                best_val_acc = val_acc
                best_epoch = epoch

            accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
            epoch_acc = accuracy_evaluator.evaluate()
            train_acc_hist.append(epoch_acc)
            train_loss_hist.append(train_loss.item())
            if epoch%20 == 0:
                print('Epoch:', epoch, 'Accuracy:', epoch_acc, 'Loss:', train_loss.item())
        self.load_state_dict(best_model)
        print("Best epoch:", best_epoch)
        self.plot_hist(train_loss_hist, train_acc_hist, "GNN_train.png")
        self.plot_hist(val_loss_hist, val_acc_hist, "GNN_val.png")
        self.plot_hist([], test_acc_hist, "GNN_test.png")


    def plot_hist(self, loss_hist, acc_hist, file_name):
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color=color)
        ax1.plot(loss_hist, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('acc', color=color)  # we already handled the x-label with ax1
        ax2.plot(acc_hist, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(file_name)

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)).to(self.device), self.data['graph']['utility']['A'].to(self.device))[self.data['train_test_val']['idx_test']]
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

class GraphConvolution(Module):
    def __init__(self, in_features, out_hidden, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_hidden
        self.weight = Parameter(torch.zeros(in_features, out_hidden))
        if bias:
            self.bias = Parameter(torch.ones(out_hidden))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj_m):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj_m, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
            