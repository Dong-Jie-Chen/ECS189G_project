'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from codes.base_class.method import method
from codes.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from codes.stage_3_code.Dataset_Loader import Dataset_Loader
import torch
from torch import nn
import numpy as np


class Method_CNN_ORL(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 100
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.conv_layer_1 = nn.Conv2d(3, 32, 3, 1).to(self.device)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU().to(self.device)
        self.conv_layer_2 = nn.Conv2d(32, 32, 3, 1).to(self.device)
        self.conv_layer_3 = nn.Conv2d(32, 64, 3, 1).to(self.device)
        self.fc_layer_1 = nn.Linear(3*2*64, 128).to(self.device)
        self.fc_layer_2 = nn.Linear(128, 40).to(self.device)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_2 = nn.LogSoftmax(dim=1).to(self.device)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.conv_layer_1(x))
        h = nn.MaxPool2d(5).to(self.device)(h)
        h = nn.ReLU().to(self.device)(self.conv_layer_2(h))
        h = nn.MaxPool2d(4).to(self.device)(h)
        h = nn.ReLU().to(self.device)(self.conv_layer_3(h))
        h = torch.flatten(h, 1)
        h = nn.ReLU().to(self.device)(self.fc_layer_1(h))
        # outout layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_2(self.fc_layer_2(h))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html

        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss().to(self.device)
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        mini_batches = Dataset_Loader.create_mini_batches("ORL", X, y, self.batch_size)
        # Set up one-cycle learning rate scheduler
        #sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.learning_rate, epochs=20,
                                                    #steps_per_epoch=len(mini_batches))
        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...

            for mini_batch in mini_batches:
                X, y = mini_batch
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
                y_pred = self.forward(X)
                # convert y to torch.tensor as well
                y_true = y
                # calculate the training loss
                train_loss = loss_function(y_pred, y_true)

                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()
                #sched.step()
            if (epoch-1)%5 == 0:
                accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
            if (epoch-1)%5 == 0:
                pred_y = self.test(self.data['test']['X'])
                accuracy_evaluator.data = {'pred_y': pred_y.cpu(), 'true_y': self.data['test']['y']}
                print('Epoch:', epoch, 'Test Accuracy:', accuracy_evaluator.evaluate(), 'Test Loss:', train_loss.item())
    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)).to(self.device))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y.cpu(), 'true_y': self.data['test']['y']}
            