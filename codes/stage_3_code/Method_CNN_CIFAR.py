'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from codes.base_class.method import method
from codes.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from codes.stage_3_code.Dataset_Loader import Dataset_Loader
from torchvision import transforms
import torch
from torch import nn
import numpy as np
import time


class Method_CNN_CIFAR(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 512
    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.conv_BN_ReLU_1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU()).to(self.device)
        self.conv_BN_ReLU_2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU()).to(self.device)
        #self.conv_BN_ReLU_3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU()).to(self.device)
        self.conv_BN_ReLU_4 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU()).to(self.device)
        self.conv_BN_ReLU_5 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU()).to(self.device)
        #self.conv_BN_ReLU_6 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU()).to(self.device)
        #self.conv_BN_ReLU_7 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU()).to(self.device)
        #self.conv_BN_ReLU_8 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU()).to(self.device)
        #self.conv_BN_ReLU_9 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU()).to(self.device)
        #self.conv_BN_ReLU_10 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU()).to(self.device)
        #self.conv_BN_ReLU_11 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU()).to(self.device)
        #self.conv_BN_ReLU_12 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU()).to(self.device)
        #self.conv_BN_ReLU_13 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU()).to(self.device)
        self.res_1 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU()).to(self.device)
        self.res_2 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU()).to(self.device)
        self.fc_layer_1 = nn.Sequential(nn.Linear(1*1*512, 512), nn.BatchNorm1d(512), nn.ReLU()).to(self.device)
        self.fc_layer_2 = nn.Linear(512, 10).to(self.device)
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.train_tfms = nn.Sequential(transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Normalize(*stats,inplace=True))
        self.test_tfms = nn.Sequential(transforms.Normalize(*stats))
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_last = nn.LogSoftmax(dim=1).to(self.device)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x, train_flag=True):
        '''Forward propagation'''
        # hidden layer embeddings
        if train_flag:
          x = self.train_tfms(x)
        else:
          x = self.test_tfms(x)
        h = self.conv_BN_ReLU_1(x)
        h = self.conv_BN_ReLU_2(h)
        h = nn.MaxPool2d(2).to(self.device)(h)
        h = self.res_1(h) + h
        h = self.conv_BN_ReLU_4(h)
        h = nn.MaxPool2d(2).to(self.device)(h)
        h = self.conv_BN_ReLU_5(h)
        #h = self.conv_BN_ReLU_6(h) + h
        #h = self.conv_BN_ReLU_7(h)
        h = nn.MaxPool2d(2).to(self.device)(h)
        #h = self.conv_BN_ReLU_8(h)
        h = self.res_2(h) + h
        #h = self.conv_BN_ReLU_10(h)
        h = nn.AvgPool2d(4).to(self.device)(h)
        #h = self.conv_BN_ReLU_11(h)
        #h = self.conv_BN_ReLU_12(h)
        #h = self.conv_BN_ReLU_13(h)
        #h = nn.MaxPool2d(2).to(self.device)(h)
        h = torch.flatten(h, 1)
        h = self.fc_layer_1(h)
        h = self.fc_layer_2(h)
        # outout layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_last(h)
        return h

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay = 1e-4)
        # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html

        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss().to(self.device)
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        mini_batches = Dataset_Loader.create_mini_batches("CIFAR", X, y, self.batch_size)
        # Set up one-cycle learning rate scheduler
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.learning_rate, epochs=self.max_epoch,
                                                    steps_per_epoch=len(mini_batches))
        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            start = time.time()
            for mini_batch in mini_batches:
                X, y = mini_batch
                #X, y = X.to(self.device), y.to(self.device)
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
                nn.utils.clip_grad_value_(self.parameters(), 0.1)
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()
                sched.step()
            duration = time.time() - start
            if (epoch-1)%1 == 0:
                accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item(), 'Time:', duration)
            if (epoch-1)%1 == 0:
                pred_y = self.test(self.data['test']['X'])
                accuracy_evaluator.data = {'pred_y': pred_y.cpu(), 'true_y': self.data['test']['y']}
                print('Epoch:', epoch, 'Test Accuracy:', accuracy_evaluator.evaluate(), 'Test Loss:', train_loss.item())
        del X, y
    def test(self, X):
        # do the testing, and result the result
        y = np.zeros(X.shape)
        mini_batches = Dataset_Loader.create_mini_batches("CIFAR", X, y, self.batch_size, shuffle=False)
        y_pred = np.zeros(shape=(1,10))
        for mini_batch in mini_batches:
            X, y = mini_batch
            X = X.to(self.device)
            y_pred_batch = self.forward(X, train_flag=False)
            y_pred = np.append(y_pred, y_pred_batch.cpu().detach().numpy(), axis=0)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        y_pred = torch.LongTensor(y_pred[1:])
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y.cpu(), 'true_y': self.data['test']['y']}
            