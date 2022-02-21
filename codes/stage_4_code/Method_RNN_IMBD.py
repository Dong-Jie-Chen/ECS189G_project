'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from codes.base_class.method import method
from codes.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from codes.stage_4_code.Dataset_Loader import Dataset_Loader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import time


class Method_RNN_IMDB(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 50
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    embed_dim = 64
    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, dataset):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        #self.embedding = nn.EmbeddingBag(num_embeddings=vocab_size, embedding_dim=self.embed_dim, sparse=True)
        self.embedding = nn.Embedding(dataset.vocab_size, self.embed_dim, padding_idx=dataset.TEXT.vocab.stoi[dataset.TEXT.pad_token])
        self.rnn_1 = nn.LSTM(self.embed_dim, 16, 2, bidirectional=True)
        self.fc = nn.Linear(16 * 2, 1)
        self.act = nn.Sigmoid()


    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x, text_length, train_flag=True):
        '''Forward propagation'''
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'))
        packed_output, (hidden, cell) = self.rnn_1(packed_embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        dense_outputs = self.fc(hidden)
        outputs = self.act(dense_outputs)
        return outputs

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, dataset, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        mini_batches = dataset.create_mini_batches("IMBD", self.batch_size)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.BCELoss().to(self.device)
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        acc_hist = []
        loss_hist = []
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            start = time.time()
            epoch_loss = 0
            epoch_acc = 0
            for mini_batch in mini_batches:
                X, text_length = mini_batch.text
                optimizer.zero_grad()
                # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
                y_pred = self.forward(X, text_length).squeeze(1)
                # convert y to torch.tensor as well
                y_true = mini_batch.label
                # calculate the training loss
                train_loss = loss_function(y_pred, y_true)

                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()
            epoch_loss = train_loss.item()
            #epoch_loss += train_loss.item()
            #epoch_loss = epoch_loss / self.batch_size
            #epoch_acc = epoch_acc / len(mini_batches)


            duration = time.time() - start
            if (epoch-1)%1 == 0:
                accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': np.round(y_pred.detach().cpu())}
                epoch_acc = accuracy_evaluator.evaluate()
                print('Epoch:', epoch, 'Accuracy:', epoch_acc, 'Loss:', train_loss.item(), 'Time:', duration)
            #print(epoch_loss, epoch_acc)
            loss_hist.append(epoch_loss)
            acc_hist.append(epoch_acc)
            if (epoch-1)%1 == 0:
                pred_y = self.test(dataset)
                accuracy_evaluator.data = {'pred_y': pred_y.cpu(), 'true_y': self.data['test']['y']}
                print('Epoch:', epoch, 'Test Accuracy:', accuracy_evaluator.evaluate(), 'Test Loss:', train_loss.item())
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
        plt.savefig('history_IMBD.png')

    def test(self, dataset):
        # do the testing, and result the result
        mini_batches = dataset.create_mini_batches("IMBD", self.batch_size, train=False)
        y_pred = np.zeros(shape=(1))
        for mini_batch in mini_batches:
            X, text_length = mini_batch.text
            X = X.to(self.device)
            y_pred_batch = self.forward(X, text_length).squeeze(1)
            y_pred = np.append(y_pred, y_pred_batch.cpu().detach().numpy())
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        y_pred = torch.LongTensor(np.round(y_pred[1:]))
        return y_pred
    
    def run(self, dataset):
        print('method running...')
        print('--start training...')
        self.train(dataset, self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(dataset, self.data['test']['X'])
        return {'pred_y': pred_y.cpu(), 'true_y': self.data['test']['y']}
            