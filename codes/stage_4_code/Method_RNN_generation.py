'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from codes.base_class.method import method
from codes.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from codes.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import time


class Method_RNN_generation(method, nn.Module):
    data = None
    word_to_index = None
    index_to_word = None
    max_epoch = 50
    learning_rate = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    embed_dim = 500
    sequence_length = 3
    num_layers = 1
    hidden_size = 512
    def __init__(self, mName, mDescription, dataset):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(dataset.vocab_size, self.embed_dim).to(self.device)
        self.rnn_1 = nn.GRU(self.embed_dim, self.hidden_size, self.num_layers).to(self.device)
        self.fc = nn.Linear(self.hidden_size, dataset.vocab_size).to(self.device)

    def forward(self, x, pre_sentence, train_flag=True):
        '''Forward propagation'''
        embedded = self.embedding(x)
        output, state = self.rnn_1(embedded, pre_sentence)
        dense_outputs = self.fc(output)
        return dense_outputs, state

    def train(self):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss().to(self.device)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        loss_hist, acc_hist = [], []
        for epoch in range(self.max_epoch):
            start = time.time()
            state_h, state_c = self.init_state(self.sequence_length)
            epoch_loss = 0
            epoch_acc = 0
            for batch, (X, y) in enumerate(self.data):
                optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
                y_pred, (state_h) = self.forward(X, (state_h))
                # convert y to torch.tensor as well
                y_true = y
                # calculate the training loss
                train_loss = loss_function(y_pred.transpose(1, 2), y)

                state_h = state_h.detach()
                state_c = state_c.detach()
                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()
                accuracy_evaluator.data = {'true_y': y_true[:,-1].cpu(), 'pred_y': y_pred[:,-1,:].max(1)[1].cpu()}
                epoch_acc += accuracy_evaluator.evaluate()
            epoch_loss = train_loss.item()
            epoch_acc = epoch_acc / (batch + 1)

            duration = time.time() - start
            if (epoch-1)%1 == 0:
                print('Epoch:', epoch, 'Accuracy:', epoch_acc, 'Loss:', train_loss.item(), 'Time:', duration)
            #print(epoch_loss, epoch_acc)
            loss_hist.append(epoch_loss)
            acc_hist.append(epoch_acc)

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
        plt.savefig('history_generation.png')

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, sequence_length, self.hidden_size).to(self.device))

    def test(self, dataset, text, next_words=20):
        words = text.split(' ')
        state_h, state_c = self.init_state(len(words))
        for i in range(0, next_words):
            x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]]).to(self.device)
            y_pred, (state_h) = self.forward(x, (state_h))
            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
            word_index = np.argmax(p)
            words.append(dataset.index_to_word[word_index])
        return words

    def run(self, dataset):
        self.word_to_index = dataset.word_to_index
        self.index_to_word = dataset.index_to_word
        print('method running...')
        print('--start training...')
        self.train()
        print('--start testing...')
        result = self.test(dataset, 'what did the')
        print(result)

        return result
            