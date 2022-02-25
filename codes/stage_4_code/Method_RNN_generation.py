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
    max_epoch = 5
    learning_rate = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    embed_dim = 64
    sequence_length = 3
    num_layers = 3
    hidden_size = 256
    def __init__(self, mName, mDescription, dataset):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(dataset.vocab_size, self.embed_dim).to(self.device)
        self.rnn_1 = nn.LSTM(self.embed_dim, self.hidden_size, self.num_layers).to(self.device)
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
        loss_hist = []
        for epoch in range(self.max_epoch):
            start = time.time()
            state_h, state_c = self.init_state(self.sequence_length)
            epoch_loss = 0
            for batch, (X, y) in enumerate(self.data):
                optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
                y_pred, (state_h, state_c) = self.forward(X, (state_h, state_c))
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
            epoch_loss = train_loss.item()
            #epoch_loss += train_loss.item()
            #epoch_loss = epoch_loss / self.batch_size
            #epoch_acc = epoch_acc / len(mini_batches)


            duration = time.time() - start
            if (epoch-1)%1 == 0:
                print('Epoch:', epoch, 'Loss:', train_loss.item(), 'Time:', duration)
            #print(epoch_loss, epoch_acc)
            loss_hist.append(epoch_loss)

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color=color)
        ax1.plot(loss_hist, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig('history_IMBD.png')

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, sequence_length, self.hidden_size).to(self.device))

    def test(self, dataset, text, next_words=20):
        # do the testing, and result the result
        words = text.split(' ')
        state_h, state_c = self.init_state(len(words))
        for i in range(0, next_words):
            x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]]).to(self.device)
            y_pred, (state_h, state_c) = self.forward(x, (state_h, state_c))
            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            words.append(dataset.index_to_word[word_index])
        return words

    def run(self, dataset):
        print('method running...')
        print('--start training...')
        self.train()
        print('--start testing...')
        result = self.test(dataset, 'What did the')
        print(result)

        return result
            