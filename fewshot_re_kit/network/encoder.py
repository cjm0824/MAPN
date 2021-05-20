'''
CNN做编码器，一维卷积，最大池化，
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch import optim

class Encoder(nn.Module):
    def __init__(self, max_length, word_embedding_dim=300, hidden_size=230):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim # + pos_embedding_dim * 2
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3, padding=1)   #卷积
        self.pool = nn.MaxPool1d(max_length)    #最大池化
    def forward(self, inputs):
        return self.cnn(inputs)

    def cnn(self, inputs): #inputs torch.Size([100,  40,300])
        x = self.conv(inputs.transpose(1, 2)) #torch.Size([100, 40, 300])->转置、卷积-> torch.Size([100, 230, 40])
        x = F.relu(x)   #torch.Size([100, 230, 40])
        x = self.pool(x)    #torch.Size([100, 230, 1])
        # x = x.squeeze(2)    #torch.Size([100, 230])
        return x.squeeze(2) # n x hidden_size


class Encoder_lstm(nn.Module):
    def __init__(self, max_length, word_embedding, hidden_size, num_layers=1):
        super(Encoder_lstm, self).__init__()
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding
        self.num_layers=num_layers
        self.rnn = nn.RNN(input_size=self.embedding_dim,
                            hidden_size=self.hidden_size,
                            nonlinearity='relu',
                            batch_first=True)
        self.cnn = nn.Conv1d(self.hidden_size, self.hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1d(max_length)
    def forward(self, inputs):
        # x= inputs.transpose(0, 1)
        outputs, _=self.rnn(inputs)
        outs = self.cnn(outputs.transpose(1, 2))
        outs = F.relu(outs)
        outs = self.pool(outs).squeeze(2)
        return outs

