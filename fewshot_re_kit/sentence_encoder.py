import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import optim
# from . import network
from fewshot_re_kit.network import embedding
from fewshot_re_kit.network import encoder
# from transformers import BertModel, BertTokenizer, BertConfig
class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=300,hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        # self.embedding =
        self.embedding = embedding.Embedding(word_vec_mat, max_length, word_embedding_dim)
        # self.encoder = encoder.Encoder(max_length, word_embedding_dim, hidden_size)
        self.encoder = encoder.Encoder_lstm(max_length, word_embedding_dim, hidden_size)

    def forward(self, inputs):
        x = self.embedding(inputs)  #torch.Size([100, 40, 300])
        x = self.encoder(x) #torch.Size([100, 230])
        return x


