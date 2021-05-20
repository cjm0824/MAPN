'''
每个词的embedding是词向量
将原论文中的位置编码部分去除掉了
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import json
import numpy as np
# from transformers import BertModel

class Embedding(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=300):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        
        # Word embedding
        unk = torch.randn(1, word_embedding_dim) / math.sqrt(word_embedding_dim)
        blk = torch.zeros(1, word_embedding_dim)
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0] + 2, self.word_embedding_dim, padding_idx=word_vec_mat.shape[0] + 1)
        self.word_embedding.weight.data.copy_(torch.cat((word_vec_mat, unk, blk), 0))


    def forward(self, inputs):#inputd:torch.Size([4, 7, 8, 40]),s_labels:(4,7*8)
        word = inputs['word']#torch.Size([4, 7, 8, 40])

        x = self.word_embedding(word)#torch.Size([4, 7, 8, 40, 300])
        # f = open('_processed_data/w2v_word2id.json', encoding='utf-8')
        # res = f.read()
        # word_id = json.loads(res)#词表
        #
        # all = open('word_frequency/new_all.json', 'r', encoding='utf-8')
        # all_g_read = all.read()
        # all_word_fre = json.loads(all_g_read)
        #
        # for i in range(len(s_labels)):
        #     input = x[i]#torch.Size([7, 8, 40, 300])
        #     in_word = word[i]#torch.Size([7, 8, 40])
        #     for j in range(len(in_word)):
        #         for k in range(len(in_word[j])):
        #             label = s_labels[i][j * len(in_word[j]) + k]  # 每个人句子对应的类
        #             # g = open('word_frequency/'+label+'.json','r',encoding='utf-8')
        #             # g_read = g.read()
        #             # word_fre = json.loads(g_read)
        #             word_f = all_word_fre[label] if label in all_word_fre else 'Key Not Exist!'
        #             wf = []
        #             sentence_embedding = input[j][k] #（40*300）
        #             sentence_word = in_word[j][k]#每一个句子 40
        #             for k_ in sentence_word:
        #                 a = str(int(k_))
        #                 fre = word_f[a] if a in word_f else 'Key Not Exist!'
        #                 wf.append(fre)
        #             arf = []  #每个词的权重
        #             wf_array = np.array(wf) #转成array方便计算和
        #             for wfi in wf:
        #                 arf.append(wfi/sum(wf_array))
        #             for w in range(len(arf)):
        #                 sentence_embedding[w] = arf[w]*sentence_embedding[w]
        # x = x.view(-1, self.max_length,self.word_embedding_dim)#torch.Size([140, 40, 300])
        return x

