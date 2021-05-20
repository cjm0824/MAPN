import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from math import sqrt

class ProtoIDATT(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, shots, hidden_size=230):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.drop = nn.Dropout()
        # for instance-level attention
        self.fc = nn.Linear(hidden_size, hidden_size, bias=True)
        # for feature-level attention
        #
        self.conv1 = nn.Conv2d(1, 32, (shots, 1), padding=(shots // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (shots, 1), padding=(shots // 2, 0))
        self.conv_final = nn.Conv2d(32, 1, (shots, 1), stride=(shots, 1))
        # # self.fc1 = nn.Linear(4*5*shots, 4*5*shots)
        self.conv_end = nn.Conv2d(1, 1, (shots, 1), stride=(shots, 1))
        # self.fc2 = nn.Linear(4*5*5, 20)
        # #

        self.dim_k = hidden_size
        self.dim_v = hidden_size
        self.linear_q = nn.Linear(hidden_size, self.dim_k, bias=False)
        self.linear_k = nn.Linear(hidden_size, self.dim_k, bias=False)
        self.linear_v = nn.Linear(hidden_size, self.dim_v, bias=False)
        self._norm_fact = 1 / sqrt(self.dim_k)
    def __dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)

    def __batch_dist__(self, S, Q, score=None):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3, score)

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        support = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query)  # (B * N * Q, D)
        support = support.view(-1, N, K, self.hidden_size)  # (B, N, K, D)
        query = query.view(-1, N * Q, self.hidden_size)  # (B, N * Q, D)

        B = support.size(0)  # Batch size
        NQ = query.size(1)  # Num of instances for each batch in the query set


        x = support.view(B*N, K, self.hidden_size)#torch.Size([20, 5, 230])
        q = self.linear_q(x)  # batch, n, dim_k torch.Size([20, 5, 230])
        k = self.linear_k(x)  # batch, n, dim_k torch.Size([20, 5, 230])
        v = self.linear_v(x)  # batch, n, dim_v torch.Size([20, 5, 230])
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n torch.Size([20, 5, 5])
        dist = torch.softmax(dist, dim=-1)  # batch, n, n torch.Size([20, 5, 5])
        x = torch.bmm(dist, v).unsqueeze(1) #torch.Size([20, 5, 230])

        fea_att_score = F.relu(self.conv1(x))
        fea_att_score = self.conv_final(fea_att_score)  # (B * N, 1, 1, D)
        fea_att_score = F.relu(fea_att_score)
        # fea_att_score = self.conv_end(x) #torch.Size([20, 1, 1, 230])
        # fea_att_score = F.relu(fea_att_score)
        fea_att_score = fea_att_score.view(B, N, self.hidden_size).unsqueeze(1) #torch.Size([4, 1, 5, 230])

        # # Prototypical Networks
        # support_proto = support_result.view(B, -1, K, self.hidden_size)
        # support_proto = support_proto.sum(2)
        # support_proto = support_proto.unsqueeze(1).expand(-1, NQ, -1, -1)
        support = torch.mean(support, 2) #   (B, N, D) 计算每个实例的原型
        # support = support.unsqueeze(1)
        logits = -self.__batch_dist__(support, query, fea_att_score)
        # logits = -self.__batch_dist__(support, query, fea_att_score)
        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred



