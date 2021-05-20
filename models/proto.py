import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Proto(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, hidden_size=230):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, hidden_size)

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)  # (B, 1, N, D)   (B, N*Q, 1,D)

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        support = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query) # (B * N * Q, D)
        support = support.view(-1, N, K, self.hidden_size) # (B, N, K, D)
        query = query.view(-1, N * Q, self.hidden_size) # (B, N * Q, D)

        B = support.size(0) # Batch size
        NQ = query.size(1) # Num of instances for each batch in the query set
         
        # Prototypical Networks 
        support = torch.mean(support, 2) # Calculate prototype for each class  (B, N, D)
        dis = self.__batch_dist__(support, query) #torch.Size([4, 25, 5])
        logits = -self.__batch_dist__(support, query)#torch.Size([4, 25, 5])
        _, pred = torch.max(logits.view(-1, N), 1)
        return dis, logits, pred
    
    
    
