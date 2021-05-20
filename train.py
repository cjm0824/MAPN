# import models
from fewshot_re_kit.data_loader import myJsonFileDataLoader
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder
# from fewshot_re_kit.sentence_encoder import TextNet
from models.proto_hatt import ProtoHATT
from models.proto import Proto
from models.proto_idatt import ProtoIDATT
import sys
import torch
from torch import optim

# print("choose your model_name：(proto|proto_fatt_pool|proto_fatt_pool_2|proto_hatt_pool_2|proto_fatt_pool_3|proto_fatt_pool_4|proto_hatt|proto_fatt|proto_iatt|proto_best_support|proto_iatt_sort)")
# model_name = input()
model_name = 'proto_idatt'
N = 5
K = 5
noise_rate = 0
if len(sys.argv) > 1:
    model_name = sys.argv[1]
if len(sys.argv) > 2:
    N = int(sys.argv[2])
if len(sys.argv) > 3:
    K = int(sys.argv[3])
if len(sys.argv) > 4:
    noise_rate = float(sys.argv[4])

print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
print("Model: {}".format(model_name))

max_length =40 

train_data_loader = myJsonFileDataLoader('./data/train_data.json', './data/w2v.json', max_length=max_length)#16821*300
val_data_loader = myJsonFileDataLoader('./data/train_data.json', './data/w2v.json', max_length=max_length)
test_data_loader = myJsonFileDataLoader('./data/test_data.json', './data/w2v.json', max_length=max_length)

framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
sentence_encoder = CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)#,model)

if model_name == 'proto':
    model = Proto(sentence_encoder)
    framework.train(model, model_name, 4, 5, N, K, 5, train_iter=15000, noise_rate=noise_rate)
elif model_name == 'proto_hatt':
    model = ProtoHATT(sentence_encoder, K)
    framework.train(model, model_name, 4, 5, N, K, 5, noise_rate=noise_rate)
elif model_name == 'proto_idatt':
    model = ProtoIDATT(sentence_encoder, K)
    framework.train(model, model_name, 4, 5, N, K, 5, noise_rate=noise_rate)
else:
    raise NotImplementedError

