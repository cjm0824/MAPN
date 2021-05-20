# -*- coding:utf-8 -*-
'''
生成all_data.json文件
'''
import os
base_data_dir = '../场景'
import jieba
import jieba.posseg
import json
all_kind = {}
test_all_kind = {}
test_data_dir = '../所有小类'

for first_kind in os.listdir(base_data_dir):
    # print(first_kind)
    second_kinds_path = os.path.join(base_data_dir,first_kind)
    # print(second_kinds_path)
    # for second_kind in os.listdir(second_kinds_path):
    second_kind = first_kind.split('.')[0]
    all_kind[second_kind] = []
    # print(second_kind)
    with open(second_kinds_path,'r',encoding='utf-8') as f:
        for line in f:
            tokens = jieba.lcut(line.strip())
            all_kind[second_kind].append({'tokens':tokens})
            # print(tokens)
# json_data = json.dumps(all_kind,ensure_ascii=False)
json.dump(all_kind,open("../data/train_data.json",'w',encoding='utf-8'),ensure_ascii=False)
# json.dum

for test_first_kind in os.listdir(test_data_dir):
    print(test_first_kind)
    test_path = os.path.join(test_data_dir, test_first_kind)
    second_kind = test_first_kind.split('.')[0]
    test_all_kind[second_kind] = []
    with open(test_path, 'r', encoding='utf-8') as g:
        for row in g:
            tokens = jieba.lcut(row.strip())
            test_all_kind[second_kind].append({'tokens':tokens})
json.dump(test_all_kind, open('../data/test_data.json', 'w', encoding='utf-8'), ensure_ascii=False)