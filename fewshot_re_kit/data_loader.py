import json
import os
import multiprocessing
import numpy as np
import random
import torch
from torch.autograd import Variable


class myJsonFileDataLoader():


    def __init__(self, file_name, word_vec_file_name, max_length=40, case_sensitive=False, reprocess=True, cuda=True):

        '''

        :param file_name: train_data.json
        :param word_vec_file_name: w2v.json
        :param max_length: 句子长度
        '''
        self.file_name = file_name
        self.word_vec_file_name = word_vec_file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        self.cuda = cuda
        #获取进程加载模块
        if reprocess:  # or not self._load_preprocessed_file():  # Try to load pre-processed files:
            # Check files 检查文件是否存在
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")
            if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
                raise Exception("[ERROR] Word vector file doesn't exist")

            # 加载数据存储文件（训练集和词向量文件），获取原始数据和原始词向量
            print("加载数据集文件...")
            self.ori_data = json.load(open(self.file_name, "r", encoding='utf-8'))
            print("加载完成")
            print("加载词向量文件...")
            self.ori_word_vec = json.load(open(self.word_vec_file_name, "r", encoding='utf-8'))
            print("加载完成")

            # Pre-process word vec
            # word_vec_tot：获取词向量总数
            self.word2id = {}
            self.word_vec_tot = len(self.ori_word_vec)
            #？？？
            UNK = self.word_vec_tot #unk:16821
            #？？？
            BLANK = self.word_vec_tot + 1 #BLANK:16822
            # 获取词向量的维度
            self.word_vec_dim = len(self.ori_word_vec[0]['vec'])
            # 得到word_vec_tot个维度为word_vec_dim的词向量
            print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
            print("Building word vector matrix and mapping...")
            #构建一个元素为0的词向量矩阵
            self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
            for cur_id, word in enumerate(self.ori_word_vec):
                w = word['word']#w:"词"
                # if not case_sensitive:
                #     w = w.lower()
                self.word2id[w] = cur_id
                self.word_vec_mat[cur_id, :] = word['vec']
                self.word_vec_mat[cur_id] = self.word_vec_mat[cur_id] / np.sqrt(np.sum(self.word_vec_mat[cur_id] ** 2))

            #UNK和BLANK的作用是什么 我们跑一下代码吧？
            self.word2id['UNK'] = UNK
            self.word2id['BLANK'] = BLANK
            print("Finish building")

            # 预处理数据 Pre-process data
            print("Pre-processing data...")
            self.instance_tot = 0  #训练集实例总数
            #relation:类名
            for relation in self.ori_data:
                self.instance_tot += len(self.ori_data[relation])

            self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)

            self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
            self.rel2scope = {}  # left close right open
            i = 0
            for relation in self.ori_data:
                self.rel2scope[relation] = [i, i]
                for ins in self.ori_data[relation]:
                    words = ins['tokens']
                    cur_ref_data_word = self.data_word[i]
                    for j, word in enumerate(words):
                        if j < max_length:
                            if word in self.word2id:
                                cur_ref_data_word[j] = self.word2id[word]
                            else:
                                cur_ref_data_word[j] = UNK
                    for j in range(j + 1, max_length):
                        cur_ref_data_word[j] = BLANK
                    self.data_length[i] = len(words)
                    if len(words) > max_length:
                        self.data_length[i] = max_length
                    i += 1
                self.rel2scope[relation][1] = i

            print("Finish pre-processing")

            print("Storing processed files...")
            name_prefix = '.'.join(file_name.split('/')[-1].split('.')[:-1])
            word_vec_name_prefix = '.'.join(word_vec_file_name.split('/')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)
            np.save(os.path.join(processed_data_dir, name_prefix + '_word.npy'), self.data_word)

            np.save(os.path.join(processed_data_dir, name_prefix + '_length.npy'), self.data_length)
            json.dump(self.rel2scope, open(os.path.join(processed_data_dir, name_prefix + '_rel2scope.json'), 'w',encoding='utf-8'),ensure_ascii=False)
            np.save(os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy'), self.word_vec_mat)
            json.dump(self.word2id, open(os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json'), 'w',encoding='utf-8'),ensure_ascii=False)
            print("Finish storing")

    def next_one(self, N, K, Q, noise_rate=0):
        #target_classes 随机选择的N个类别的集合
        target_classes = random.sample(self.rel2scope.keys(), N)
        noise_classes = []
        for class_name in self.rel2scope.keys():
            if not (class_name in target_classes):
                noise_classes.append(class_name)
        support_set = {'word': []}
        query_set = {'word': []}
        query_label = []
        second_class_name = []
        for i, class_name in enumerate(target_classes):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), K + Q, False)  # 这个类别选K+Q个
            word = self.data_word[indices]
            support_word, query_word, _ = np.split(word, [K, K + Q])
            for j in range(K):
                prob = np.random.rand()
                if prob < noise_rate:
                    noise_class_name = noise_classes[np.random.randint(0, len(noise_classes))]
                    scope = self.rel2scope[noise_class_name]
                    indices = np.random.choice(list(range(scope[0], scope[1])), 1, False)
                    word = self.data_word[indices]

                    support_word[j] = word

            support_set['word'].append(support_word)

            query_set['word'].append(query_word)

            query_label += [i] * Q
            second_class_name += [class_name] * Q

        support_set['word'] = np.stack(support_set['word'], 0)

        query_set['word'] = np.concatenate(query_set['word'], 0)

        query_label = np.array(query_label)

        perm = np.random.permutation(N * Q)
        query_set['word'] = query_set['word'][perm]

        query_label = query_label[perm]

        query_class_labels = []
        for i in perm.tolist():
            query_class_labels.append(second_class_name[i])

        return support_set, query_set, query_label, second_class_name, query_class_labels

    def next_batch(self, B, N, K, Q, noise_rate=0):
        support = {'word': []}
        query = {'word': []}
        label = []
        second_class_names = []
        query_labels = []
        for one_sample in range(B):
            current_support, current_query, current_label, second_class_name, query_class_labels = self.next_one(N, K, Q, noise_rate=noise_rate)
            support['word'].append(current_support['word'])

            query['word'].append(current_query['word'])

            label.append(current_label)
            second_class_names.append(second_class_name)
            query_labels.append(query_class_labels)
        support['word'] = Variable(torch.from_numpy(np.stack(support['word'], 0)).long().view(-1, self.max_length))
        query['word'] = Variable(torch.from_numpy(np.stack(query['word'], 0)).long().view(-1, self.max_length))
        label = Variable(torch.from_numpy(np.stack(label, 0).astype(np.int64)).long())

        # To cuda
        if self.cuda:
            for key in support:
                support[key] = support[key].cuda()
            for key in query:
                query[key] = query[key].cuda()
            label = label.cuda()

        return support, query, label, second_class_names, query_labels

    def lookup(self, ins):
        words = ins['tokens']
        cur_ref_data_word = np.zeros(self.max_length, dtype=np.int32)
        for j, word in enumerate(words):
            word = word.lower()
            if j < self.max_length:
                if word in self.word2id:
                    cur_ref_data_word[j] = self.word2id[word]
                else:
                    cur_ref_data_word[j] = self.word2id['UNK']
        for j in range(j + 1, self.max_length):
            cur_ref_data_word[j] = self.word2id['BLANK']
        data_length = len(words)
        if len(words) > self.max_length:
            data_length = self.max_length

        return cur_ref_data_word, data_length


if __name__ == '__main__':
    maxlen = 40
    train_data_loader = myJsonFileDataLoader('../data/train_data.json', '../data/w2v.json', max_length=maxlen)
    # train_data_loader = myJsonFileDataLoader('../result_data/train_data.json', '../result_data/q2v.json',
    #                                          max_length=maxlen)

    # 把词向量转换成json格式
    # from gensim.models import Word2Vec
    # embedding_path = '../result_data/w2v/new_w2v.model'
    # embdding_dict = Word2Vec.load(embedding_path)
    # w2v_dict = []
    # for word in embdding_dict.wv.vocab:
    #     w2v_dict.append({"word":word,"vec":[float(x) for x in list(embdding_dict[word])]})
    # with open("../result_data/w2v.json",'w',encoding='utf-8') as f:
    #     json.dump(w2v_dict,f,ensure_ascii=False)
