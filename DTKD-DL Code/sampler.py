# Author:徐睿峰
# -*- codeing: utf-8 -*-
# @time ：2024/4/23 15:24
# @Author :xuruifeng
# @Site : 
# @file : sampler.py
# @Sofeware : PyCharm


import numpy as np
import json
import random
from transformers import BertTokenizer
import logging
logging.basicConfig(filename='train.log', level=logging.DEBUG)
# 将print输出重定向到日志文件
print = logging.debug
class data_sampler(object):
    def __init__(self, config=None, seed=None, relation_file=None, rel_des_file=None, rel_index=None,
                 rel_cluster_labels=None, training_file=None, valid_file=None, test_file=None):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path,
                                                       additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
        self.rel, self.rel_id = self.read_relations(relation_file)
        self.rel_des, self.rel_des_id,self.rel_emb,self.rel_mask= self.read_relation_des(rel_des_file)
        self.training_data = self.load_data(training_file,seed)
        self.valid_data = self.load_data(valid_file,seed)
        self.test_data = self.load_data_test(test_file,seed)
        self.task_length = config.task_length

        # Find the data for the round
        self.cluster_to_labels = self.get_cluster_label(rel_index=rel_index, rel_cluster_labels=rel_cluster_labels)
        self.shuffle_index = self.get_seed(seed)
        self.task = 0

        # record relations
        self.seen_relations = []
        self.history_test_data = {}

    def read_relations(self, file):
        '''
        :param file: input relation file
        :return:  a list of relations, and a mapping from relations to their ids.
        '''
        rel = []
        rel_id = {}
        with open(file) as file_in:
            for line in file_in:
                rel.append(line.strip())
        for i, x in enumerate(rel):
            rel_id[x] = i
        return rel, rel_id

    def read_relation_des(self, file):

        rel_des = []
        rel_emb=[]
        rel_des_id = {}
        rel_mask=[]
        with open(file) as file_in:
            for line in file_in:
                rel_des.append(line.strip())
        for i in range(len(rel_des)):
            rel_per_des = ['[CLS]']
            next_rel_des = rel_des[i].split(" ")
            for i in range(len(next_rel_des)):
                rel_per_des.append(next_rel_des[i])
            rel_per_des.append('[SEP]')
            rel_emb.append(rel_per_des)
            rel_per_mask = np.zeros((256), dtype=np.int32)
            rel_per_mask[:len(rel_per_des)] = 1
            rel_mask.append(rel_per_mask)
        for i, x in enumerate(rel_des):
            rel_des_id[x] = i
        return rel_des, rel_des_id, rel_emb, rel_mask

    def load_data(self, file,seed):
        # 这里的处理方式相对于之前emnlp22 简单
        #  这里的load_data 居然没有使用 dataloader 这里的速度会不会很慢 没有进行批量提取
        samples = []
        with open(file,encoding='utf-8') as file_in:
            for line in file_in:
                # 转义符 \t 代表一个水平制表符（Horizontal Tab），
                # 通常用于在文本中创建制表位，类似于在键盘上按下“Tab”键所产生的效果
                # 这里也可以看出来有人对fewRel 进行了处理，每一个数据中间的并非用空格隔开的 而是用'\t'分开的
                # 举例：['25', '6 14 29 19 9 61 49 13 21 40',
                #       "a disciple of paul gauguin and friend of paul sérusier ,
                #       he belonged to the circle of artists known as ' les nabis . '",
                #       'paul sérusier', '8 9', 'les nabis', '21 22', 'Q326606', 'Q503708']
                items = line.strip().split('\t')
                # item[0]='25'
                if (len(items[0]) > 0):
                    #  当前句子的关系id或者关系索引
                    relation_ix = int(items[0])
                    # 代表后续有东西的话
                    if items[1] != 'noNegativeAnswer':
                        # candidata_ixs 这里就是10各类别的id
                        # 候选索引
                        candidate_ixs = [int(ix) for ix in items[1].split()]
                        # 将整个样本句子拿出来
                        #  这里为啥没有以.对句子进行划分
                        sentence = items[2].split('\n')[0]
                        # 头部实体
                        headent = items[3]
                        # headidex 头部实体的索引
                        headidx = [int(ix) for ix in items[4].split()]
                        # 尾实体
                        tailent = items[5]
                        # 尾实体的id
                        tailidx = [int(ix) for ix in items[6].split()]
                        # 头尾实体的id
                        headid = items[7]
                        tailid = items[8]
                        #  把数据进行处理之后再重新整理
                        samples.append(
                            [relation_ix, candidate_ixs, sentence, headent, headidx, tailent, tailidx,
                             headid, tailid])
        # 创建了含有80个关系的列表,并且每个元素同样都是列表
        read_data = [[] for i in range(self.config.num_of_relation)]

        for sample in samples:
            #  单独拿出来每个句子,进行简单预处理
            text = sample[2]  # text=sentcence
            #  这里的作用是将text 变成一个一个数据
            #  举例:['a', 'disciple', 'of', 'paul', 'gauguin', 'and', 'friend', 'of', 'paul']
            split_text = text.split(" ")
            # 在头实体和尾实体前面加入[E11]和[E12] [E21]和[E22]
            new_headent = ' [E11] ' + sample[3] + ' [E12] '
            new_tailent = ' [E21] ' + sample[5] + ' [E22] '

            if sample[4][0] < sample[6][0]:
                #  在头实体前面加入前半句话 ，在头实体后面加入后半句话，当前到为实体之前 之后再加改好的尾实体和后面的话
                #  其实就是将改好的尾实体和头实体加入到他该到的地方
                new_text = " ".join(split_text[0:sample[4][0]]) + new_headent + " ".join(
                    split_text[sample[4][-1] + 1:sample[6][0]]) \
                           + new_tailent + " ".join(split_text[sample[6][-1] + 1:len(split_text)])
            else:
                new_text = " ".join(split_text[0:sample[6][0]]) + new_tailent + " ".join(
                    split_text[sample[6][-1] + 1:sample[4][0]]) \
                           + new_headent + " ".join(split_text[sample[4][-1] + 1:len(split_text)])
            #  tokenized_sample这里代表希望将其单词化
            tokenized_sample = {}
            tokenized_sample['relation'] = sample[0] - 1
            tokenized_sample['neg_labels'] = [can_idx - 1 for can_idx in sample[1]]
            # padding='max_length' 和max_length=self.config.max_length 是有一定关联的 这里的padding=256
            tokenized_sample['tokens'] = self.tokenizer.encode(new_text,
                                                               padding='max_length',
                                                               truncation=True,
                                                               max_length=self.config.max_length)
            if self.config.task=="fewrel":
                tokenized_sample['rel_des_emb']=self.tokenizer.encode(self.rel_emb[tokenized_sample['relation']],
                                                                  padding='max_length',
                                                                  truncation=True,
                                                                  max_length=self.config.max_length)
                tokenized_sample["rel_des_mask"] = self.rel_mask[tokenized_sample['relation']]
                read_data[tokenized_sample['relation']].append(tokenized_sample)
            else:
                tokenized_sample['rel_des_emb'] = self.tokenizer.encode(self.rel_emb[tokenized_sample['relation']],
                                                                        padding='max_length',
                                                                        truncation=True,
                                                                        max_length=self.config.max_length)
                tokenized_sample["rel_des_mask"] = self.rel_mask[tokenized_sample['relation']]
                read_data[tokenized_sample['relation']].append(tokenized_sample)
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        read_data_query = [[] for i in range(self.config.num_of_relation)]
        read_data_1 = [[] for i in range(self.config.num_of_relation)]
        for i in range(self.config.num_of_relation):
            read_data_query[i].append(random.choice(read_data[i]))
            for x in read_data[i]:
                if x != read_data_query[i][0]:
                    read_data_1[i].append(x)
        for i in range(self.config.num_of_relation):
            for j in read_data_1[i]:
                j["tokens_query"] = read_data_query[i][0]["tokens"]
                j["mask_query"] = read_data_query[i][0]["tokens"]
        return read_data_1
    def load_data_test(self, file,seed):
        # 这里的处理方式相对于之前emnlp22 简单
        #  这里的load_data 居然没有使用 dataloader 这里的速度会不会很慢 没有进行批量提取
        samples = []
        with open(file,encoding='utf-8') as file_in:
            for line in file_in:
                # 转义符 \t 代表一个水平制表符（Horizontal Tab），
                # 通常用于在文本中创建制表位，类似于在键盘上按下“Tab”键所产生的效果
                # 这里也可以看出来有人对fewRel 进行了处理，每一个数据中间的并非用空格隔开的 而是用'\t'分开的
                # 举例：['25', '6 14 29 19 9 61 49 13 21 40',
                #       "a disciple of paul gauguin and friend of paul sérusier ,
                #       he belonged to the circle of artists known as ' les nabis . '",
                #       'paul sérusier', '8 9', 'les nabis', '21 22', 'Q326606', 'Q503708']
                items = line.strip().split('\t')
                # item[0]='25'
                if (len(items[0]) > 0):
                    #  当前句子的关系id或者关系索引
                    relation_ix = int(items[0])
                    # 代表后续有东西的话
                    if items[1] != 'noNegativeAnswer':
                        # candidata_ixs 这里就是10各类别的id
                        # 候选索引
                        candidate_ixs = [int(ix) for ix in items[1].split()]
                        # 将整个样本句子拿出来
                        #  这里为啥没有以.对句子进行划分
                        sentence = items[2].split('\n')[0]
                        # 头部实体
                        headent = items[3]
                        # headidex 头部实体的索引
                        headidx = [int(ix) for ix in items[4].split()]
                        # 尾实体
                        tailent = items[5]
                        # 尾实体的id
                        tailidx = [int(ix) for ix in items[6].split()]
                        # 头尾实体的id
                        headid = items[7]
                        tailid = items[8]
                        #  把数据进行处理之后再重新整理
                        samples.append(
                            [relation_ix, candidate_ixs, sentence, headent, headidx, tailent, tailidx,
                             headid, tailid])
        # 创建了含有80个关系的列表,并且每个元素同样都是列表
        read_data = [[] for i in range(self.config.num_of_relation)]

        for sample in samples:
            #  单独拿出来每个句子,进行简单预处理
            text = sample[2]  # text=sentcence
            #  这里的作用是将text 变成一个一个数据
            #  举例:['a', 'disciple', 'of', 'paul', 'gauguin', 'and', 'friend', 'of', 'paul']
            split_text = text.split(" ")
            # 在头实体和尾实体前面加入[E11]和[E12] [E21]和[E22]
            new_headent = ' [E11] ' + sample[3] + ' [E12] '
            new_tailent = ' [E21] ' + sample[5] + ' [E22] '

            if sample[4][0] < sample[6][0]:
                #  在头实体前面加入前半句话 ，在头实体后面加入后半句话，当前到为实体之前 之后再加改好的尾实体和后面的话
                #  其实就是将改好的尾实体和头实体加入到他该到的地方
                new_text = " ".join(split_text[0:sample[4][0]]) + new_headent + " ".join(
                    split_text[sample[4][-1] + 1:sample[6][0]]) \
                           + new_tailent + " ".join(split_text[sample[6][-1] + 1:len(split_text)])
            else:
                new_text = " ".join(split_text[0:sample[6][0]]) + new_tailent + " ".join(
                    split_text[sample[6][-1] + 1:sample[4][0]]) \
                           + new_headent + " ".join(split_text[sample[4][-1] + 1:len(split_text)])
            #  tokenized_sample这里代表希望将其单词化
            tokenized_sample = {}
            tokenized_sample['relation'] = sample[0] - 1
            tokenized_sample['neg_labels'] = [can_idx - 1 for can_idx in sample[1]]
            # padding='max_length' 和max_length=self.config.max_length 是有一定关联的 这里的padding=256
            tokenized_sample['tokens'] = self.tokenizer.encode(new_text,
                                                               padding='max_length',
                                                               truncation=True,
                                                               max_length=self.config.max_length)
            read_data[tokenized_sample['relation']].append(tokenized_sample)
        return read_data
    def get_seed(self,seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        # 这里和set_seed 一个操作
        shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(shuffle_index_old)
        shuffle_index_old = np.argsort(shuffle_index_old)
        shuffle_index = np.insert(shuffle_index_old, 0, self.task_length - 1)
        return  shuffle_index
    def get_cluster_label(self,rel_index,rel_cluster_labels):
        rel_index = np.load(rel_index)
        rel_cluster_label = np.load(rel_cluster_labels)
        cluster_to_labels={}
        # {0: [79, 3, 12, 23, 28, 1, 55, 60, 41, 2], 7: [24, 63, 5, 77, 64, 48, 11, 20, 13, 16]}
        for index, i in enumerate(rel_index):
            if rel_cluster_label[index] in cluster_to_labels.keys():
                cluster_to_labels[rel_cluster_label[index]].append(i - 1)
            else:
                cluster_to_labels[rel_cluster_label[index]] = [i - 1]
        return cluster_to_labels

    # Iterators
    def __iter__(self):
        return self


    def __next__(self):
        if self.task == self.task_length:
            self.task = 0
            raise StopIteration()
        # # self.cluster_to_labels ={0: [79, 3, 12, 23, 28, 1, 55, 60, 41, 2], 7: [24, 63, 5, 77, 64, 48, 11, 20, 13, 16]}
        #  这里的index 是这一轮的10个类别
        indexs = self.cluster_to_labels[self.shuffle_index[self.task]]  # 每个任务出现的id
        print(f"the {self.task}th task:{indexs}")
        self.task += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        for index in indexs:

            current_relations.append(self.rel[index])


            self.seen_relations.append(self.rel[index])


            cur_training_data[self.rel[index]] = self.training_data[index]
            cur_valid_data[self.rel[index]] = self.valid_data[index]
            cur_test_data[self.rel[index]] = self.test_data[index]
            #  把所有的测试数据全部存储下来
            self.history_test_data[self.rel[index]] = self.test_data[index]

        return cur_training_data, cur_valid_data, cur_test_data, current_relations, \
               self.history_test_data, self.seen_relations

