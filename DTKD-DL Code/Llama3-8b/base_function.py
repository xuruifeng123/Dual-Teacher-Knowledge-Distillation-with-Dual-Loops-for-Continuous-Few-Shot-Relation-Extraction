# @Descripttion ：
# -*- codeing: utf-8 -*-
# @time ：2025/6/19 17:15
# @Author : Ruifeng Xu
# @Site : 
# @file : base_function.py
# @Sofeware : PyCharm
import random
import numpy as np
import json
import os
import shutil
from transformers import pipeline
# from ollama import Client
from tqdm import tqdm
import requests
import re
import sys
import time
from loguru import logger

from transformers import AutoModelForCausalLM, pipeline



def get_cluster_label(rel_index, rel_cluster_labels):
    rel_index = np.load(rel_index)
    rel_cluster_label = np.load(rel_cluster_labels)
    cluster_to_labels = {}
    # {0: [79, 3, 12, 23, 28, 1, 55, 60, 41, 2], 7: [24, 63, 5, 77, 64, 48, 11, 20, 13, 16]}
    for index, i in enumerate(rel_index):
        if rel_cluster_label[index] in cluster_to_labels.keys():
            cluster_to_labels[rel_cluster_label[index]].append(i - 1)
        else:
            cluster_to_labels[rel_cluster_label[index]] = [i - 1]
    return cluster_to_labels


def  save_json_file(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        # 写入JSON数据
        json.dump(data, file, ensure_ascii=False, indent=4)


def read_json_file(config,file_path):
    samples = []
    with open(file_path, encoding='utf-8') as file_in:
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
    read_data = [[] for i in range(config)]
    for sample in samples:
        tokenized_sample = {}
        tokenized_sample['relation'] = sample[0] - 1
        # padding='max_length' 和max_length=self.config.max_length 是有一定关联的 这里的padding=256
        tokenized_sample['sentence'] = sample[2]
        tokenized_sample['head'] = sample[3]
        tokenized_sample['tail'] = sample[5]
        read_data[tokenized_sample['relation']].append(tokenized_sample)
    return read_data


def read_relations(file):
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




def transform_data_train(dataload,rel_id,cluster_to_labels,task):
    all_data=[]
    instruction= 'You are a classifier.\n ' \
                            'I will provide a text and two entities.\n ' \
                            'Your goal is to determine the relationship between these two entities and ' \
                            'select one category ID that best represents the relationship between these two entities.\n ' \
                            'The categories are as follows:\n\n' + \
                            '\n'.join(f"{value}: {key}" for key,value in rel_id.items())+ \
                            "\n\n"+\
                            "It is important to note that when outputting, " \
                            "you do not need to output the questions I gave you, " \
                            "just the category ID, without any other information."+ \
                            "\nNow I will provide the sentence and entities in the format of the example input. "

    for key,value in rel_id.items():
        if value in cluster_to_labels:
            for data in dataload[value]:

                new_data={}
                new_data["instruction"]=instruction
                new_data["input"]=f'sentence:{data["sentence"]}'+'\n'+f'head entity:{data["head"]}'+'\n'+f'tail entity2:{data["tail"]}'
                new_data["output"]=f"{data['relation']}"
                all_data.append(new_data)

    return all_data

def transform_data(dataload,rel_id,cluster_to_labels,task):
    all_data=[]
    instruction= 'You are a classifier.\n ' \
                            'I will provide a text and two entities.\n ' \
                            'Your goal is to determine the relationship between these two entities and ' \
                            'select one category ID that best represents the relationship between these two entities.\n ' \
                            'The categories are as follows:\n\n' + \
                            '\n'.join(f"{value}: {key}" for key,value in rel_id.items())+ \
                            "\n\n"+\
                            "It is important to note that when outputting, " \
                            "you do not need to output the questions I gave you, " \
                            "just the category ID, without any other information."+ \
                            "\nNow I will provide the sentence and entities in the format of the example input. "

    for key,value in rel_id.items():
        if value in cluster_to_labels:
            for data in dataload[value]:

                new_data={}
                new_data["instruction"]=instruction
                new_data["input"]=f'sentence:{data["sentence"]}'+'\n'+f'head entity:{data["head"]}'+'\n'+f'tail entity2:{data["tail"]}'
                new_data["output"]=f"{data['relation']}"
                all_data.append(new_data)

    return all_data

def transform_data_test(dataload,rel_id,cluster_to_labels,task):
    all_data=[]
    if task =="fewrel":
        instruction= 'You are a classifier.\n ' \
                                'I will provide a text and two entities.\n ' \
                                'Your goal is to determine the relationship between these two entities and ' \
                                'select one category ID that best represents the relationship between these two entities.\n ' \
                                'The categories are as follows:\n\n' + \
                                '\n'.join(f"{value}: {key}" for key,value in rel_id.items())+ \
                                "\nHere are examples:\n" + "### Example input:\n" + "sentence:" + "jan černý ( 4 march 1874 , uherský ostroh , moravia , austria - hungary – 10 april 1959 , uherský ostroh , czechoslovakia ) was a czechoslovak civil servant and politician ." + \
                                "\nhead entity:" + f"czechoslovakia" + "\ntail entity:" + f"jan černý" + \
                                "\n### Example output:" + "\n"+"5" + "\n\n"+\
                                "It is important to note that when outputting, " \
                                "you do not need to output the questions I gave you, " \
                                "just the category ID, without any other information."+ \
                                "\nNow I will provide the sentence and entities in the format of the example input. "
    else:
        instruction= 'You are a classifier.\n ' \
                                'I will provide a text and two entities.\n ' \
                                'Your goal is to determine the relationship between these two entities and ' \
                                'select one category ID that best represents the relationship between these two entities.\n ' \
                                'The categories are as follows:\n\n' + \
                                '\n'.join(f"{value}: {key}" for key,value in rel_id.items())+ \
                                "\nHere are examples:\n" + "### Example input:\n" + "sentence:" + "a naval researcher , professor li jie , told the state-run china daily newspaper wednesday that dispatching of china 's navy would increase its prominence on the world stage ." + \
                                "\nhead entity:" + f"li jie" + "\ntail entity:" + f"china" + \
                                "\n### Example output:" + "6" + "\n\n"+\
                                "It is important to note that when outputting, " \
                                "you do not need to output the questions I gave you, " \
                                "just the category ID, without any other information."+ \
                                "\nNow I will provide the sentence and entities in the format of the example input. "
    for key,value in rel_id.items():
        if value in cluster_to_labels:
            for data in dataload[value]:

                new_data={}
                new_data["instruction"]=instruction
                new_data["input"]=f'sentence:{data["sentence"]}'+'\n'+f'head entity:{data["head"]}'+'\n'+f'tail entity2:{data["tail"]}'
                new_data["output"]=f"{data['relation']}"
                all_data.append(new_data)

    return all_data
def get_seed(seed):
    random.seed(seed)
    shuffle_index_old = list(range(8 - 1))
    random.shuffle(shuffle_index_old)
    shuffle_index_old = np.argsort(shuffle_index_old)
    shuffle_index = np.insert(shuffle_index_old, 0, 8 - 1)
    return shuffle_index

def save_loss_png(source_folder, destination_folder,task,number,task_length):
    # source_folder = "/home/xurf23/xurf_project/LLM/save_lora"  # 替换为你的源文件夹路径
    # destination_folder = "/home/xurf23/xurf_project/LLM/fewrel_loss"  # 替换为你的目标文件夹路径

    # 创建目标文件夹（如果不存在）
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历源文件夹中的所有文件
    for index, filename in enumerate(os.listdir(source_folder)):
        # 检查文件是否是图片（可以根据需要添加更多图片扩展名）
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            # 构建完整的文件路径
            source_path = os.path.join(source_folder, filename)
            # 构建新的文件名
            new_filename = f"image_{task}_5_{number}_{task_length}.png"
            destination_path = os.path.join(destination_folder, new_filename)

            # 复制文件到目标文件夹并重命名
            shutil.copy(source_path, destination_path)
            logger.info(f"已复制并重命名: {filename} -> {new_filename}")

class data_sampler(object):
    def __init__(self, config,seed,test_file,relation_file,rel_index, rel_cluster_labels):
        self.config = config
        self.test_data = self.load_data_test(test_file)
        self.task = 0
        self.task_length=8
        self.cluster_to_labels=get_cluster_label(rel_index=rel_index, rel_cluster_labels=rel_cluster_labels)
        self.shuffle_index=get_seed(seed=seed)
        self.rel, self.rel_id=read_relations(relation_file)
        self.seen_relations = []
    def load_data_test(self, file):
        # 这里的处理方式相对于之前emnlp22 简单
        #  这里的load_data 居然没有使用 dataloader 这里的速度会不会很慢 没有进行批量提取
        samples = []
        with open(file, encoding='utf-8') as file_in:
            for line in file_in:
                items = line.strip().split('\t')
                if (len(items[0]) > 0):
                    relation_ix = int(items[0])
                    if items[1] != 'noNegativeAnswer':
                        candidate_ixs = [int(ix) for ix in items[1].split()]
                        sentence = items[2].split('\n')[0]
                        headent = items[3]
                        headidx = [int(ix) for ix in items[4].split()]
                        tailent = items[5]
                        tailidx = [int(ix) for ix in items[6].split()]
                        headid = items[7]
                        tailid = items[8]
                        samples.append(
                            [relation_ix, candidate_ixs, sentence, headent, headidx, tailent, tailidx,
                             headid, tailid])

        read_data = [[] for i in range(self.config)]

        for sample in samples:
            tokenized_sample = {}
            tokenized_sample['relation'] = sample[0] - 1
            tokenized_sample['neg_labels'] = [can_idx - 1 for can_idx in sample[1]]
            # padding='max_length' 和max_length=self.config.max_length 是有一定关联的 这里的padding=256
            tokenized_sample['sentence'] = sample[2]
            tokenized_sample['head'] = sample[3]
            tokenized_sample['tail'] = sample[5]
            read_data[tokenized_sample['relation']].append(tokenized_sample)
        return read_data

    def __iter__(self):
        return self
    def __next__(self):
        if self.task == self.task_length:
            self.task = 0
            raise StopIteration()
            # # self.cluster_to_labels ={0: [79, 3, 12, 23, 28, 1, 55, 60, 41, 2], 7: [24, 63, 5, 77, 64, 48, 11, 20, 13, 16]}
            #  这里的index 是这一轮的10个类别
        indexs = self.cluster_to_labels[self.shuffle_index[self.task]]  # 每个任务出现的id
        logger.info(f"the {self.task}th task:{indexs}")
        self.task += 1

        current_relations = []
        cur_test_data = {}

        for index in indexs:
            current_relations.append(self.rel[index])
            self.seen_relations.append(self.rel[index])

            cur_test_data[self.rel[index]] = self.test_data[index]
            #  把所有的测试数据全部存储下来

        return  cur_test_data, current_relations,self.seen_relations\



# def eval_LLM(model_id, test_data, instruction):
#     classifier = pipeline(
#         "text-generation",
#         model=model_id,
#         device_map=7,
#         batch_size=1

#     )
#     correct = 0
#     n = len(test_data)
#     i = 0
#     for data in test_data:
#         sentence = data["sentence"]
#         entity_1 = data["head"]
#         entity_2 = data["tail"]
#         label = data["relation"]
#         prompt = f"{instruction}\n\n"
#         prompt += f"### input:\n"
#         prompt += f"sentence:{sentence}\n"
#         prompt += f"head entity:{entity_1}\n"
#         prompt += f"tail entity:{entity_2}\n\n"

#         prompt += "### output\n"
#         i += 1
#         output = classifier(prompt, temperature=0.2, max_new_tokens=10, truncation=True, num_return_sequences=1)
#         predicted_label = output[0]['generated_text']
#         pattern = r'### output\s*.*?(\d+)'
#         match = re.search(pattern, predicted_label, re.DOTALL)
#         if match:
#             number = match.group(1).strip()
#             predicted_label = int(number)
#             sys.stdout.write(f"predicted label :{predicted_label},true label:{label}")
#             sys.stdout.flush()
#             if predicted_label == label:
#                 correct += 1
#             if i % 20 == 0:
#                 # 检查预测的标签是否为有效的 ID，并映射回对应的类别 ID
#                 logger.info(f"accuracy:{(correct / n) * 100}")
#         else:
#             pass

#     return correct / n

def change_sh(file_sh,task,step):
    with open(file=file_sh, mode="r") as file:
        line = file.readlines()
        line[6] = f"    --dataset {task}_train_5_{step} \\\n"
    with open(file=file_sh, mode="w") as file:
        file.writelines(line)

def eval_LLM(model_id, test_data, instruction):
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    
    # accelerator = Accelerator()
    

    # model = AutoModelForCausalLM.from_pretrained(model_id)
    # model = accelerator.prepare(model)
    classifier = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        # device=7,
        # device_map="auto",
        # device_map=accelerator.device,
        batch_size=1

    )
    correct = 0
    n = len(test_data)
    i = 0
    for data in test_data:
        sentence = data["sentence"]
        entity_1 = data["head"]
        entity_2 = data["tail"]
        label = data["relation"]
        prompt = f"{instruction}\n\n"
        prompt += f"### input:\n"
        prompt += f"sentence:{sentence}\n"
        prompt += f"head entity:{entity_1}\n"
        prompt += f"tail entity:{entity_2}\n\n"

        prompt += "### output\n"
        i += 1
        output = classifier(prompt, temperature=0.2, max_new_tokens=10, truncation=True, num_return_sequences=1,max_len=512)
        predicted_label = output[0]['generated_text']
        pattern = r'### output\s*.*?(\d+)'
        match = re.search(pattern, predicted_label, re.DOTALL)
        if match:
            number = match.group(1).strip()
            predicted_label = int(number)
            sys.stdout.write(f"predicted label :{predicted_label},true label:{label}")
            sys.stdout.flush()
            if predicted_label == label:
                correct += 1
            if i % 20 == 0:
                # 检查预测的标签是否为有效的 ID，并映射回对应的类别 ID
                logger.info(f"accuracy:{(correct / n) * 100}")
        else:
            pass

    return correct / n

def change_sh_text(file_sh,task,step):
    with open(file=file_sh, mode="r") as file:
        line = file.readlines()
        line[7] = f"    --eval_dataset {task}_test_5_{step} \\\n"
    with open(file=file_sh, mode="w") as file:
        file.writelines(line)

def eval_accuracy(path):
    extracted_data = []
    correct = 0
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行JSON数据
            item = json.loads(line)
            label = item.get('label')
            predict = item.get('predict', '')

            # 使用正则表达式提取predict中的数字
            match = re.search(r'(\d+)', predict)
            if match:
                predict_number = match.group(1)
                extracted_data.append({
                'label': label,
                'predict': predict_number[:2]
            })
            else:
                predict_number ="-99" # 如果没有找到数字，则设置为None或默认值
                extracted_data.append({
                'label': label,
                'predict': predict_number
            })

           

        # 打印提取的结果
        for entry in extracted_data:
           if entry['label'] == entry['predict'] or  entry['label'] == entry['predict'][0]:
               correct += 1
           else:
               pass

        return correct/len(extracted_data)