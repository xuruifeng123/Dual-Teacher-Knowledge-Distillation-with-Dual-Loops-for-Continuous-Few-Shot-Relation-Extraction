# @Descripttion ：
# -*- codeing: utf-8 -*-
# @time ：2025/6/19 17:08
# @Author : Ruifeng Xu
# @Site : 
# @file : data_transforms.py
# @Sofeware : PyCharm


import numpy as np
from base_function import get_cluster_label,save_json_file,read_json_file,read_relations,transform_data_train,get_seed,transform_data_test,transform_data
import random
def trainsforms_data_train(config,rel_cluster_labels,rel_index,file_path,category,relation_name,seed,task):
    task_length = 8
    # rel_cluster_labels = "data/fewrel_train/CFRLdata_10_100_10_5/rel_cluster_label_0.npy"
    # rel_index = "data/fewrel_train/rel_index.npy"
    cluster_to_labels = get_cluster_label(rel_index=rel_index, rel_cluster_labels=rel_cluster_labels)

    print(cluster_to_labels)

    for i in range(len(cluster_to_labels)):
        shuffle_index=get_seed(seed)
        indexs =cluster_to_labels[shuffle_index[i]]  # 每个任务出现的id
        print(f"the {i}th task:{indexs}")
        # read_data = read_json_file(config=config, file_path="E:\RE\LLM\data\\fewrel_train\\CFRLdata_10_100_10_5\\train_0.txt")
        # _, rel_id = read_relations("data/fewrel_train/relation_name.txt")
        read_data = read_json_file(config=config, file_path=file_path)
        _, rel_id = read_relations(relation_name)

        all_data = transform_data(read_data, rel_id,indexs,task)
        save_json_file(all_data, output_file=f"/home/xurf23/xurf_project/LLaMA-Factory-main/LLaMA-Factory-main/data/{task}_{category}_5_{i + 1}.json")

def trainsforms_data_text_1(config,rel_cluster_labels,rel_index,file_path,category,relation_name,seed,task):
    task_length = 8
    # rel_cluster_labels = "data/fewrel_train/CFRLdata_10_100_10_5/rel_cluster_label_0.npy"
    # rel_index = "data/fewrel_train/rel_index.npy"
    cluster_to_labels = get_cluster_label(rel_index=rel_index, rel_cluster_labels=rel_cluster_labels)

    print(cluster_to_labels)

    for i in range(len(cluster_to_labels)):
        shuffle_index=get_seed(seed)
        indexs =cluster_to_labels[shuffle_index[i]]  # 每个任务出现的id
        print(f"the {i}th task:{indexs}")
        # read_data = read_json_file(config=config, file_path="E:\RE\LLM\data\\fewrel_train\\CFRLdata_10_100_10_5\\train_0.txt")
        # _, rel_id = read_relations("data/fewrel_train/relation_name.txt")
        read_data = read_json_file(config=config, file_path=file_path)
        _, rel_id = read_relations(relation_name)

        all_data = transform_data(read_data, rel_id,indexs)
        save_json_file(all_data, output_file=f"/home/xurf23/xurf_project/LLaMA-Factory-main/LLaMA-Factory-main/data/{task}_{category}_5_{i + 1}.json")
        
def trainsforms_data_text(config,rel_cluster_labels,rel_index,file_path,category,relation_name,seed,task):
    task_length = 8
    # rel_cluster_labels = "data/fewrel_train/CFRLdata_10_100_10_5/rel_cluster_label_0.npy"
    # rel_index = "data/fewrel_train/rel_index.npy"
    cluster_to_labels = get_cluster_label(rel_index=rel_index, rel_cluster_labels=rel_cluster_labels)

    print(cluster_to_labels)

    for i in range(len(cluster_to_labels)):
        shuffle_index=get_seed(seed)
        indexs=[]
        for j in range (i+1):
            index =cluster_to_labels[shuffle_index[j]]
            for k in index:
                indexs.append(k)
        print(f"the {i}th task:{indexs}")  # 每个任务出现的id
        # read_data = read_json_file(config=config, file_path="E:\RE\LLM\data\\fewrel_train\\CFRLdata_10_100_10_5\\train_0.txt")
        # _, rel_id = read_relations("data/fewrel_train/relation_name.txt")
        read_data = read_json_file(config=config, file_path=file_path)
        _, rel_id = read_relations(relation_name)

        all_data = transform_data_test(read_data, rel_id,indexs,task)
        # save_json_file(all_data, output_file=f"/home/xurf23/xurf_project/LLaMA-Factory-main/LLaMA-Factory-main/data/fewrel_{category}_5_{i + 1}.json")
        save_json_file(all_data, output_file=f"/home/xurf23/xurf_project/LLaMA-Factory-main/LLaMA-Factory-main/data/{task}_{category}_5_{i + 1}.json")
