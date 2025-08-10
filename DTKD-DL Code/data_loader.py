# Author:徐睿峰
# -*- codeing: utf-8 -*-
# @time ：2024/4/23 15:25
# @Author :xuruifeng
# @Site : 
# @file : data_loader.py
# @Sofeware : PyCharm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# dataset = data_set(data, config)
# data = relation_dataset = training_data[relation]
# relation_dataset =[{关系：11,[11个类别还是10个来],[进行单词化编码之后的数据]},.....]
class data_set(Dataset):

    def __init__(self, data,config=None):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):

        label = torch.tensor([item['relation'] for item in data])
        neg_labels = [torch.tensor(item['neg_labels']) for item in data]
        tokens = [torch.tensor(item['tokens']) for item in data]
        if self.config.task == "fewrel":
            rel_des_emb =[torch.tensor(item['rel_des_emb']) for item in data]
            rel_des_mask =[torch.tensor(item['rel_des_mask']) for item in data]
            tokens_query=[torch.tensor(item['tokens_query']) for item in data]
            mask_query = [torch.tensor(item['mask_query']) for item in data]
            return (
                label,
                neg_labels,
                tokens,
                rel_des_emb,
                rel_des_mask,
                tokens_query,
                mask_query
                )
        else:
            rel_des_emb = [torch.tensor(item['rel_des_emb']) for item in data]
            rel_des_mask = [torch.tensor(item['rel_des_mask']) for item in data]
            tokens_query = [torch.tensor(item['tokens_query']) for item in data]
            mask_query = [torch.tensor(item['mask_query']) for item in data]
            return (
                label,
                neg_labels,
                tokens,
                rel_des_emb,
                rel_des_mask,
                tokens_query,
                mask_query
            )
# proto, _ = get_proto(config, encoder, dropout_layer, training_data[relation])
# data_loader = get_data_loader(config, relation_dataset, shuffle=False, drop_last=False, batch_size=1)
# data = relation_dataset = training_data[relation]
def get_data_loader(config, data, shuffle = False, drop_last = False, batch_size = None):
    # len(training_data[relation]) 更像一个非法标志
    dataset = data_set(data, config)
    # batch_size=1
    if batch_size == None:
        batch_size = min(config.batch_size_per_step, len(data))
    else:
        batch_size = min(batch_size, len(data))
    # dataset= tensor类型直接就可以
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader

class data_set1(Dataset):

    def __init__(self, data,config=None):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):

        label = torch.tensor([item['relation'] for item in data])
        neg_labels = [torch.tensor(item['neg_labels']) for item in data]
        tokens = [torch.tensor(item['tokens']) for item in data]
        return (
            label,
            neg_labels,
            tokens,
        )
def get_data_loader1(config, data, shuffle = False, drop_last = False, batch_size = None):
    # len(training_data[relation]) 更像一个非法标志
    dataset = data_set1(data, config)
    # batch_size=1
    if batch_size == None:
        batch_size = min(config.batch_size_per_step, len(data))
    else:
        batch_size = min(batch_size, len(data))
    # dataset= tensor类型直接就可以
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader