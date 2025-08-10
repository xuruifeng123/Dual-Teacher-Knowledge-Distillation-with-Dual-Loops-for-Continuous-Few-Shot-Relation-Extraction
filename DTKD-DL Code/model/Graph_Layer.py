# Author:徐睿峰
# -*- codeing: utf-8 -*-
# @time ：2024/4/16 16:09
# @Author :xuruifeng
# @Site : 
# @file : Graph_Layer.py
# @Sofeware : PyCharm
import torch
from torch import nn
from torch.nn import functional as F
from  model.base_model import base_model


class BiAttention(nn.Module):
    def __init__(self, k_dim, n_hid, temperature=1.0, dropout=0.1):
        super().__init__()

        self.bilinear = nn.Bilinear(k_dim, k_dim, n_hid)
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, mask=None):
        attn = self.bilinear(inp, inp).mean(-1)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1).unsqueeze(-2)
        output = torch.matmul(attn, inp).sum(-2)
        return output, attn

class graph_layer(base_model):
    def __init__(self,config):
        super(graph_layer, self).__init__()
        self.config = config
        self.linear_forward=nn.Linear(config.encoder_output_size*2,config.encoder_output_size*2)
        self.linear_forward_1=nn.Linear(config.encoder_output_size*2,config.encoder_output_size)
        self.linear_transform = nn.Linear(config.encoder_output_size* 4, config.encoder_output_size, bias=True)
        self.graph = BiAttention(config.encoder_output_size*2, 32)
        self.drop = nn.Dropout(config.drop_out)
        self.layer_normalization = nn.LayerNorm([config.encoder_output_size])
    def forward(self,support_emb=None,rel_rep=None):

            support = support_emb
            support3 = self.linear_forward_1(support)

            support1 = support.view(-1, 1, 1, self.config.encoder_output_size * 2)
            B = support.size(0)
            if self.config.task == "fewrel":
                support = torch.cat([support1, rel_rep], dim=-2)
                support1 = support.mean(2)
                inp = self.linear_forward(support).view(B * 1, 1+1, -1)
                support2, _ = self.graph(inp)
                support2 = support2.view(B, 1, -1)
                support1=support1.view(-1, 1,self.config.encoder_output_size * 2)
                final_support = torch.cat([support2, support1], dim=-1)
                final_support=final_support.view(B,-1)
                final_support=self.linear_transform(final_support)



            else:
                support = torch.cat([support1, rel_rep], dim=-2)
                support1 = support.mean(2)
                inp = self.linear_forward(support).view(B * 1, 1+1, -1)
                support2, _ = self.graph(inp)
                support2 = support2.view(B, 1, -1)
                support1 = support1.view(-1, 1, self.config.encoder_output_size * 2)
                final_support = torch.cat([support2, support1], dim=-1)
                final_support = final_support.view(B, -1)
                final_support = self.linear_transform(final_support)

            return  final_support,support3



