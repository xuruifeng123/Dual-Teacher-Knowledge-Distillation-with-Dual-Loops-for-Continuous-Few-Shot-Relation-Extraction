import argparse
import random
from sampler import data_sampler
import torch
from transformers import  BertTokenizer
from model.bert_encoder import Bert_Encoder
from model.dropout_layer import Dropout_Layer
from model.classifier import Softmax_Layer
from data_loader import get_data_loader,get_data_loader1
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
import collections
from copy import deepcopy
import os
alpha=0.3
import logging
# 配置logging模块

def train_simple_model(config, encoder,encoder1, descencoder,graph_layer,dropout_layer, classifier,classifier1, training_data, epochs,
                       map_relid2tempid):
    data_loader = get_data_loader(config, training_data, shuffle=True)
    encoder.train()
    encoder1.train()
    dropout_layer.train()
    graph_layer.train()
    classifier.train()
    classifier1.train()
    descencoder.train()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': encoder1.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001},
        {'params': classifier1.parameters(), 'lr': 0.001},
        {'params': descencoder.parameters(), 'lr': 0.00001},
        {'params': graph_layer.parameters(), 'lr': 0.00001},
    ])
    for epoch_i in range(epochs):
        losses = []
        for step, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            labels, _, tokens, rel_des_emb, rel_des_mask, _, _ = batch_data

            labels = labels.to(config.device)
            labels = [map_relid2tempid[x.item()] for x in labels]
            labels = torch.tensor(labels).to(config.device)

            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            rel_des_emb = torch.stack([x.to(config.device) for x in rel_des_emb], dim=0)
            rel_des_mask = torch.stack([x.to(config.device) for x in rel_des_mask], dim=0)
            rel_gol, rel_loc = descencoder(rel_des_emb, rel_des_mask)
            rel_loc = torch.mean(rel_loc[:, 1:, :], 1)
            rel_rep = torch.cat((rel_gol, rel_loc), -1).view(-1, 1, 1, rel_gol.shape[1] * 2)

            reps = encoder(tokens)
            reps1=encoder1(tokens)

            reps, _ = dropout_layer(reps)
            reps1,_=graph_layer(reps1,rel_rep)

            logits1 = classifier(reps)
            logits2 = classifier1(reps1)

            loss1 = criterion1(logits1, labels)
            loss2 = criterion2(logits2, labels)
            loss=loss1+loss2
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")


def evaluate_strict_model(config, encoder,dropout_layer, classifier, test_data, seen_relations, map_relid2tempid,rel2id):
    data_loader = get_data_loader1(config, test_data, batch_size=1)
    encoder.eval()
    dropout_layer.eval()
    classifier.eval()
    n = len(test_data)

    correct = 0
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens, = batch_data
        labels = labels.to(config.device)
        labels = [map_relid2tempid[x.item()] for x in labels]
        labels = torch.tensor(labels).to(config.device)

        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        reps = encoder(tokens)
        reps, _ = dropout_layer(reps)
        logits = classifier(reps)

        seen_relation_ids = [rel2id[relation] for relation in seen_relations]
        seen_relation_ids = [map_relid2tempid[relation] for relation in seen_relation_ids]
        seen_sim = logits[:,seen_relation_ids].cpu().data.numpy()
        max_smi = np.max(seen_sim,axis=1)

        label_smi = logits[:,labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1

    return correct/n




def train_mem_model(config, encoder, drop_layer, classifier, training_data, epochs,
                    map_relid2tempid,new_relation_data,prev_encoder,prev_graph_encoder,prev_graph_decencoder ,
                    prev_graph_layer,prev_drop_layer,prev_classifier,
                    prev_graph_classifier,prev_relation_index,state,relation_index,encoder_FOMAML,droplayer_FOMAML,classifier_FOMAML):
    data_loader = get_data_loader(config, training_data,shuffle=True)
    # expanded_train_data_for_initial, expanded_prev_samples 增广后的数据
    encoder.train()
    drop_layer.train()
    classifier.train()
    encoder_FOMAML.train()
    droplayer_FOMAML.train()
    classifier_FOMAML.train()
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': encoder.parameters(), 'lr': 0.0001},
        {'params': drop_layer.parameters(), 'lr': 0.0001},
        {'params': classifier.parameters(), 'lr': 0.01}
    ])
    META_optimizer = optim.Adam([
        {'params': encoder_FOMAML.parameters(), 'lr': 0.00001},
        {'params': droplayer_FOMAML.parameters(), 'lr': 0.00001},
        {'params': classifier_FOMAML.parameters(), 'lr': 0.001}
    ])
    distill_criterion = nn.CosineEmbeddingLoss()
    T = state["T"]
    alpha=state["W"]
    LCT=None
    for epoch_i in range(epochs):

        losses = []
        LTes=[]
        for step, batch_data in enumerate(data_loader):
            LT=torch.tensor(0.0).to(config.device)
            labels, _, tokens,rel_des_emb, rel_des_mask,tokens_query,mask_query= batch_data
            optimizer.zero_grad()
            logits_all = []
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)

            labels = labels.to(config.device)
            origin_labels = labels[:]
            labels = [map_relid2tempid[x.item()] for x in labels]
            labels = torch.tensor(labels).to(config.device)
            reps = encoder(tokens)


            normalized_reps_emb = F.normalize(reps.view(-1, reps.size()[1]), p=2, dim=1)
            outputs, _= drop_layer(reps)
            if prev_drop_layer is not None:
                reps1=prev_graph_encoder(tokens)
                prev_outputs, _= prev_drop_layer(reps)
                positives, negatives = construct_hard_triplets(prev_outputs, origin_labels, new_relation_data)
            else:
                positives, negatives = construct_hard_triplets(outputs, origin_labels, new_relation_data)
            for _ in range(config.f_pass):
                final_support,_= drop_layer(reps)
                logits = classifier(final_support)
                logits_all.append(logits)


            positives = torch.cat(positives, 0).to(config.device)
            negatives = torch.cat(negatives, 0).to(config.device)
            anchors = outputs
            logits_all = torch.stack(logits_all)
            m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B
            loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))
            loss2 = compute_jsd_loss(logits_all)
            tri_loss = triplet_loss(anchors, positives, negatives)
            loss = loss1 + loss2+ tri_loss
            if prev_encoder is not None:
                prev_reps = prev_encoder(tokens).detach()
                normalized_prev_reps_emb = F.normalize(prev_reps.view(-1, prev_reps.size()[1]), p=2, dim=1)

                feature_distill_loss = distill_criterion(normalized_reps_emb, normalized_prev_reps_emb,
                                                         torch.ones(tokens.size(0)).to(
                                                             config.device))
                loss += feature_distill_loss*0.5

            if prev_drop_layer is not None and prev_classifier is not None:
                prediction_distill_loss_1 = None
                prediction_distill_loss_2 = None
                dropout_output_all = []
                prev_graph_output_all = []
                prev_output_all=[]
                pre_logits_all=[]
                pre_logits_graph_all=[]
                rel_des_emb = torch.stack([x.to(config.device) for x in rel_des_emb], dim=0)
                rel_des_mask = torch.stack([x.to(config.device) for x in rel_des_mask], dim=0)
                rel_gol, rel_loc = prev_graph_decencoder(rel_des_emb, rel_des_mask)
                rel_loc = torch.mean(rel_loc[:, 1:, :], 1)
                rel_rep = torch.cat((rel_gol, rel_loc), -1).view(-1, 1, 1, rel_gol.shape[1] * 2)
                for i in range(config.f_pass):
                    final_support, _= drop_layer(reps)
                    graph_support,_=prev_graph_layer(reps1,rel_rep)
                    final_support1,_=prev_drop_layer(reps)

                    dropout_output_all.append(final_support)
                    prev_graph_output_all.append(graph_support)
                    prev_output_all.append(final_support1)

                    pre_logits_graph = prev_graph_classifier(graph_support).detach()
                    pre_logits_graph_all.append(pre_logits_graph)
                    pre_logits = prev_classifier(final_support1).detach()
                    pre_logits_all.append(pre_logits)

                    pre_logits_graph = F.softmax(pre_logits_graph.index_select(1, prev_relation_index) / T, dim=1)
                    pre_logits = F.softmax(pre_logits.index_select(1, prev_relation_index) / T, dim=1)
                    log_logits = F.log_softmax(logits_all[i].index_select(1, prev_relation_index) / T, dim=1)



                    if i == 0:
                        prediction_distill_loss_1 = -torch.mean(torch.sum(pre_logits_graph * log_logits, dim=1))
                        prediction_distill_loss_2 = -torch.mean(torch.sum(pre_logits * log_logits, dim=1))
                    else:
                        prediction_distill_loss_1 += -torch.mean(torch.sum(pre_logits_graph * log_logits, dim=1))
                        prediction_distill_loss_2 += -torch.mean(torch.sum(pre_logits * log_logits, dim=1))

                prediction_distill_loss_1 /= config.f_pass
                prediction_distill_loss_2 /= config.f_pass
                loss += prediction_distill_loss_1*alpha+(1-alpha)*prediction_distill_loss_2

                pre_logits_graph_all=torch.stack(pre_logits_graph_all)
                pre_logits_all=torch.stack(pre_logits_all)
                mean_logits_graph_all=torch.mean(pre_logits_graph_all,dim=0)
                mean_logits_all=torch.mean(pre_logits_all,dim=0)
                normalized_pre_logits_graph = F.normalize(mean_logits_graph_all.view(-1, mean_logits_graph_all.size()[1]),
                                                p=2, dim=1)
                normalized_pre_logits = F.normalize(mean_logits_all.view(-1, mean_logits_all.size()[1]), p=2,
                                                     dim=1)
                teacher_distill_loss = distill_criterion(normalized_pre_logits_graph, normalized_pre_logits,
                                                        torch.ones(tokens.size(0)).to(
                                                            config.device))
                LT+=teacher_distill_loss

                dropout_output_all = torch.stack(dropout_output_all)
                prev_output_all = torch.stack(prev_output_all)
                mean_dropout_output_all = torch.mean(dropout_output_all, dim=0)
                mean_prev_output_all = torch.mean(prev_output_all,dim=0)
                normalized_output = F.normalize(mean_dropout_output_all.view(-1, mean_dropout_output_all.size()[1]), p=2, dim=1)
                normalized_prev_output = F.normalize(mean_prev_output_all.view(-1, mean_prev_output_all.size()[1]), p=2, dim=1)
                hidden_distill_loss = distill_criterion(normalized_output, normalized_prev_output,
                                                         torch.ones(tokens.size(0)).to(
                                                             config.device))
                loss += hidden_distill_loss


            loss.backward()
            losses.append(loss.item())
            LTes.append(LT.item())
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")
        LCT=np.array(LTes).mean()
    for epoch_i in range(epochs):
        losses = []
        for step, batch_data in enumerate(data_loader):
            labels, _, tokens,rel_des_emb, rel_des_mask,tokens_query,mask_query= batch_data
            # labels, _, tokens, rel_des_emb, rel_des_mask, _, _ = batch_data
            META_optimizer.zero_grad()
            # logits_all = []
            logits_all_query = []

            # tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            tokens_query = torch.stack([x.to(config.device) for x in tokens_query], dim=0)

            labels = labels.to(config.device)
            origin_labels = labels[:]
            labels = [map_relid2tempid[x.item()] for x in labels]
            labels = torch.tensor(labels).to(config.device)

            # reps = encoder(tokens)
            reps_query = encoder(tokens_query)

            # normalized_reps_emb = F.normalize(reps.view(-1, reps.size()[1]), p=2, dim=1)
            normalized_reps_emb_query = F.normalize(reps_query.view(-1, reps_query.size()[1]), p=2, dim=1)

            outputs, _ = drop_layer(reps_query)
            if prev_drop_layer is not None:
                reps1 = prev_graph_encoder(tokens_query)
                prev_outputs, _ = prev_drop_layer(reps_query)
                positives, negatives = construct_hard_triplets(prev_outputs, origin_labels, new_relation_data)
            else:
                positives, negatives = construct_hard_triplets(outputs, origin_labels, new_relation_data)
            for _ in range(config.f_pass):
                # final_support, _ = drop_layer(reps)
                final_query,_=drop_layer(reps_query)

                # logits = classifier(final_support)
                logits1 = classifier(final_query)

                logits_all_query.append(logits1)
                # logits_all.append(logits)

            positives = torch.cat(positives, 0).to(config.device)
            negatives = torch.cat(negatives, 0).to(config.device)
            anchors = outputs

            # logits_all = torch.stack(logits_all)
            logits_all_query= torch.stack(logits_all_query)

            m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B

            # loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))
            loss3 = criterion(logits_all_query.reshape(-1, logits_all_query.shape[-1]), m_labels.reshape(-1))
            # loss2 = compute_jsd_loss(logits_all)
            loss4 = compute_jsd_loss(logits_all_query)
            tri_loss = triplet_loss(anchors, positives, negatives)
            # loss = loss1+ loss2+ tri_loss +loss3*0.5+loss4*0.5
            # loss = loss1 + loss2 + tri_loss
            loss = tri_loss +loss3*0.5+loss4*0.5
            if prev_encoder is not None:
                # prev_reps = prev_encoder(tokens).detach()
                prev_reps_query = prev_encoder(tokens_query).detach()

                # normalized_prev_reps_emb = F.normalize(prev_reps.view(-1, prev_reps.size()[1]), p=2, dim=1)
                normalized_prev_reps_emb_query = F.normalize(prev_reps_query.view(-1, prev_reps_query.size()[1]), p=2, dim=1)

                # feature_distill_loss = distill_criterion(normalized_reps_emb, normalized_prev_reps_emb,
                #                                          torch.ones(tokens.size(0)).to(
                #                                              config.device))
                feature_distill_loss_query = distill_criterion(normalized_reps_emb_query, normalized_prev_reps_emb_query,
                                                         torch.ones(tokens_query.size(0)).to(
                                                             config.device))
                # loss += feature_distill_loss*0.5 +feature_distill_loss_query*0.5
                loss += feature_distill_loss_query * 0.5

            if prev_drop_layer is not None and prev_classifier is not None:
                prediction_distill_loss_1 = None
                prediction_distill_loss_2 = None
                prediction_distill_loss_3 = None
                prediction_distill_loss_4 = None

                # dropout_output_all = []
                # prev_graph_output_all = []
                # prev_output_all = []
                prev_graph_output_all_query = []
                dropout_output_all_query = []
                prev_output_all_query = []

                rel_des_emb = torch.stack([x.to(config.device) for x in rel_des_emb], dim=0)
                rel_des_mask = torch.stack([x.to(config.device) for x in rel_des_mask], dim=0)
                rel_gol, rel_loc = prev_graph_decencoder(rel_des_emb, rel_des_mask)
                rel_loc = torch.mean(rel_loc[:, 1:, :], 1)
                rel_rep = torch.cat((rel_gol, rel_loc), -1).view(-1, 1, 1, rel_gol.shape[1] * 2)
                for i in range(config.f_pass):
                    # final_support, _ = drop_layer(reps)
                    final_query,_ =drop_layer(reps_query)

                    # graph_support, _ = prev_graph_layer(reps1, rel_rep)
                    graph_query,_=prev_graph_layer(reps1,rel_rep)

                    # final_support1, _ = prev_drop_layer(reps)
                    final_support2,_=prev_drop_layer(reps_query)

                    # dropout_output_all.append(final_support)
                    # prev_graph_output_all.append(graph_support)
                    prev_graph_output_all_query.append(graph_query)
                    # prev_output_all.append(final_support1)
                    prev_output_all_query.append(final_support2)
                    dropout_output_all_query.append(final_query)

                    # pre_logits_graph = prev_graph_classifier(graph_support).detach()
                    pre_logits_graph_query = prev_graph_classifier(graph_query).detach()
                    # pre_logits = prev_classifier(final_support1).detach()
                    pre_logits_query = prev_classifier(final_support2).detach()

                    # pre_logits_graph = F.softmax(pre_logits_graph.index_select(1, prev_relation_index) / T, dim=1)
                    pre_logits_graph_query = F.softmax(pre_logits_graph_query.index_select(1, prev_relation_index) / T,dim=1)
                    # pre_logits = F.softmax(pre_logits.index_select(1, prev_relation_index) / T, dim=1)
                    pre_logits_query = F.softmax(pre_logits_query.index_select(1, prev_relation_index) / T, dim=1)

                    # log_logits = F.log_softmax(logits_all[i].index_select(1, prev_relation_index) / T, dim=1)
                    log_logits_query = F.log_softmax(logits_all_query[i].index_select(1, prev_relation_index) / T,
                                                     dim=1)
                    if i == 0:
                        # prediction_distill_loss_1 = -torch.mean(torch.sum(pre_logits_graph * log_logits, dim=1))
                        # prediction_distill_loss_2 = -torch.mean(torch.sum(pre_logits * log_logits, dim=1))
                        prediction_distill_loss_3 = -torch.mean(torch.sum(pre_logits_query * log_logits_query, dim=1))
                        prediction_distill_loss_4 = -torch.mean(torch.sum(pre_logits_graph_query * log_logits_query, dim=1))
                    else:
                        # prediction_distill_loss_1 += -torch.mean(torch.sum(pre_logits_graph * log_logits, dim=1))
                        # prediction_distill_loss_2 += -torch.mean(torch.sum(pre_logits * log_logits, dim=1))
                        prediction_distill_loss_3 += -torch.mean(torch.sum(pre_logits_query * log_logits_query, dim=1))
                        prediction_distill_loss_4 += -torch.mean(torch.sum(pre_logits_graph_query * log_logits_query, dim=1))
                # prediction_distill_loss_1 /= config.f_pass
                # prediction_distill_loss_2 /= config.f_pass
                prediction_distill_loss_3 /= config.f_pass
                prediction_distill_loss_4 /= config.f_pass
                # loss += prediction_distill_loss_1*alpha+(1-alpha)*prediction_distill_loss_2+prediction_distill_loss_3*(1-alpha)\
                #         +prediction_distill_loss_4*0
                loss += prediction_distill_loss_3*(1-alpha)+prediction_distill_loss_4*0
                # loss += prediction_distill_loss_1 * alpha + (1 - alpha) * prediction_distill_loss_2
                # dropout_output_all = torch.stack(dropout_output_all)
                # prev_output_all = torch.stack(prev_output_all)
                prev_output_all_query=torch.stack(prev_output_all_query)
                dropout_output_all_query  = torch.stack(dropout_output_all_query )

                # mean_dropout_output_all = torch.mean(dropout_output_all, dim=0)
                mean_dropout_output_all_query = torch.mean(dropout_output_all_query, dim=0)
                # mean_prev_output_all = torch.mean(prev_output_all, dim=0)
                mean_prev_output_all_query = torch.mean(prev_output_all_query,dim=0)

                # normalized_output = F.normalize(mean_dropout_output_all.view(-1, mean_dropout_output_all.size()[1]),
                #                                 p=2, dim=1)
                normalized_output_query = F.normalize(mean_dropout_output_all_query.view(-1, mean_dropout_output_all_query.size()[1]), p=2, dim=1)
                # normalized_prev_output = F.normalize(mean_prev_output_all.view(-1, mean_prev_output_all.size()[1]), p=2,
                #                                      dim=1)
                mean_prev_output_all_query = F.normalize(mean_prev_output_all_query.view(-1,mean_prev_output_all_query.size()[1]), p=2, dim=1)

                # hidden_distill_loss = distill_criterion(normalized_output, normalized_prev_output,
                #                                         torch.ones(tokens.size(0)).to(
                #                                             config.device))
                hidden_distill_loss_1 = distill_criterion(normalized_output_query, mean_prev_output_all_query,
                                                        torch.ones(tokens_query.size(0)).to(
                                                            config.device))

                # loss += hidden_distill_loss*0.5+hidden_distill_loss_1*0.5
                loss += hidden_distill_loss_1 * 0.5

            loss.backward()
            losses.append(loss.item())
            META_optimizer.step()
        print(f"loss is {np.array(losses).mean()}")
        LCD = np.array(losses).mean()
    return LCT
def train_mem_model_1(config, encoder, drop_layer, classifier, training_data, epochs,
                    map_relid2tempid,new_relation_data,prev_encoder,prev_graph_encoder,prev_graph_decencoder ,
                    prev_graph_layer,prev_drop_layer,prev_classifier,
                    prev_graph_classifier,prev_relation_index,state,
                    ):
    data_loader = get_data_loader(config, training_data,shuffle=True)
    # expanded_train_data_for_initial, expanded_prev_samples 增广后的数据
    encoder.train()
    drop_layer.train()
    classifier.train()
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    criterion = nn.CrossEntropyLoss()
    META_optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': drop_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])
    distill_criterion = nn.CosineEmbeddingLoss()
    T = state["T"]
    alpha = state["W"]
    LCD=0
    for epoch_i in range(epochs):

        losses = []
        for step, batch_data in enumerate(data_loader):
            # labels, _, tokens,rel_des_emb, rel_des_mask,tokens_query,mask_query= batch_data
            labels, _, tokens,rel_des_emb, rel_des_mask,_,_= batch_data
            META_optimizer.zero_grad()
            logits_all = []
            logits_all_query = []

            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            # tokens_query = torch.stack([x.to(config.device) for x in tokens_query], dim=0)

            labels = labels.to(config.device)
            origin_labels = labels[:]
            labels = [map_relid2tempid[x.item()] for x in labels]
            labels = torch.tensor(labels).to(config.device)

            reps = encoder(tokens)
            # reps_query = encoder(tokens_query)


            normalized_reps_emb = F.normalize(reps.view(-1, reps.size()[1]), p=2, dim=1)
            # normalized_reps_emb_query = F.normalize(reps.view(-1, reps_query.size()[1]), p=2, dim=1)

            outputs, _= drop_layer(reps)
            if prev_drop_layer is not None:
                reps1 = prev_graph_encoder(tokens)
                prev_outputs, _= prev_drop_layer(reps)
                positives, negatives = construct_hard_triplets(prev_outputs, origin_labels, new_relation_data)
            else:
                positives, negatives = construct_hard_triplets(outputs, origin_labels, new_relation_data)
            for _ in range(config.f_pass):
                final_support,_= drop_layer(reps)
                # final_query,_=drop_layer(reps_query)

                logits = classifier(final_support)
                # logits1 = classifier(final_query)

                # logits_all_query.append(logits1)
                logits_all.append(logits)


            positives = torch.cat(positives, 0).to(config.device)
            negatives = torch.cat(negatives, 0).to(config.device)
            anchors = outputs

            logits_all = torch.stack(logits_all)
            # logits_all_query= torch.stack(logits_all_query)


            m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B

            loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))
            # loss3 = criterion(logits_all_query.reshape(-1, logits_all_query.shape[-1]), m_labels.reshape(-1))
            loss2 = compute_jsd_loss(logits_all)
            # loss4 = compute_jsd_loss(logits_all_query)
            tri_loss = triplet_loss(anchors, positives, negatives)
            # loss = loss1+ loss2+ tri_loss +loss3*0.5+loss4*0.5
            loss = loss1+ loss2+ tri_loss
            # loss = loss1+ loss2+ tri_loss
            if prev_encoder is not None:
                prev_reps = prev_encoder(tokens).detach()
                # prev_reps_query = prev_encoder(tokens_query).detach()

                normalized_prev_reps_emb = F.normalize(prev_reps.view(-1, prev_reps.size()[1]), p=2, dim=1)
                # normalized_prev_reps_emb_query = F.normalize(prev_reps.view(-1, prev_reps_query.size()[1]), p=2, dim=1)

                feature_distill_loss = distill_criterion(normalized_reps_emb, normalized_prev_reps_emb,
                                                         torch.ones(tokens.size(0)).to(
                                                             config.device))
                # feature_distill_loss_query = distill_criterion(normalized_reps_emb_query, normalized_prev_reps_emb_query,
                #                                          torch.ones(tokens.size(0)).to(
                #                                              config.device))
                # loss += feature_distill_loss*0.5 +feature_distill_loss_query*0.5
                loss += feature_distill_loss*0.5

            if prev_drop_layer is not None and prev_classifier is not None:
                prediction_distill_loss_1 = None
                prediction_distill_loss_2 = None
                prediction_distill_loss_3 = None
                prediction_distill_loss_4 = None

                dropout_output_all = []
                prev_graph_output_all = []
                prev_output_all=[]
                prev_graph_output_all_query=[]
                dropout_output_all_query = []
                prev_output_all_query = []


                rel_des_emb = torch.stack([x.to(config.device) for x in rel_des_emb], dim=0)
                rel_des_mask = torch.stack([x.to(config.device) for x in rel_des_mask], dim=0)
                rel_gol, rel_loc = prev_graph_decencoder(rel_des_emb, rel_des_mask)
                rel_loc = torch.mean(rel_loc[:, 1:, :], 1)
                rel_rep = torch.cat((rel_gol, rel_loc), -1).view(-1, 1, 1, rel_gol.shape[1] * 2)
                for i in range(config.f_pass):
                    final_support, _= drop_layer(reps)
                    # final_query,_ =drop_layer(reps_query)

                    graph_support,_=prev_graph_layer(reps1,rel_rep)
                    # graph_query,_=prev_graph_layer(reps1,rel_rep)

                    final_support1,_=prev_drop_layer(reps)
                    # final_support2,_=prev_drop_layer(reps_query)

                    dropout_output_all.append(final_support)
                    prev_graph_output_all.append(graph_support)
                    # prev_graph_output_all_query.append(graph_query)
                    prev_output_all.append(final_support1)
                    # prev_output_all_query.append(final_support2)
                    # dropout_output_all_query.append(final_query)

                    pre_logits_graph = prev_graph_classifier(graph_support).detach()
                    # pre_logits_graph_query = prev_graph_classifier(graph_query).detach()
                    pre_logits = prev_classifier(final_support1).detach()
                    # pre_logits_query = prev_classifier(final_support2).detach()


                    pre_logits_graph = F.softmax(pre_logits_graph.index_select(1, prev_relation_index) / T, dim=1)
                    # pre_logits_graph_query = F.softmax(pre_logits_graph_query.index_select(1, prev_relation_index) / T,dim=1)
                    pre_logits = F.softmax(pre_logits.index_select(1, prev_relation_index) / T, dim=1)
                    # pre_logits_query = F.softmax(pre_logits_query.index_select(1, prev_relation_index) / T, dim=1)

                    log_logits = F.log_softmax(logits_all[i].index_select(1, prev_relation_index) / T, dim=1)
                    # log_logits_query = F.log_softmax(logits_all_query[i].index_select(1, prev_relation_index) / T, dim=1)
                    if i == 0:
                        prediction_distill_loss_1 = -torch.mean(torch.sum(pre_logits_graph * log_logits, dim=1))
                        prediction_distill_loss_2 = -torch.mean(torch.sum(pre_logits * log_logits, dim=1))
                        # prediction_distill_loss_3 = -torch.mean(torch.sum(pre_logits_query * log_logits_query, dim=1))
                        # prediction_distill_loss_4 = -torch.mean(torch.sum(pre_logits_graph_query * log_logits_query, dim=1))
                    else:
                        prediction_distill_loss_1 += -torch.mean(torch.sum(pre_logits_graph * log_logits, dim=1))
                        prediction_distill_loss_2 += -torch.mean(torch.sum(pre_logits * log_logits, dim=1))
                        # prediction_distill_loss_3 += -torch.mean(torch.sum(pre_logits_query * log_logits_query, dim=1))
                        # prediction_distill_loss_4 += -torch.mean(torch.sum(pre_logits_graph_query * log_logits_query, dim=1))
                prediction_distill_loss_1 /= config.f_pass
                prediction_distill_loss_2 /= config.f_pass
                # prediction_distill_loss_3 /= config.f_pass
                # prediction_distill_loss_4 /= config.f_pass
                # loss += prediction_distill_loss_1*alpha+(1-alpha)*prediction_distill_loss_2+prediction_distill_loss_3*(1-alpha)\
                #         +prediction_distill_loss_4*0
                loss += prediction_distill_loss_1 * alpha + (
                            1 - alpha) * prediction_distill_loss_2
                # loss += prediction_distill_loss_1 * alpha + (1 - alpha) * prediction_distill_loss_2
                dropout_output_all = torch.stack(dropout_output_all)
                prev_output_all = torch.stack(prev_output_all)
                # prev_output_all_query=torch.stack(prev_output_all_query)
                # dropout_output_all_query  = torch.stack(dropout_output_all_query )

                mean_dropout_output_all = torch.mean(dropout_output_all, dim=0)
                # mean_dropout_output_all_query = torch.mean(dropout_output_all_query, dim=0)
                mean_prev_output_all = torch.mean(prev_output_all,dim=0)
                # mean_prev_output_all_query = torch.mean(prev_output_all_query,dim=0)

                normalized_output = F.normalize(mean_dropout_output_all.view(-1, mean_dropout_output_all.size()[1]), p=2, dim=1)
                # normalized_output_query = F.normalize(mean_dropout_output_all_query.view(-1, mean_dropout_output_all_query.size()[1]), p=2, dim=1)
                normalized_prev_output = F.normalize(mean_prev_output_all.view(-1, mean_prev_output_all.size()[1]), p=2, dim=1)
                # mean_prev_output_all_query = F.normalize(mean_prev_output_all_query.view(-1,mean_prev_output_all_query.size()[1]), p=2, dim=1)

                hidden_distill_loss = distill_criterion(normalized_output, normalized_prev_output,
                                                         torch.ones(tokens.size(0)).to(
                                                             config.device))
                # hidden_distill_loss_1 = distill_criterion(normalized_output_query, mean_prev_output_all_query,
                #                                         torch.ones(tokens.size(0)).to(
                #                                             config.device))

                # loss += hidden_distill_loss*0.5+hidden_distill_loss_1*0.5
                loss += hidden_distill_loss*0.5


            loss.backward()
            losses.append(loss.item())
            META_optimizer.step()
        print(f"loss is {np.array(losses).mean()}")
        LCD=np.array(losses).mean()
    return LCD

def compute_jsd_loss(m_input):
    # m_input: the result of m times dropout after the classifier.
    # size: m*B*C
    m = m_input.shape[0]
    mean = torch.mean(m_input, dim=0)
    jsd = 0
    for i in range(m):
        # Lcsf = − 1|Dj |∑x∈Dj|Rj |∑r=1yx,r · log Px,r,
        loss = F.kl_div(F.log_softmax(mean, dim=-1), F.softmax(m_input[i], dim=-1), reduction='none')
        loss = loss.sum()
        jsd += loss / m
    return jsd


def select_data(config, encoder, drop_layer, relation_dataset):
    data_loader = get_data_loader(config, relation_dataset,shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()
    drop_layer.eval()
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens,rel_des_emb,rel_des_mask,tokens_query,mask_query = batch_data
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            reps = encoder(tokens)
            feature= drop_layer(reps)[1].cpu()
        features.append(feature)

    features = np.concatenate(features)
    num_clusters = min(config.num_protos, len(relation_dataset))
    distances = KMeans(n_clusters=num_clusters, random_state=0,n_init=10).fit_transform(features)
    memory = []
    for k in range(num_clusters):
        sel_index = np.argmin(distances[:, k])
        instance = relation_dataset[sel_index]
        memory.append(instance)
    return memory


def get_proto(config, encoder, drop_layer, relation_dataset):
    #  这里是将数据用dataloader 拿出来
    data_loader = get_data_loader(config, relation_dataset,shuffle=False, drop_last=False, batch_size=1)
    features = []
    #  把两个设置成eval形式 防止这里使用改变了他的权重
    encoder.eval()
    drop_layer.eval()
    # 依次拿出来数据
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens,rel_des_emb,rel_des_mask,tokens_query,mask_query= batch_data

        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)


        with torch.no_grad():
            reps = encoder(tokens)
            feature = drop_layer(reps)[1].cpu()

        features.append(feature)
        # 整合成一个tensor
    features = torch.cat(features, dim=0)
        # 将数据取均值获得原型
    proto = torch.mean(features, dim=0, keepdim=True).cpu()
        # 这个标准是平方根后的结果
        # 为啥要有标准呢
    standard = torch.sqrt(torch.var(features, dim=0)).cpu()
    return proto, standard

def generate_relation_data(protos, relation_standard):
    relation_data = {}
    relation_sample_nums = 10
    for id in protos.keys():
        relation_data[id] = []
        difference = np.random.normal(loc=0, scale=1, size=relation_sample_nums)
        # print(difference)
        # print("==============================================================")
        for diff in difference:
            # print(f"relation_standard: {relation_standard[id]}")
            # print(f"relation_standard: {relation_standard.size()}")
            # print(f"protos:{protos.size()}")
            # print(protos[id])
            relation_data[id].append(protos[id] + diff * relation_standard[id])
            # print("============================================================")
            # print(relation_data[id])
    return relation_data

def generate_current_relation_data(config, encoder, drop_layer, relation_dataset,):
    data_loader = get_data_loader(config, relation_dataset, shuffle=False, drop_last=False, batch_size=1)
    relation_data = []
    encoder.eval()
    drop_layer.eval()
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens,rel_des_emb,rel_des_mask,tokens_query,mask_query = batch_data

        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)


        with torch.no_grad():
            reps = encoder(tokens)
            feature = drop_layer(reps)[1].cpu()
        relation_data.append(feature)
    return relation_data

def construct_hard_triplets(output, labels, relation_data):
    positive = []
    negative = []
    pdist = nn.PairwiseDistance(p=2)
    for rep, label in zip(output, labels):
        positive_relation_data = relation_data[label.item()]
        negative_relation_data = []
        for key in relation_data.keys():
            if key != label.item():
                negative_relation_data.extend(relation_data[key])
        positive_distance = torch.stack([pdist(rep.cpu(), p) for p in positive_relation_data])
        negative_distance = torch.stack([pdist(rep.cpu(), n) for n in negative_relation_data])
        positive_index = torch.argmax(positive_distance)
        # print(f"positive_index:{positive_index}")
        # print(f"positive_index:{positive_index.size()}")
        negative_index = torch.argmin(negative_distance)
        # print(f"negative_index:{negative_index}")
        # print(f"negative_index:{negative_index.size()}")
        positive.append(positive_relation_data[positive_index.item()])
        negative.append(negative_relation_data[negative_index.item()])


    return positive, negative


def data_augmentation(config, encoder, train_data, prev_train_data):
    # train_data_for_initial 这一轮的训练数据
    # prev_samples 从开始到这次运行之前的记忆体的sample
    # 第一运行的时候没有
    expanded_train_data = train_data[:]
    expanded_prev_train_data = prev_train_data[:]
    encoder.eval()
    #  运行到现在的所有数据
    all_data = train_data + prev_train_data
    tokenizer = BertTokenizer.from_pretrained(config.bert_path, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    entity_index = []
    entity_mention = []
    for sample in all_data:
        # 找个e11 e12 对应值的索引
        e11 = sample['tokens'].index(30522)
        e12 = sample['tokens'].index(30523)
        e21 = sample['tokens'].index(30524)
        e22 = sample['tokens'].index(30525)
        # 加入索引到列表
        entity_index.append([e11,e12])
        # 这里把每个数据的头实体和尾实体 都拿出来放到entity_mention
        entity_mention.append(sample['tokens'][e11+1:e12])
        entity_index.append([e21,e22])
        entity_mention.append(sample['tokens'][e21+1:e22])

    data_loader = get_data_loader(config, all_data, shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens,rel_des_emb,rel_des_mask,tokens_query,mask_query= batch_data
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            # [B, H * 2]
            feature = encoder(tokens)
        feature1, feature2 = torch.split(feature, [config.encoder_output_size,config.encoder_output_size], dim=1)
        features.append(feature1)
        features.append(feature2)
    # 这里相当于将所有的feature重新整合打包成一个tensor 也就是从列表到tensor
    features = torch.cat(features, dim=0)
    # similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)

    similarity_matrix = []
    for i in range(len(features)):
        similarity_matrix.append([0]*len(features))

    for i in range(len(features)):
        for j in range(i,len(features)):
            similarity = F.cosine_similarity(features[i],features[j],dim=0)
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

    similarity_matrix = torch.tensor(similarity_matrix).to(config.device)
    zero = torch.zeros_like(similarity_matrix).to(config.device)
    diag = torch.diag_embed(torch.diag(similarity_matrix))
    similarity_matrix -= diag
    similarity_matrix = torch.where(similarity_matrix<0.95, zero, similarity_matrix)
    nonzero_index = torch.nonzero(similarity_matrix)
    expanded_train_count = 0
    expanded_count=0
    for origin, replace in nonzero_index:
        sample_index = int(origin/2)
        sample = all_data[sample_index]
        if entity_mention[origin] == entity_mention[replace]:
            continue
        new_tokens = sample['tokens'][:entity_index[origin][0]+1] + entity_mention[replace] + \
                     sample['tokens'][entity_index[origin][1]:]
        if len(new_tokens) < config.max_length:
            new_tokens = new_tokens + [0]*(config.max_length-len(new_tokens))
        else:
            new_tokens = new_tokens[:config.max_length]

        new_sample = {
            'relation': sample['relation'],
            'neg_labels': sample['neg_labels'],
            'tokens': new_tokens,
        }
        if sample_index < len(train_data) and expanded_train_count < 5 * len(train_data):
            expanded_train_data.append(new_sample)
            expanded_train_count += 1
            expanded_count += 1
        else:
            expanded_prev_train_data.append(new_sample)
            expanded_count += 1
    print(f"增广数据的数量：{expanded_count}")
    return expanded_train_data, expanded_prev_train_data

