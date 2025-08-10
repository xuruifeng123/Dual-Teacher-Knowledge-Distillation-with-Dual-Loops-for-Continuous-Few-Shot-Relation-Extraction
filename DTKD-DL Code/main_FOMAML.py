# @Descripttion ：
# -*- codeing: utf-8 -*-
# @time ：2025/6/18 22:52
# @Author : Ruifeng Xu
# @Site : 
# @file : main_FOMAML.py
# @Sofeware : PyCharm

import argparse
import math
import random
from config import Config
import torch
import numpy as np
from copy import deepcopy
import pdb
import time
from sampler import data_sampler
from model.bert_encoder import Bert_Encoder, desc_encoder, Bert_Encoder1
from model.dropout_layer import Dropout_Layer
from model.classifier import Softmax_Layer, Softmax_Layer1
# from model.Graph_Layer import graph_layer
from data_loader import get_data_loader
from base_function_FOMAML import train_simple_model, evaluate_strict_model, train_mem_model, select_data, train_mem_model_1
from base_function_FOMAML import generate_current_relation_data, get_proto, generate_relation_data
from base_function_FOMAML import data_augmentation
from model.Graph_Layer import graph_layer
from model.RL_Agent import RL_Env, Agent, policy_improve
import logging

# 配置logging模块

epsilon = 1


def main():
    # terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="fewrel", type=str, help="type of task")
    parser.add_argument("--shot", default=10, type=int, help="N-way-k-shot")
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--gpu', default=0, type=int, required=True)
    parser.add_argument('--seed', default=100, type=int, help=" random seed")
    parser.add_argument('--total_round', default=6, type=int)
    parser.add_argument('--task_length', default=8, type=int, help="Number of tasks")
    parser.add_argument('--cat_entity_rep', action='store_true',
                        help='concatenate entity representation as sentence rep')
    parser.add_argument('--backend_model', default="bert", help='checkpoint name')

    args = parser.parse_args()
    config = Config(args.config)
    config.task_length = args.task_length
    config.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    config.batch_size_per_step = int(config.batch_size / config.gradient_accumulation_steps)
    config.shot = args.shot
    config.task = args.task
    config.step1_epochs = 5
    config.step2_epochs = 7
    config.step3_epochs = 20
    result_cur_time = []
    result_cur_test = []
    result_whole_test = []
    #  这里的bwt fwt 我得看看是啥意思用来存储啥的
    bwt_whole = []
    fwt_whole = []
    relation_divides = []
    for i in range(10):
        relation_divides.append([])
    relation_divides = []
    for i in range(10):
        relation_divides.append([])

    for rou in range(config.total_round):
        # 6轮随机值 分别是 100 ，200，300，400，500，600
        encoder = Bert_Encoder(config=config).to(config.device)
        encoder1 = Bert_Encoder1(config=config).to(config.device)
        graphlayer = graph_layer(config).to(config.device)
        droplayer = Dropout_Layer(config).to(config.device)
        descencoder = desc_encoder(config).to(config.device)

        random.seed(config.seed + rou * 100)

        # graphlayer=graph_layer(config).to(config.device)

        # test_cur_RL=[]
        # test_total_RL=[]

        env = RL_Env(config)
        RL_Agent = Agent(env)
        stand = 0
        epsilon2 = 15
        total_reward = 0
        rounds = rou
        start_time = time.time()
        while True:

            if args.task == "fewrel":
                config.num_of_relation = 80
                if args.shot == 5:
                    sampler = data_sampler(config=config, seed=config.seed + rounds * 100,
                                           relation_file="data/fewrel/relation_name.txt",
                                           rel_des_file="data/fewrel/relation_description.txt",
                                           rel_cluster_labels="data/fewrel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy",
                                           rel_index="data/fewrel/rel_index.npy",
                                           training_file="data/fewrel/CFRLdata_10_100_10_5/train_0.txt",
                                           valid_file="data/fewrel/CFRLdata_10_100_10_5/valid_0.txt",
                                           test_file="data/fewrel/CFRLdata_10_100_10_5/test_0.txt")
                elif args.shot == 10:
                    sampler = data_sampler(config=config, seed=config.seed + rounds * 100,
                                           relation_file="data/fewrel/relation_name.txt",
                                           rel_des_file="data/fewrel/relation_description.txt",
                                           rel_cluster_labels="data/fewrel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy",
                                           rel_index="data/fewrel/rel_index.npy",
                                           training_file="data/fewrel/CFRLdata_10_100_10_10/train_0.txt",
                                           valid_file="data/fewrel/CFRLdata_10_100_10_10/valid_0.txt",
                                           test_file="data/fewrel/CFRLdata_10_100_10_10/test_0.txt")
                else:
                    raise ValueError("The input must be an integer 5 or 10")
            elif args.task == "tacred":
                config.num_of_relation = 41
                if args.shot == 5:
                    sampler = data_sampler(config=config, seed=config.seed + rounds * 100,
                                           relation_file="data/tacred/relation_name.txt",
                                           rel_des_file="data/tacred/relation_description.txt",
                                           rel_cluster_labels="data/tacred/CFRLdata_10_100_10_5/rel_cluster_label_0.npy",
                                           rel_index="data/tacred/rel_index.npy",
                                           training_file="data/tacred/CFRLdata_10_100_10_5/train_0.txt",
                                           valid_file="data/tacred/CFRLdata_10_100_10_5/valid_0.txt",
                                           test_file="data/tacred/CFRLdata_10_100_10_5/test_0.txt")
                elif args.shot == 10:
                    sampler = data_sampler(config=config, seed=config.seed + rounds * 100,
                                           relation_file="data/tacred/relation_name.txt",
                                           rel_des_file="data/tacred/relation_description.txt",
                                           rel_cluster_labels="data/tacred/CFRLdata_10_100_10_10/rel_cluster_label_0.npy",
                                           rel_index="data/tacred/rel_index.npy",
                                           training_file="data/tacred/CFRLdata_10_100_10_10/train_0.txt",
                                           valid_file="data/tacred/CFRLdata_10_100_10_10/valid_0.txt",
                                           test_file="data/tacred/CFRLdata_10_100_10_10/test_0.txt")
                else:
                    raise ValueError("The input must be an integer 5 or 10")
            else:
                print("The task is wrong")
                assert (0)
            rel2id = sampler.rel_id
            # rounds+=1

            classifier = None
            prev_classifier = None
            prev_encoder = None
            prev_drop_layer = None
            prev_graph_encoder = None
            prev_graph_descencoder = None
            prev_graph_layer = None
            prev_graph_classifier = None

            history_data = []
            relation_standard = {}
            memorized_samples = {}
            history_relations = []
            forward_accs = []

            state = env.reset()
            prev_act = -1
            prev_state = {"input": None, "teacher": None, "student": None, "T": 0, "T_index": 0, "W": 0, "W_index": 0,
                          "iter": -1}
            states = []
            valid = []
            for steps, (
                    training_data, valid_data, test_data, current_relations, historic_test_data,
                    seen_relations) in enumerate(
                sampler):

                state["input"] = current_relations
                print(current_relations)
                prev_relations = history_relations[:]
                train_data_for_initial = []
                count = 0
                for relation in current_relations:
                    history_relations.append(relation)
                    train_data_for_initial += training_data[relation]
                    relation_divides[count].append(float(rel2id[relation]))
                    count += 1

                temp_rel2id = [rel2id[x] for x in seen_relations]
                map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
                prev_relation_index = []
                prev_samples = []
                relation_index = []
                for relation in current_relations:
                    relation_index.append(map_relid2tempid[rel2id[relation]])
                relation_index = torch.tensor(prev_relation_index).to(config.device)

                for relation in prev_relations:
                    prev_relation_index.append(map_relid2tempid[rel2id[relation]])
                    prev_samples += memorized_samples[relation]
                prev_relation_index = torch.tensor(prev_relation_index).to(config.device)

                classifier = Softmax_Layer(input_size=encoder.output_size, num_class=len(history_relations)).to(
                    config.device)
                classifier1 = Softmax_Layer1(input_size=encoder.output_size, num_class=len(history_relations)).to(
                    config.device)

                print(f"当前任务的数据{len(train_data_for_initial)}")
                train_simple_model(config=config, encoder=encoder, descencoder=descencoder, graph_layer=graphlayer
                                   , dropout_layer=droplayer, classifier=classifier,
                                   training_data=train_data_for_initial,
                                   epochs=config.step1_epochs, map_relid2tempid=map_relid2tempid,
                                   encoder1=encoder1, classifier1=classifier1)

                print(f"simple finished")

                # prev_graph_encoder1 = deepcopy(encoder1)
                # prev_graph_descencoder1 = deepcopy(descencoder)
                # prev_graph_layer1 = deepcopy(graphlayer)
                # prev_graph_classifier1 = deepcopy(classifier1)

                temp_protos = {}
                for relation in current_relations:
                    proto, standard = get_proto(config=config, encoder=encoder,
                                                drop_layer=droplayer, relation_dataset=training_data[relation],
                                                )
                    temp_protos[rel2id[relation]] = proto
                    relation_standard[rel2id[relation]] = standard
                print(f"以往数据显示：{prev_relations}")
                for relation in prev_relations:
                    proto, _ = get_proto(config=config, encoder=encoder, drop_layer=droplayer,
                                         relation_dataset=memorized_samples[relation])
                    temp_protos[rel2id[relation]] = proto

                new_relation_data = generate_relation_data(temp_protos, relation_standard)

                for relation in current_relations:
                    new_relation_data[rel2id[relation]].extend(
                        generate_current_relation_data(config, encoder, droplayer, training_data[relation]))

                print(f"current data number:                {len(train_data_for_initial)}")
                stand += 1
                epsilon = 0.1 + 0.9 * math.exp(-1 * stand / epsilon2)
                act, tag = RL_Agent.play(state, epsilon)
                state = state.copy()
                config.step = steps
                print(f"当前蒸馏温度与权重：{state}")
                encoder_FOMAML=deepcopy(encoder)
                droplayer_FOMAML=deepcopy(droplayer)
                classifier_FOMAML=deepcopy(classifier)
                LCT = train_mem_model(config=config, encoder=encoder, drop_layer=droplayer, classifier=classifier,
                                      training_data=train_data_for_initial, epochs=config.step2_epochs,
                                      map_relid2tempid=map_relid2tempid, prev_graph_encoder=prev_graph_encoder,
                                      prev_graph_layer=prev_graph_layer,
                                      prev_graph_classifier=prev_graph_classifier,
                                      prev_relation_index=prev_relation_index,
                                      prev_graph_decencoder=prev_graph_descencoder,
                                      new_relation_data=new_relation_data, prev_encoder=prev_encoder,
                                      prev_drop_layer=prev_drop_layer,
                                      prev_classifier=prev_classifier, state=state,
                                      relation_index=relation_index,encoder_FOMAML=encoder_FOMAML,droplayer_FOMAML=droplayer_FOMAML,classifier_FOMAML=classifier_FOMAML)

                for relation in current_relations:
                    memorized_samples[relation] = select_data(config, encoder_FOMAML, droplayer_FOMAML, training_data[relation])
                train_data_for_memory = []
                # train_data_for_memory += expanded_prev_samples
                train_data_for_memory += prev_samples
                for relation in current_relations:
                    train_data_for_memory += memorized_samples[relation]

                print(f"输出从运行开始到现在的关系：{len(seen_relations)}")
                print(f"输出要训练的记忆体的数量：{len(train_data_for_memory)}")
                LCD = train_mem_model_1(config=config, encoder=encoder_FOMAML, drop_layer=droplayer_FOMAML, classifier=classifier_FOMAML,
                                        training_data=train_data_for_memory, epochs=config.step3_epochs,
                                        map_relid2tempid=map_relid2tempid, prev_graph_encoder=prev_graph_encoder,
                                        prev_graph_layer=prev_graph_layer,
                                        prev_graph_classifier=prev_graph_classifier,
                                        prev_relation_index=prev_relation_index,
                                        prev_graph_decencoder=prev_graph_descencoder,
                                        new_relation_data=new_relation_data, prev_encoder=prev_encoder,
                                        prev_drop_layer=prev_drop_layer,
                                        prev_classifier=prev_classifier, state=state)
                print(f"memory finished")

                next_state, done = env.step(act, tag, current_data=current_relations, teacher=LCT, student=LCD)
                next_state = next_state.copy()

                prev_graph_encoder = deepcopy(encoder1)
                prev_encoder = deepcopy(encoder_FOMAML)

                prev_graph_descencoder = deepcopy(descencoder)
                prev_graph_layer = deepcopy(graphlayer)
                prev_drop_layer = deepcopy(droplayer_FOMAML)

                prev_graph_classifier = deepcopy(classifier1)
                prev_classifier = deepcopy(classifier_FOMAML)
                print(f'Restart Num {rou + 1}')
                print(f'task--{steps + 1}:')

                # expanded_train_data_for_initial, expanded_prev_samples = \
                #     data_augmentation(config, encoder,train_data_for_initial,prev_samples)
                # valid+=expanded_train_data_for_initial
                # valid+=expanded_prev_samples
                # valid_acc = evaluate_strict_model(config=config, encoder=prev_encoder,
                #                                     dropout_layer=prev_drop_layer, classifier=classifier,
                #                                     test_data=valid, seen_relations=seen_relations,
                #                                     map_relid2tempid=map_relid2tempid, rel2id=rel2id)
                # reward = env.reward(loss=LCD+LCT,accuracy=valid_acc)

                reward = env.reward(loss=LCD + LCT)
                print(f"当前奖励：{reward}")
                if prev_state["iter"] != -1:
                    return_val = reward + RL_Agent.gamma * (
                        0 if done else RL_Agent.value_q[state["iter"], :].argmax())
                    RL_Agent.value_n[prev_state["iter"]][prev_act] += 1
                    RL_Agent.value_q[prev_state["iter"]][prev_act] += (return_val -
                                                                       RL_Agent.value_q[prev_state["iter"]][prev_act]) / \
                                                                      RL_Agent.value_n[prev_state["iter"]][prev_act]
                prev_act = act
                prev_state = state
                states.append(state)
                state = next_state

            test_policy = RL_Agent.pi.cpu().detach().numpy()
            ret = policy_improve(RL_Agent)

            if not ret:
                break

        # RL_OVER
        print(f"最终强化学习对应策略:{test_policy}")
        states1 = states
        print(f"最后的蒸馏温度和权重:f{states1}")

        test_cur = []
        test_total = []
        history_data = []
        relation_standard = {}
        memorized_samples = {}
        history_relations = []
        forward_accs = []
        # rel2id = sampler.rel_id
        # rounds+=1
        classifier = None
        prev_classifier = None
        prev_encoder = None
        prev_drop_layer = None
        prev_graph_encoder = None
        prev_graph_descencoder = None
        prev_graph_layer = None
        prev_graph_classifier = None
        if args.task == "fewrel":
            config.num_of_relation = 80
            if args.shot == 5:
                sampler = data_sampler(config=config, seed=config.seed + rounds * 100,
                                       relation_file="data/fewrel/relation_name.txt",
                                       rel_des_file="data/fewrel/relation_description.txt",
                                       rel_cluster_labels="data/fewrel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy",
                                       rel_index="data/fewrel/rel_index.npy",
                                       training_file="data/fewrel/CFRLdata_10_100_10_5/train_0.txt",
                                       valid_file="data/fewrel/CFRLdata_10_100_10_5/valid_0.txt",
                                       test_file="data/fewrel/CFRLdata_10_100_10_5/test_0.txt")
            elif args.shot == 10:
                sampler = data_sampler(config=config, seed=config.seed + rounds * 100,
                                       relation_file="data/fewrel/relation_name.txt",
                                       rel_des_file="data/fewrel/relation_description.txt",
                                       rel_cluster_labels="data/fewrel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy",
                                       rel_index="data/fewrel/rel_index.npy",
                                       training_file="data/fewrel/CFRLdata_10_100_10_10/train_0.txt",
                                       valid_file="data/fewrel/CFRLdata_10_100_10_10/valid_0.txt",
                                       test_file="data/fewrel/CFRLdata_10_100_10_10/test_0.txt")
            else:
                raise ValueError("The input must be an integer 5 or 10")
        elif args.task == "tacred":
            config.num_of_relation = 41
            if args.shot == 5:
                sampler = data_sampler(config=config, seed=config.seed + rounds * 100,
                                       relation_file="data/tacred/relation_name.txt",
                                       rel_des_file="data/tacred/relation_description.txt",
                                       rel_cluster_labels="data/tacred/CFRLdata_10_100_10_5/rel_cluster_label_0.npy",
                                       rel_index="data/tacred/rel_index.npy",
                                       training_file="data/tacred/CFRLdata_10_100_10_5/train_0.txt",
                                       valid_file="data/tacred/CFRLdata_10_100_10_5/valid_0.txt",
                                       test_file="data/tacred/CFRLdata_10_100_10_5/test_0.txt")
            elif args.shot == 10:
                sampler = data_sampler(config=config, seed=config.seed + rounds * 100,
                                       relation_file="data/tacred/relation_name.txt",
                                       rel_des_file="data/tacred/relation_description.txt",
                                       rel_cluster_labels="data/tacred/CFRLdata_10_100_10_10/rel_cluster_label_0.npy",
                                       rel_index="data/tacred/rel_index.npy",
                                       training_file="data/tacred/CFRLdata_10_100_10_10/train_0.txt",
                                       valid_file="data/tacred/CFRLdata_10_100_10_10/valid_0.txt",
                                       test_file="data/tacred/CFRLdata_10_100_10_10/test_0.txt")
            else:
                raise ValueError("The input must be an integer 5 or 10")
        else:
            print("The task is wrong")
            assert (0)
        rel2id = sampler.rel_id

        for steps, (
                training_data, valid_data, test_data, current_relations, historic_test_data,
                seen_relations) in enumerate(
            sampler):
            # state=None
            state["input"] = current_relations
            state["teacher"] = 0.0
            state["student"] = 0.0
            state["T"] = states1[steps]["T"]
            state["T_index"] = states[steps]["T_index"]
            state["W"] = states1[steps]["W"]
            state["W_index"] = states1[steps]["W_index"]
            state["iter"] = 0
            print(current_relations)
            prev_relations = history_relations[:]
            train_data_for_initial = []
            count = 0
            for relation in current_relations:
                history_relations.append(relation)
                train_data_for_initial += training_data[relation]
                relation_divides[count].append(float(rel2id[relation]))
                count += 1

            temp_rel2id = [rel2id[x] for x in seen_relations]
            map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
            prev_relation_index = []
            prev_samples = []
            relation_index = []
            for relation in current_relations:
                relation_index.append(map_relid2tempid[rel2id[relation]])
            relation_index = torch.tensor(prev_relation_index).to(config.device)

            for relation in prev_relations:
                prev_relation_index.append(map_relid2tempid[rel2id[relation]])
                prev_samples += memorized_samples[relation]
            prev_relation_index = torch.tensor(prev_relation_index).to(config.device)

            classifier = Softmax_Layer(input_size=encoder.output_size, num_class=len(history_relations)).to(
                config.device)
            classifier1 = Softmax_Layer1(input_size=encoder.output_size, num_class=len(history_relations)).to(
                config.device)

            test_data_1 = []
            for relation in current_relations:
                test_data_1 += test_data[relation]
            if steps != 0:
                forward_acc = evaluate_strict_model(config=config, encoder=prev_encoder,
                                                    dropout_layer=prev_drop_layer, classifier=classifier,
                                                    test_data=test_data_1, seen_relations=seen_relations,
                                                    map_relid2tempid=map_relid2tempid, rel2id=rel2id)
                forward_accs.append(forward_acc)
            print(f"当前任务的数据{len(train_data_for_initial)}")
            train_simple_model(config=config, encoder=encoder, descencoder=descencoder, graph_layer=graphlayer
                               , dropout_layer=droplayer, classifier=classifier, training_data=train_data_for_initial,
                               epochs=config.step1_epochs, map_relid2tempid=map_relid2tempid,
                               encoder1=encoder1, classifier1=classifier1)

            print(f"simple finished")

            # prev_graph_encoder1 = deepcopy(encoder1)
            # prev_graph_descencoder1 = deepcopy(descencoder)
            # prev_graph_layer1 = deepcopy(graphlayer)
            # prev_graph_classifier1 = deepcopy(classifier1)

            temp_protos = {}
            for relation in current_relations:
                proto, standard = get_proto(config=config, encoder=encoder,
                                            drop_layer=droplayer, relation_dataset=training_data[relation],
                                            )
                temp_protos[rel2id[relation]] = proto
                relation_standard[rel2id[relation]] = standard
            print(f"以往数据显示：{prev_relations}")
            for relation in prev_relations:
                proto, _ = get_proto(config=config, encoder=encoder, drop_layer=droplayer,
                                     relation_dataset=memorized_samples[relation])
                temp_protos[rel2id[relation]] = proto

            new_relation_data = generate_relation_data(temp_protos, relation_standard)

            for relation in current_relations:
                new_relation_data[rel2id[relation]].extend(
                    generate_current_relation_data(config, encoder, droplayer, training_data[relation]))

            # if steps!=0:
            #     expanded_train_data_for_initial, expanded_prev_samples = data_augmentation(config, encoder,train_data_for_initial,prev_samples)
            print(f"current data number:                {len(train_data_for_initial)}")
            # print(f"当前蒸馏温度与权重：{state}")
            encoder_FOMAML = deepcopy(encoder)
            droplayer_FOMAML = deepcopy(droplayer)
            classifier_FOMAML = deepcopy(classifier)
            LCT = train_mem_model(config=config, encoder=encoder, drop_layer=droplayer, classifier=classifier,
                                  training_data=train_data_for_initial, epochs=config.step2_epochs,
                                  map_relid2tempid=map_relid2tempid, prev_graph_encoder=prev_graph_encoder,
                                  prev_graph_layer=prev_graph_layer,
                                  prev_graph_classifier=prev_graph_classifier, prev_relation_index=prev_relation_index,
                                  prev_graph_decencoder=prev_graph_descencoder,
                                  new_relation_data=new_relation_data, prev_encoder=prev_encoder,
                                  prev_drop_layer=prev_drop_layer,
                                  prev_classifier=prev_classifier, state=state,
                                  relation_index=relation_index,encoder_FOMAML=encoder_FOMAML,droplayer_FOMAML=droplayer_FOMAML,classifier_FOMAML=classifier_FOMAML)

            for relation in current_relations:
                memorized_samples[relation] = select_data(config, encoder_FOMAML, droplayer_FOMAML, training_data[relation])
            train_data_for_memory = []
            # if steps!=0:
            #     train_data_for_memory += expanded_prev_samples
            train_data_for_memory += prev_samples
            for relation in current_relations:
                train_data_for_memory += memorized_samples[relation]

            print(f"输出从运行开始到现在的关系：{len(seen_relations)}")
            print(f"输出要训练的记忆体的数量：{len(train_data_for_memory)}")
            LCD = train_mem_model_1(config=config, encoder=encoder_FOMAML, drop_layer=droplayer_FOMAML, classifier=classifier_FOMAML,
                                    training_data=train_data_for_memory, epochs=config.step3_epochs,
                                    map_relid2tempid=map_relid2tempid, prev_graph_encoder=prev_graph_encoder,
                                    prev_graph_layer=prev_graph_layer,
                                    prev_graph_classifier=prev_graph_classifier,
                                    prev_relation_index=prev_relation_index,
                                    prev_graph_decencoder=prev_graph_descencoder,
                                    new_relation_data=new_relation_data, prev_encoder=prev_encoder,
                                    prev_drop_layer=prev_drop_layer,
                                    prev_classifier=prev_classifier, state=state)

            print(f"memory finished")

            test_data_1 = []
            for relation in current_relations:
                test_data_1 += test_data[relation]

            test_data_2 = []
            for relation in seen_relations:
                test_data_2 += historic_test_data[relation]
            history_data.append(test_data_1)

            print(f"输出当前测试集的长度：{len(test_data_1)}")
            print(f"输出到当前任务流的测试集总长度：{len(test_data_2)}")

            cur_acc = evaluate_strict_model(config, encoder_FOMAML, droplayer_FOMAML, classifier_FOMAML,
                                            test_data_1, seen_relations, map_relid2tempid, rel2id)
            total_acc = evaluate_strict_model(config, encoder_FOMAML, droplayer_FOMAML, classifier_FOMAML,
                                              test_data_2, seen_relations, map_relid2tempid, rel2id)

            accuracy = []
            temp_rel2id = [rel2id[x] for x in history_relations]
            map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
            for data in history_data:
                # accuracy.append(
                #     evaluate_strict_model(config, encoder, classifier, data, history_relations, map_relid2tempid))
                accuracy.append(evaluate_strict_model(config, encoder_FOMAML, droplayer_FOMAML, classifier_FOMAML, data, history_relations,
                                                      map_relid2tempid, rel2id))
            test_cur.append(cur_acc)
            test_total.append(total_acc)
            prev_graph_encoder = deepcopy(encoder1)
            prev_encoder = deepcopy(encoder_FOMAML)

            prev_graph_descencoder = deepcopy(descencoder)
            prev_graph_layer = deepcopy(graphlayer)
            prev_drop_layer = deepcopy(droplayer_FOMAML)

            prev_graph_classifier = deepcopy(classifier1)
            prev_classifier = deepcopy(classifier_FOMAML)
            print(f'Restart Num {rou + 1}')
            print(f'task--{steps + 1}:')
            print(f'current test acc:{cur_acc}')
            print(f'history test acc:{total_acc}')
            print(accuracy)

        end_time = time.time()
        result_cur_time.append(np.array(end_time - start_time))
        result_cur_test.append(np.array(test_cur))
        result_whole_test.append(np.array(test_total) * 100)

        avg_result_cur_test = np.average(result_cur_test, 0)
        avg_result_all_test = np.average(result_whole_test, 0)

        std_result_all_test = np.std(result_whole_test, 0)
        avg_result_cur_time = np.average(result_cur_time, 0)
        std_result_all_time = np.std(result_cur_time, 0)
        print("六轮训练的平均执行时间:")
        print(avg_result_cur_time)
        print("六轮运行的时间标准差")
        print(std_result_all_time)

        print("result_whole_test")
        print(result_whole_test)
        print("avg_result_cur_test")
        print(avg_result_cur_test)
        print("avg_result_all_test")
        print(avg_result_all_test)
        print("std_result_all_test")
        print(std_result_all_test)
        accuracy = []
        temp_rel2id = [rel2id[x] for x in history_relations]
        map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
        for data in history_data:
            accuracy.append(evaluate_strict_model(config, encoder, droplayer, classifier, data, history_relations,
                                                  map_relid2tempid, rel2id))
        print(accuracy)
        bwt = 0.0
        for k in range(len(accuracy) - 1):
            bwt += accuracy[k] - test_cur[k]
        bwt /= len(accuracy) - 1
        bwt_whole.append(bwt)
        fwt_whole.append(np.average(np.array(forward_accs)))
        print("bwt_whole")
        print(bwt_whole)
        print("fwt_whole")
        print(fwt_whole)
        avg_bwt = np.average(np.array(bwt_whole))
        print("avg_bwt_whole")
        print(avg_bwt)
        avg_fwt = np.average(np.array(fwt_whole))
        print("avg_fwt_whole")
        print(avg_fwt)


if __name__ == '__main__':
    config = Config("config.ini")
    main()
    # model = graph_layer(config)
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # model_1 = Dropout_Layer(config)
    # total_params_1 = sum(p.numel() for p in model_1.parameters() if p.requires_grad)
    # model_2 = Softmax_Layer(input_size=768, num_class=80)
    # total_params_2 = sum(p.numel() for p in model_2.parameters() if p.requires_grad)
    # model_3 = Softmax_Layer1(input_size=768, num_class=80)
    # total_params_3 = sum(p.numel() for p in model_3.parameters() if p.requires_grad)
    # model_4 = Bert_Encoder(config)
    # total_params_4 = sum(p.numel() for p in model_4.parameters() if p.requires_grad)
    # model_5 = Bert_Encoder1(config)
    # total_params_5 = sum(p.numel() for p in model_5.parameters() if p.requires_grad)
    # print(f"关系信息中间层参数数量: {total_params}")
    # print(f"本身中间层参数数量: {total_params_1}")
    # print(f"分类器总参数数量: {total_params_2}")
    # print(f"关系信息分类器总参数数量: {total_params_3}")
    # print(f"提取器总参数数量: {total_params_4}")
    # print(f"关系信息提取器总参数数量: {total_params_5}")