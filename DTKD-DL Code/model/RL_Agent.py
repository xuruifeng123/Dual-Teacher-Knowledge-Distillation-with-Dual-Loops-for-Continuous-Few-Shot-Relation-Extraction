# Author:徐睿峰
# -*- codeing: utf-8 -*-
# @time ：2024/5/15 12:31
# @Author :xuruifeng
# @Site : 
# @file : RL_Agent.py
# @Sofeware : PyCharm


import gym
from gym.spaces import Discrete
import random
import itertools
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
alpha=0.5
choice_weight = [0.35, 0.65]
class RL_Env (gym.Env):
    def __init__(self,config):

        self.config=config
        self.temp_values = self.config.temp_options
        self.weight_values = self.config.weight_options
        self.action_space = [self.temp_values, self.weight_values]
        self.action_space_choice = Discrete(len(self.action_space))
        self.pos={"input":None,"teacher":None,"student":None,"T":self.temp_values[0],"T_index":0,"W":self.weight_values[0],"W_index":0,"iter":0}
        self.final_pos=self.config.task_length
    def step(self,act,tag,current_data,teacher,student):
        if self.pos["iter"] == self.final_pos:
            return self.pos ,True
        if tag == False:
            self.pos["iter"] += 1
            self.pos["input"]=current_data
            self.pos["teacher"]=teacher
            self.pos["student"]=student
            return self.pos,False
        if act==0:
            step=random.choice(self.temp_values)
            while step==self.pos["T"]:
                step = random.choice(self.temp_values)
            self.pos["T"]=step
            self.pos["T_index"]=self.temp_values.index(step)
            self.pos["iter"] +=1
            self.pos["input"] = current_data
            self.pos["teacher"] = teacher
            self.pos["student"] = student
            return self.pos, False
        elif act ==1:
            step = random.choice(self.weight_values)
            while step == self.pos["T"]:
                step = random.choice(self.weight_values)
            self.pos["W"] = step
            self.pos["W_index"]=self.weight_values.index(step)
            self.pos["iter"] += 1
            self.pos["input"] = current_data
            self.pos["teacher"] = teacher
            self.pos["student"] = student
            return self.pos, False
        else:
            raise ValueError("act is wrong")

    def reward(self, accuracy=None,loss=None):
        if accuracy:
            return alpha*(-loss)+(1-alpha)*accuracy*100
        else:
            return alpha*(-loss)

    def reset(self):
        self.pos = {"input":None,"teacher":None,"student":None,"T":self.temp_values[0],"T_index":0,"W":self.weight_values[0],"W_index":0,"iter":0}
        return self.pos


class Agent(object):
    def __init__(self, env):
        self.act_len = env.action_space_choice.n
        self.trajectory_len = env.final_pos
        self.pi = torch.tensor(np.zeros(self.trajectory_len, dtype=int))
        self.value_n = torch.tensor(np.zeros((self.trajectory_len, self.act_len)),)
        self.value_q = torch.tensor(np.zeros((self.trajectory_len, self.act_len)),)
        self.gamma = 0.8
        self.act=[0,1]

    def play(self, state, epsilon=0.0):
        # epsilon represents the exploration probability.
        # If within the epsilon coverage range, a random action will be returned (representing exploration);
        # otherwise, the currently known optimal strategy will be returned
        if np.random.rand() < epsilon:
            while True:
                act1=np.random.randint(self.act_len)
                act2=np.random.choice(self.act,p=choice_weight)
                if act2 == act1:
                    return act2,True
        else:
            return self.pi[state["iter"]].item(),False


def policy_improve(agent):
    new_policy=torch.zeros_like(agent.pi)
    for i in range (1,agent.trajectory_len):
        new_policy[i]=(agent.value_q[i,:]).argmax()
    if torch.all(torch.eq(new_policy, agent.pi)):
        return False
    else:
        agent.pi = new_policy
        return True





# decision_dict = {
#             0: self.action_space[0],
#             1: self.action_space[1],
#             2: self.action_space[2],
#             3: self.action_space[3],
#             4: self.action_space[4],
#             5: self.action_space[5],
#             6: self.action_space[6],
#             7: self.action_space[7],
#             8: self.action_space[8]
#
#         }
#         if self.pos["iter"]==self.final_pos:
#             return self.pos ,True
#         if tag==False:
#             self.pos["iter"] += 1
#             self.pos["input"]=current_data
#             self.pos["teacher"]=teacher
#             self.pos["student"]=student
#             return self.pos,False
#
#         for i in decision_dict:
#             if act ==i:
#                 step=decision_dict[i]
#                 self.pos["T"]=step[0]
#                 self.pos["T_index"] = self.temp_values.index(step[0])
#                 self.pos["W"]=step[1]
#                 self.pos["W_index"] = self.weight_values.index(step[1])
#                 self.pos["iter"] += 1
#                 self.pos["input"] = current_data
#                 self.pos["teacher"] = teacher
#                 self.pos["student"] = student
#                 return self.pos, False