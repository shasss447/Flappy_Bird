import torch
import numpy as np
import random
from itertools import count
import pygame as pg
import matplotlib.pyplot as plt
import torch.optim as optim
import math
import memory_recall

class Agent():
    def __init__(self, BATCH_SIZE, MEMORY_SIZE, GAMMA, input_dim, output_dim, action_dim, action_dict, EPS_START, EPS_END, EPS_DECAY_VALUE, lr, TAU, network_type='DDQN') -> None:
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.MEMORY_SIZE=MEMORY_SIZE
        self.action_dim = action_dim
        self.action_dict = action_dict
        self.EPS_START=EPS_START
        self.EPS_END=EPS_END
        self.EPS_DECAY_VALUE=EPS_DECAY_VALUE
        self.eps = EPS_START
        self.TAU = TAU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.episode_durations = []
        self.cache_recall = memory_recall.MemoryRecall(memory_size=MEMORY_SIZE)
        self.network_type = network_type

        
        @torch.no_grad()
        def take_action(self, state):
         self.eps = self.eps*self.EPS_DECAY_VALUE
         self.eps = max(self.eps, self.EPS_END)
         if self.eps < np.random.rand():
            state = state[None, :]
            action_idx = torch.argmax(self.policy_net(state), dim=1).item()
         else:
            action_idx = random.randint(0, self.action_dim-1)
         self.steps_done += 1
         return action_idx