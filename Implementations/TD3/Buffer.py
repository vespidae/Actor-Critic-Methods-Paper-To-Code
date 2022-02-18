#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# In[2]:


# memory
class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.memory = {
            "null_state" : np.zeros((self.mem_size, *input_shape)),
            "action" : np.zeros((self.mem_size, n_actions)),
            "reward" : np.zeros(self.mem_size),
            "prime_state" : np.zeros((self.mem_size, *input_shape)),
            "terminal" : np.zeros(self.mem_size, dtype=np.bool)
        }

    def store_transition(self, null_state, action, reward, prime_state, done):
        index = self.mem_cntr % self.mem_size
        alignment = zip(["null_state", "action", "reward", "prime_state"], [null_state, action, reward, prime_state])

        for mem, value in alignment:
            self.memory[mem][index] = value

        self.mem_cntr += 1

    def sample_replay(self, batch_size):
        picks = {}

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        for mem in ["null_state", "prime_state", "action", "reward", "terminal"]:
            picks[mem] = self.memory[mem][batch]

        return picks["null_state"], picks["action"], picks["reward"], picks["prime_state"], picks["terminal"]