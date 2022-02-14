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
    def __init__(self, max_size, input_shape, action_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.memory = {
            "null_state" : np.array([np.zeros(input_shape) for reg in range(self.mem_size)]),
            "action" : np.zeros((self.mem_size, action_shape)),
            "reward" : np.zeros((self.mem_size)),
            "prime_state" : np.array([np.zeros((input_shape)) for reg in range(self.mem_size)]),
            "terminal" : np.zeros(self.mem_size, dtype=bool)
        }

    def store_transition(self, null_state, action, reward, prime_state, done):
        index = self.mem_cntr % self.mem_size
        alignment = zip(["null_state", "action", "reward", "prime_state"], [null_state, action, reward, prime_state])

        for mem, value in alignment:
            self.memory[mem][index] = value

        self.mem_cntr += 1