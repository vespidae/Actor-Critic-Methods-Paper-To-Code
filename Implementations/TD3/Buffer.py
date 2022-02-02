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