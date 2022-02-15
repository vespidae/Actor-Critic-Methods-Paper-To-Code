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


# critic network
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, chkpt_name, chkpt_dir='tmp/td3'):
        super(CriticNetwork, self).__init__()
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_name = chkpt_name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.checkpoint_name + '_td3')
        
        # the following implementation is not for 2-D state representations
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state,action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)
        
        q1 = self.q1(q1_action_value)
        
        return q1
    
    def save_checkpoint(self):
        print("Saving critic checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)
        print("Critic checkpoint saved!")
    
    def load_checkpoint(self):
        print("Loading critic checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))
        print("Critic checkpoint loaded!")


# In[5]:


# actor_online network
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, chkpt_name, chkpt_dir='tmp/td3'):
        super(ActorNetwork, self).__init__()
        self.alpha = alpha
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_name = chkpt_name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.checkpoint_name + '_td3')
        
        # the following implementation is not for 2-D state representations
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def forward(self, state):
        action_prob = self.fc1(state)
        action_prob = F.relu(action_prob)
        action_prob = self.fc2(action_prob)
        action_prob = F.relu(action_prob)
        
        a = T.tanh(self.mu(action_prob))
        
        return a
    
    def save_checkpoint(self):
        print("Saving actor_online checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)
        print("Actor checkpoint saved!")
    
    def load_checkpoint(self):
        print("Loading actor_online checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))
        print("Actor checkpoint loaded!")