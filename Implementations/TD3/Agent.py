#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import Networks
from Networks import ActorNetwork
from Networks import CriticNetwork
import Buffer
from Buffer import ReplayBuffer


# In[2]:


# agent
class Agent():
    def __init__(self, alpha, beta, input_dims, n_actions, tau, env, gamma=0.99,
                 layer1_size=400, layer2_size=300, buffer_size=1e6, batch_size=100,
                 udate_actor_interval=2, warmup=9999, max_size=10e3,
                 noise=0.1):
        #         self.alpha = alpha
        #         self.beta = beta
        self.n_actions = n_actions
        self.gamma = gamma
        #         self.layer1_size = layer1_size
        #         self.layer2_size = layer2_size
        self.batch_size = batch_size
        self.tau = tau
        self.udate_actor_interval = udate_actor_interval
        self.warmup = warmup
        self.max_size = max_size
        self.noise = noise
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.learn_step_cntr = 0
        self.time_step = 0  # countdown to end of warmup period
        self.warmup = warmup

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions, "actor")
        self.critic_one = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions, "critic_one")
        self.critic_two = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions, "critic_two")

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions, "target_actor")
        self.target_critic_one = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions,
                                               "target_critic_one")
        self.target_critic_two = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions,
                                               "target_critic_two")

        self.memory = ReplayBuffer(buffer_size, input_dims, n_actions)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step > self.warmup:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        else:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)))

        # exploratory noise
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise),
                                 dtype=T.float).to(self.actor.device)
        # clamp actions to acceptable action space (i.e. to within maximum and minimum bounds)
        mu_prime = T.clap(mu_prime, self.min_action[0], self.max_action[0])

        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
