#!/usr/bin/env python
# coding: utf-8

# In[1]:
import torch
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
                 update_actor_interval=2, warmup=9999, max_size=10e3,
                 noise=0.1):
        #         self.alpha = alpha
        #         self.beta = beta
        self.n_actions = n_actions
        self.gamma = gamma
        #         self.layer1_size = layer1_size
        #         self.layer2_size = layer2_size
        self.batch_size = batch_size
        self.tau = tau
        self.update_actor_interval = update_actor_interval
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

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        null_states, actions, rewards, prime_states, terminal = self.memory.sample_replay(self.batch_size)
        null_states = T.tensor(null_states, dtype=T.float).to(self.critic_one.device)
        actions = T.tensor(actions, dtype=T.float).to(self.critic_one.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.critic_one.device)
        prime_states = T.tensor(prime_states, dtype=T.float).to(self.critic_one.device)
        terminal = T.tensor(terminal).to(self.critic_one.device)

        # for n, a, r, p, t in zip(null_states, actions, rewards, prime_states, terminal):
        # Q = min(self.critic_one(null_states, actions), self.critic_two(null_states, actions))
        variance = T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        a_prime_raw = self.target_actor(prime_states)
        a_prime_unbounded = a_prime_raw + variance
        a_prime = T.clamp(a_prime_unbounded, self.min_action[0], self.max_action[0])

        Q_one_prime = self.target_critic_one(prime_states, a_prime)
        Q_two_prime = self.target_critic_two(prime_states, a_prime)

        Q_one_prime[terminal] = 0.0
        Q_two_prime[terminal] = 0.0

        Q_one_prime = Q_one_prime.view(-1)
        Q_two_prime = Q_two_prime.view(-1)

        Q_prime = T.min(Q_one_prime, Q_two_prime)
        y = rewards + self.gamma * Q_prime
        y = y.view(self.batch_size, 1)

        self.critic_one.optimizer.zero_grad()
        self.critic_two.optimizer.zero_grad()

        critic_one_loss = F.mse_loss(y, Q_one_prime)
        critic_two_loss = F.mse_loss(y, Q_two_prime)
        critic_loss = critic_one_loss + critic_two_loss
        critic_loss.backward()

        self.critic_one.optimizer.step()
        self.critic_two.optimizer.step()

        self.learn_step_cntr += 1

        if self.memory.mem_cntr % 2 == 0:
            self.actor.optimizer.zero_grad()
            total_actor_loss = -self.critic_one(null_states, self.actor(null_states))
            mean_actor_loss = T.mean(total_actor_loss)
            mean_actor_loss.backward()
            self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        phi_null = self.actor.state_dict()
        phi_prime = self.target_actor.state_dict()
        for param in phi_null.keys():
            phi_prime[param] = tau * phi_null[param].clone() + \
                                    (1 - tau) * phi_prime[param].clone()
        self.target_actor.load_state_dict(phi_prime)

        theta_null_one = self.critic_one.state_dict()
        theta_prime_one = self.target_critic_one.state_dict()
        for param in theta_null_one.keys():
            theta_prime_one[param] = tau * theta_null_one[param].clone() + \
                                     (1 - tau) * theta_prime_one
        self.target_critic_one.load_state_dict(theta_prime_one)

        theta_null_two = self.critic_one.state_dict()
        theta_prime_two = self.target_critic_one.state_dict()
        for param in theta_null_two.keys():
            theta_prime_two[param] = tau * theta_null_two[param].clone() + \
                                     (1 - tau) * theta_prime_two
        self.target_critic_two.load_state_dict(theta_prime_two)

