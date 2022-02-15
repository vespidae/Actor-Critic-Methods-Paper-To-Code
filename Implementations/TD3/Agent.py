#!/usr/bin/env python
# coding: utf-8

# In[1]:
# import torch
import torch as t
# import torch.nn as nn
import torch.nn.functional as f
# import torch.optim as optim
import numpy as np
# import Networks
from Networks import ActorNetwork
from Networks import CriticNetwork
# import Buffer
from Buffer import ReplayBuffer


# In[2]:


# agent
class Agent:
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

        self.actor_online = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions, "actor_online")
        self.critic_online_one = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions,
                                               "critic_online_one")
        self.critic_online_two = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions,
                                               "critic_online_two")

        self.actor_target = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions, "actor_target")
        self.critic_target_one = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions,
                                               "critic_target_one")
        self.critic_target_two = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions,
                                               "critic_target_two")

        self.memory = ReplayBuffer(buffer_size, input_dims, n_actions)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step > self.warmup:
            state = t.tensor(observation, dtype=t.float).to(self.actor_online.device)
            mu = self.actor_online.forward(state).to(self.actor_online.device)
        else:
            mu = t.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)))

        # exploratory noise
        mu_prime = mu + t.tensor(np.random.normal(scale=self.noise),
                                 dtype=t.float).to(self.actor_online.device)
        # clamp actions to acceptable action space (i.e. to within maximum and minimum bounds)
        mu_prime = t.clamp(mu_prime, self.min_action[0], self.max_action[0])

        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        null_states, actions, rewards, prime_states, terminal = self.memory.sample_replay(self.batch_size)
        null_states = t.tensor(null_states, dtype=t.float).to(self.critic_online_one.device)
        actions = t.tensor(actions, dtype=t.float).to(self.critic_online_one.device)
        rewards = t.tensor(rewards, dtype=t.float).to(self.critic_online_one.device)
        prime_states = t.tensor(prime_states, dtype=t.float).to(self.critic_online_one.device)
        terminal = t.tensor(terminal).to(self.critic_online_one.device)

        # for n, a, r, p, t in zip(null_states, actions, rewards, prime_states, terminal):
        # Q = min(self.critic_online_one(null_states, actions), self.critic_online_two(null_states, actions))
        variance = t.clamp(t.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        a_prime_raw = self.actor_target(prime_states)
        a_prime_unbounded = a_prime_raw + variance
        a_prime = t.clamp(a_prime_unbounded, self.min_action[0], self.max_action[0])

        q_prime_one = self.critic_target_one(prime_states, a_prime)
        q_prime_two = self.critic_target_two(prime_states, a_prime)

        q_prime_one[terminal] = 0.0
        q_prime_two[terminal] = 0.0

        q_prime_one = q_prime_one.view(-1)
        q_prime_two = q_prime_two.view(-1)

        q_prime = t.min(q_prime_one, q_prime_two)
        y = rewards + self.gamma * q_prime
        y = y.view(self.batch_size, 1)

        self.critic_online_one.optimizer.zero_grad()
        self.critic_online_two.optimizer.zero_grad()

        critic_one_loss = f.mse_loss(y, q_prime_one)
        critic_two_loss = f.mse_loss(y, q_prime_two)
        critic_loss = critic_one_loss + critic_two_loss
        critic_loss.backward()

        self.critic_online_one.optimizer.step()
        self.critic_online_two.optimizer.step()

        self.learn_step_cntr += 1

        if self.memory.mem_cntr % 2 == 0:
            self.actor_online.optimizer.zero_grad()
            total_actor_loss = -self.critic_online_one(null_states, self.actor_online(null_states))
            mean_actor_loss = t.mean(total_actor_loss)
            mean_actor_loss.backward()
            self.actor_online.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # update target actor parameters
        phi_null = self.actor_online.state_dict()
        phi_prime = self.actor_target.state_dict()
        for param in phi_null.keys():
            phi_prime[param] = tau * phi_null[param].clone() + \
                                    (1 - tau) * phi_prime[param].clone()

        # update target critic 1 parameters
        theta_null_one = self.critic_online_one.state_dict()
        theta_prime_one = self.critic_target_one.state_dict()
        for param in theta_null_one.keys():
            theta_prime_one[param] = tau * theta_null_one[param].clone() + \
                                     (1 - tau) * theta_prime_one.clone()

        # update target critic 2 parameters
        theta_null_two = self.critic_online_one.state_dict()
        theta_prime_two = self.critic_target_one.state_dict()
        for param in theta_null_two.keys():
            theta_prime_two[param] = tau * theta_null_two[param].clone() + \
                                     (1 - tau) * theta_prime_two.clone()

        # load changes into models
        self.actor_target.load_state_dict(phi_prime)
        self.critic_target_one.load_state_dict(theta_prime_one)
        self.critic_target_two.load_state_dict(theta_prime_two)

    def save_models(self):
        self.actor_online.save_checkpoint()
        self.critic_online_one.save_checkpoint()
        self.critic_online_two.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic_target_one.save_checkpoint()
        self.critic_target_two.save_checkpoint()

    def load_models(self):
        self.actor_online.load_checkpoint()
        self.critic_online_one.load_checkpoint()
        self.critic_online_two.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic_target_one.load_checkpoint()
        self.critic_target_two.load_checkpoint()
