#!/usr/bin/env python
# coding: utf-8

# !pip3 install box2d-py
# !pip3 install gym[Box_2D]
# !pip install matplotlib

# In[1]:


import gym
import matplotlib.pyplot as plt


# env = gym.make('LunarLander-v2')
# 
# n_games = 100
# 
# for i in range(n_games):
#     obs_nul = env.reset()
#     score = 0
#     done = False
#     while not done:
#         action = env.action_space.sample()
#         obs_prime, reward, done, info = env.step(action)
#         score += reward
#         # env.render()
#     print("Episode: {}\t Reward:{}".format(i,score))

# In[2]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[ ]:





# In[3]:


class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.device = [torch.device('cpu')]
        if torch.cuda.device_count() > 0:
            self.device = []
            for d in range(torch.cuda.device_count()):
                self.device.append(torch.device('cuda:%s' % d))
        self.target_device = 1
        # self.to(self.device[0])
        self.route_data(self)
        ## print('self.device: %s' % self.device)
        
    def route_data(self, data):
        ## print('self.target_device: %s' % self.target_device)
        moved_data = data.to(self.device[self.target_device])
        # self.target_device = 0 if self.target_device >= len(self.device) else self.target_device + 1
        
        return moved_data
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# In[4]:


class PolicyGradientAgent():
        def __init__(self, lr, input_dims, gamma=0.99, n_actions=4):
            # hyperparameters
            self.lr, self.gamma = lr, gamma

            # memory
            self.reward_memory = []
            self.action_memory = []

            # policy
            self.policy = PolicyNetwork(lr, input_dims, n_actions)
        
        def choose_action(self, observation):
            # convert observation to tensor on device
            state = torch.Tensor([observation])
            state = self.policy.route_data(state)
            
            # activation function
            probabilities = F.softmax(self.policy.forward(state),1)
            
            # convert softmax (i.e. calculated probability distribution) to categorical distribution from which we can apply sampling functions
            action_probs = torch.distributions.Categorical(probabilities)
            
            # sample from distribution
            action = action_probs.sample()
            
            # save log probabilities to feed into loss function
            log_probs = action_probs.log_prob(action)            
            self.action_memory.append(log_probs)
            
            return action.item()
        
        def store_rewards(self, reward):
            self.reward_memory.append(reward)
            
        def learn(self):
            self.policy.optimizer.zero_grad()
            
            # calculate time step returns
            G = np.zeros_like(self.reward_memory)
            for t in range(len(self.reward_memory)):
                G_sum = 0
                discount = 1
                for k in range(t, len(self.reward_memory)):
                    G_sum += self.reward_memory[k] * discount
                    discount *= self.gamma
                G[t] = G_sum
            G_raw = torch.Tensor(G)
            G = self.policy.route_data(G_raw)
            
            loss = 0
            for g, logprob in zip(G, self.action_memory):
                loss += -g * logprob
            loss.backward()
            self.policy.optimizer.step()
            
            self.action_memory = []
            self.reward_memory = []


# In[5]:


def plot_learning_curve(scores, x, figure_file):
    # running_avg = np.zeros_like(scores)
    running_avg = np.zeros(len(scores))
    
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title("Running Average of Previous 100 Scores")
    plt.savefig(figure_file)


# In[ ]:


n_games = 3000
gamma=0.99
lr = 0.0005

env = gym.make('LunarLander-v2')
agent = PolicyGradientAgent(gamma=gamma, lr=lr, input_dims=[8], n_actions=4)

figure_name = 'REINFORCE-' + 'lunar_lander_lr_%s' % str(agent.lr) + '-' + str(n_games) + '_games'
figure_file = 'plots/' + figure_name + '.png'

scores = []
for i in range(n_games):
    done = False
    obs_null = env.reset()
    score = 0
    
    while not done:
        action = agent.choose_action(obs_null)
        obs_prime, reward, done, info = env.step(action)
        env.render()
        score += reward
        agent.store_rewards(reward)
        
        obs_null = obs_prime
    agent.learn()
    scores.append(score)
    
    avg_score = np.mean(scores[-100:])
    print("Episode: {}\tScore: {}\t\tAverage Score: {}".format(i,score,avg_score))
    
# x = [i+1 for i in range(len(scores))]
# plot_learning_curve(scores, x, figure_file)

