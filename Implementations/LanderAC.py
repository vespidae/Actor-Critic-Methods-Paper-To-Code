#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym

import matplotlib.pyplot as plt


# In[ ]:


class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dim, n_actions, hlOne=256, hlTwo=256):
        super(ActorCriticNetwork, self).__init__()
        self.input_layer = nn.Linear(*input_dim,hlOne)
        self.hidden_layer = nn.Linear(hlOne,hlTwo)
        # policy functio
        self.pi = nn.Linear(hlTwo, n_actions)
        # value function
        self.V = nn.Linear(hlTwo, 1)
        # optimizer
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        # device
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
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hidden_layer(x))
        
        pi = self.pi(x)
        V = self.V(x)
        
        return (pi, V)


# In[ ]:


class Agent():
    def __init__(self, lr, gamma, input_dims, n_actions=4, hlOne=256, hlTwo=256):
        # set hyperparameters
        self.lr, self.gamma = lr, gamma
        
        # # set memory
        # self.action_memory, self.reward_memory = [], []
        
        self.hlOne, self.hlTwo = hlOne, hlTwo
        
        self.actor_critic = ActorCriticNetwork(lr, input_dims, n_actions, hlOne, hlTwo)
        
        self.log_prob = None
        
    def choose_action(self, observation):
        state = torch.Tensor([observation])
        state = self.actor_critic.route_data(state)        
        actor_policy, _ = self.actor_critic.forward(state)
        
        # get action from output
        actor_policy = F.softmax(actor_policy, dim=1)
        action_probs = torch.distributions.Categorical(actor_policy)
        action = action_probs.sample()
        
        # prepare loss profile
        self.log_prob = action_probs.log_prob(action)
        
        return action.item()
    
    def learn(self, state_null, reward, state_prime, done):
        state_null = self.actor_critic.route_data(torch.Tensor([state_null]))
        reward = self.actor_critic.route_data(torch.Tensor([reward]))
        state_prime = self.actor_critic.route_data(torch.Tensor([state_prime]))
        
        self.actor_critic.optimizer.zero_grad()
        
        _, critic_value_null = self.actor_critic.forward(state_null)
        _, critic_value_prime = self.actor_critic.forward(state_prime)
        
        delta = reward + self.gamma*critic_value_prime*(1 - int(done))
        
        actor_loss = -self.log_prob*delta
        critic_loss = delta**2
        
        (actor_loss + critic_loss).backward()
        
        self.actor_critic.optimizer.step()


# In[ ]:


def plot_learning_curve(scores, x, figure_file):
    # running_avg = np.zeros_like(scores)
    running_avg = np.zeros(len(scores))
    
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title("Running Average of Previous 100 Scores")
    plt.savefig(figure_file)


# In[ ]:


n_games = 2000
lr = 5e-6
gamma = 0.99

env = gym.make('LunarLander-v2')
agent = Agent(lr, gamma, [8], 4, 2048, 1536)

figure_name = 'ACTOR_CRITIC-' + 'lunar_lander-%s' % str(agent.hlOne) + 'time%s' % str(agent.hlTwo) +     '-lr_%s' % str(agent.lr)  + '-' + str(n_games) + '_games'
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
        
        agent.learn(obs_null, reward, obs_prime, done)
        
        obs_null = obs_prime
    
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print("Episode: {}\tScore: {}\t\tAverage Score: {}".format(i,score,avg_score))
    
x = [i+1 for i in range(len(scores))]
plot_learning_curve(scores, x, figure_file)