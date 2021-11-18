#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np

# m = torch.Tensor([2.2,4.0],)
# print(m)


# In[2]:


## Initialize policy to be evaluated
class VAgent():
    def __init__(self, gamma=0.99):
        #discount factor
        self.gamma = gamma
        #estimates of states' values (total discounted rewards: https://datascience.stackexchange.com/questions/9832/what-is-the-q-function-and-what-is-the-v-function-in-reinforcement-learning)
        self.V = {}
        #state/action spaces
        self.state_spaces = {
            #possible sums of cards
            "sum":[i for i in range(4,22)],
            #possible cards in dealer's hand
            "dealer_show_card":[i+1 for i in range(10)],
            #useable ace?
            "ace_eleven":[False,True],
            # #hit (or stay)?
            # "hit":[0,1]
        }
        #hit (or stay)?
        self.action_space = [0,1]
        #combinations of parameters
        self.states = []
        #state returns
        self.rewards = {}
        #has agent visited state before?
        self.states_visited = {}
        #states already encountered, returns already received
        self.memory = []
        
        self.init_vals()
        
    def init_vals(self):
        for total in self.state_spaces["sum"]:
            for card in self.state_spaces["dealer_show_card"]:
                for ace in self.state_spaces["ace_eleven"]:
                    for hit in self.action_space:
                        self.V[(total, card, ace)] = 0
                        self.rewards[(total, card, ace)] = []
                        self.states_visited[(total, card, ace)] = False
                        self.states.append((total, card, ace))
                    
    def policy(self, state):
        total, _, _ = state
        #stay if under 21, otherwise hit
        action = 0 if total >= 20 else 1
        return action
        
    def update_V(self):
        for idt, (state, _) in enumerate(self.memory):
            G = 0
            if not self.states_visited[state]:
                self.states_visited[state] = True
                #initialize discount factor, k, for gamma^k
                discount = 1
                
                for t, (_, reward) in enumerate(self.memory[idt:]):
                    G += reward * discount
                    discount *= self.gamma
                    self.rewards[state].append(G)
                    
        for state,_ in self.memory:
            self.V[state] = np.mean(self.rewards[state])
            
        for state in self.states:
            self.states_visited[state] = False
            
        self.memory = []


# In[3]:


## Initialize policy to be evaluated
class QAgent():
    def __init__(self, epsilon=0.1, gamma=0.99):
        #proportion of time agent will take off-greedy action
        self.epsilon = epsilon
        #discount factor
        self.gamma = gamma
        #policy per state
        self.policy = {}
        #estimates of actions' values (expected return per action)
        self.Q = {}
        #state/action spaces
        self.state_subspaces = {
            #possible sums of cards
            "sum":[i for i in range(4,22)],
            #possible cards in dealer's hand
            "dealer_show_card":[i+1 for i in range(10)],
            #useable ace?
            "ace_eleven":[False,True],
            # #hit (or stay)?
            # "hit":[0,1]
        }
        #hit (or stay)?
        self.action_space = [0,1]
        #combinations of states
        self.state_space = []
        #action returns
        self.returns = {}
        #has agent visited state-action pair before?
        self.pairs_visited = {}
        #pairs already encountered, returns already received
        self.memory = []
        
        self.init_vals()
        self.init_policy()
        
    def init_vals(self):
        for total in self.state_subspaces["sum"]:
            for card in self.state_subspaces["dealer_show_card"]:
                for ace in self.state_subspaces["ace_eleven"]:
                    state = (total, card, ace)
                    self.state_space.append(state)
                    for action in self.action_space:
                        self.Q[(state, action)] = 0
                        self.returns[(state, action)] = []
                        self.pairs_visited[(state, action)] = False
                    
    def init_policy(self):
        policy = {}
        n = len(self.action_space)
        for state in self.state_space:
            policy[state] = [1/n for _ in range(n)]
        self.policy = policy
        
    def choose_action(self, state):
        action = np.random.choice(self.action_space, p=self.policy[state])
        return action
        
    def update_Q(self):
        for idt, (state, action, _) in enumerate(self.memory):
            G = 0
            #initialize discount factor, k, for gamma^k
            discount = 1
            
            if not self.pairs_visited[(state, action)]:
                self.pairs_visited[(state, action)] = True
                
                for t, (_, _, reward) in enumerate(self.memory[idt:]):
                    G += reward * discount
                    discount *= self.gamma
                    self.returns[(state, action)].append(G)
                    
        for state, action, _ in self.memory:
            self.Q[(state, action)] = np.mean(self.returns[(state, action)])
            self.update_policy(state)
            
        for state_action in self.pairs_visited.keys():
            self.pairs_visited[state_action] = False        
            
        self.memory = []
        
    def update_policy(self, state):
        actions = [self.Q[(state, a)] for a in self.action_space]
        a_max = np.argmax(actions)
        n_actions = len(self.action_space)        
        probabilities = []
        
        for action in self.action_space:
            probability = 1 - self.epsilon + self.epsilon / n_actions if action == a_max else self.epsilon / n_actions
            probabilities.append(probability)
        self.policy[state] = probabilities


# In[4]:


#main
import gym
import matplotlib.pyplot as plt


# In[5]:


episodes = 200000
win_lose_draw = {-1:0, 0:0, 1:0}
win_rates = []

agent = QAgent(epsilon = 0.001)
env = gym.make('Blackjack-v1')

#traverse episodes
for i in range(episodes):
    if i > 0 and i % 1000 == 0:
        pct = win_lose_draw[1] / i
        win_rates.append(pct)
    if i % 50000 == 0:
        rates = win_rates[-1] if win_rates else 0.0
        print("Starting episode {}: win rate = {}".format(i, rates))
        
        
    #initialize state
    state_null = env.reset()
    done = False
    
    while not done:
        #get action, a, from state
        action = agent.choose_action(state_null)
        #get reward, q, from state action
        state_prime, reward, done, info = env.step(action)
        #store results
        agent.memory.append((state_null, action, reward))
        #move on to next state
        state_null = state_prime
        
    #update value function
    agent.update_Q()
    
    win_lose_draw[reward] += 1
    
plt.plot(win_rates)
plt.show()

# print(agent.V[(21, 3, True)])
# print(agent.V[(4, 1, False)])


# In[ ]:




