#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym


# In[ ]:


env = gym.make('LunarLander-v2')

n_games = 100

for i in range(n_games):
    obs_nul = env.reset()
    score = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs_prime, reward, done, info = env.step(action)
        score += reward
        env.render()
    print("Episode: {}\t Reward:{}".format(i,score))


# In[ ]:




