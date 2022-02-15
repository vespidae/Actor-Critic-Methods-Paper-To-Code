#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


class OUActionNoise():
    # initialize mean, std, 
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, xNull=None):
        self.mu, self.sigma, self.theta, self.dt, self.xNull = mu, sigma, theta, dt, xNull
        self.reset()

    # allows us to use the name of an object as a function
    def __call__(self):
        # get temporal correlation of noise
        x = self.xPrevious + self.theta * (self.mu - self.xPrevious) *         self.dt + self.sigma * np.sqrt(self.dt) + np.random.normal(size=self.mu.shape)
        self.xPrevious = x

        return x

    # set initial value of xPrevious
    def reset(self):
        self.xPrevious = self.xNull if self.xNull is not None else np.zeros_like(self.mu)


# In[ ]:


class ReplayBuffer():
    def __init__(self, max_size, input_shape, action_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.memory = {
            "null_state" : np.array([np.zeros(input_shape) for reg in range(self.mem_size)]),# np.zeros((self.mem_size, *input_shape)),
            "action" : np.zeros((self.mem_size, action_shape)),
            "reward" : np.zeros(self.mem_size),
            "prime_state" : np.array([np.zeros(input_shape) for reg in range(self.mem_size)]),#np.zeros((self.mem_size, *input_shape)),
            "terminal" : np.zeros(self.mem_size, dtype=bool),
        }

        # mask for setting critic values for new state to zero
        # self.term_mem = np.zeros(self.mem_size, dtype=np.bool)


    def store_transition(self, null_state, action, reward, prime_state, done):
        index = self.mem_cntr % self.mem_size
        alignment = zip(["null_state", "action", "reward", "prime_state"],                        [null_state, action, reward, prime_state])

        for mem, value in alignment:
            # print("mem: {}\tself.mem_cntr: {}\tvalue.shape: {}\tself.memory[mem][self.mem_cntr].shape: {}".format(\
            #     mem, self.mem_cntr, value.shape, self.memory[mem][self.mem_cntr].shape))
            # if mem not "null_state" and not "prime_state":
            #     print("Attr: {}, Index: {}, Value: {}".format(mem, self.mem_cntr % self.mem_size, value))
            self.memory[mem][index] = value

        self.mem_cntr += 1

#     def sample_replay(self, proportion):

#         sample_size = np.ceil(len(self.replays) * proportion)
#         return np.random.choice(self.replays, sample_size)

    def sample_replay(self, batch_size):
        picks = {}

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        for mem in ["null_state", "prime_state", "action", "reward", "terminal"]:
            picks[mem] = self.memory[mem][batch]

        return picks["null_state"], picks["action"], picks["reward"], picks["prime_state"], picks["terminal"]


# In[ ]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[ ]:


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, hl_one, hl_two, chkpt_name, chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()

        self.beta, self.input_dims, self.hl_one, self.hl_two, self.n_actions, self.chkpt_name, self.chkpt_dir =         beta, input_dims, hl_one, hl_two, n_actions, chkpt_name, chkpt_dir

        self.chkpt_file = os.path.join(self.chkpt_dir, self.chkpt_name + '_ddpg')

        # define layers
        # self.frame = nn.Conv2d(3,1,3)
        self.flatten = nn.Flatten()
        self.input = nn.Linear(np.prod(self.input_dims),self.hl_one)
        self.hidden = nn.Linear(self.hl_one, self.hl_two)

        # define normalizers
        self.lnormi = nn.LayerNorm(self.hl_one)
        self.lnormh = nn.LayerNorm(self.hl_two)

        # a calculation
        self.action_output = nn.Linear(self.n_actions, self.hl_two)

        # Q calculation
        self.critic_output = nn.Linear(self.hl_two, 1)

        # initialize layers
        for layer in [self.input, self.hidden, self.action_output]:
            fan_in = 1 / np.sqrt(layer.weight.data.size()[0])
            layer.weight.data.uniform_(-fan_in, fan_in)
            layer.bias.data.uniform_(-fan_in, fan_in)

        critic_fan_in = 0.003
        self.critic_output.weight.data.uniform_(-critic_fan_in, critic_fan_in)
        self.critic_output.bias.data.uniform_(-critic_fan_in, critic_fan_in)

        # define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.beta, weight_decay=0.01)

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

    def forward(self, state, action):
        state_value = self.flatten(state)
        state_value = self.lnormi(self.input(state_value))
        state_value = F.relu(state_value)
        state_value = self.lnormh(self.hidden(state_value))

        action_value = self.action_output(action)

        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.critic_output(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print("saving {} checkpoint...".format(self.chkpt_name))
        torch.save(self.state_dict(), self.chkpt_file)
        print("{} checkpoint saved.".format(self.chkpt_name))

    def load_checkpoint(self):
        print("loading {} checkpoint...".format(self.chkpt_name))
        self.load_state_dict(torch.load(self.chkpt_file))
        print("{} checkpoint loaded.".format(self.chkpt_name))


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, hl_one, hl_two, chkpt_name, chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()

        self.alpha, self.input_dims, self.hl_one, self.hl_two, self.n_actions, self.chkpt_name, self.chkpt_dir =         alpha, input_dims, hl_one, hl_two, n_actions, chkpt_name, chkpt_dir

        self.chkpt_file = os.path.join(self.chkpt_dir, self.chkpt_name + '_ddpg')

        # define layers
        # self.frame = nn.Conv2d(3,1,3)
        self.flatten = nn.Flatten()
        self.input = nn.Linear(np.prod(self.input_dims),self.hl_one)
        self.hidden = nn.Linear(self.hl_one, self.hl_two)
        # self.output = nn.Linear(self.hl_two, self.n_actions)

        # define normalizers
        self.lnormi = nn.LayerNorm(self.hl_one)
        self.lnormh = nn.LayerNorm(self.hl_two)

        # define mu
        self.mu = nn.Linear(self.hl_two, self.n_actions)

        # initialize layers
        for layer in [self.input, self.hidden]:
            fan_in = 1 / np.sqrt(layer.weight.data.size()[0])
            layer.weight.data.uniform_(-fan_in,fan_in)
            layer.bias.data.uniform_(-fan_in,fan_in)

        mu_fan_in = 3e-3
        self.mu.weight.data.uniform_(-mu_fan_in, mu_fan_in)
        self.mu.bias.data.uniform_(-mu_fan_in, mu_fan_in)

        # define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)#, weight_decay=1e-4)

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
        # print(state.shape)
        # print(self.frame)
        # print(self.flatten(state).shape)
        # plt.imshow(state[0].cpu())
        x = self.flatten(state)
        x = F.relu(self.lnormi(self.input(x)))
        x = F.relu(self.lnormh(self.hidden(x)))
        A = torch.tanh(self.mu(x))
        # action_value = self.mu(time)

        return A

    def save_checkpoint(self):
        print("saving {} checkpoint...".format(self.chkpt_name))
        torch.save(self.state_dict(), self.chkpt_file)
        print("{} checkpoint saved.".format(self.chkpt_name))

    def load_checkpoint(self):
        print("loading {} checkpoint...".format(self.chkpt_name))
        self.load_state_dict(torch.load(self.chkpt_file))
        print("{} checkpoint loaded.".format(self.chkpt_name))


# In[ ]:


class Agent():
    def __init__(self, alpha, beta, tau, input_dims, n_actions, gamma=0.99, hlOne=400, hlTwo=300, buffer_size=1e6, batch_size=64):
        self.alpha, self.beta, self.gamma, self.tau, self.input_dims, self.n_actions, self.hlOne, self.hlTwo, self.buffer_size, self.batch_size =         alpha, beta, gamma, tau, input_dims, n_actions, hlOne, hlTwo, buffer_size, batch_size
        # for argument,name in zip([self.alpha, self.beta, self.gamma, self.tau, self.input_dims, self.n_actions, self.hlOne, self.hlTwo, self.buffer_size, self.batch_size],["alpha", "beta", "gamma", "tau", "input_dims", "n_actions", "hlOne", "hlTwo", "buffer_size", "batch_size"]):
        #     print("Argument: {}\tName: {}".format(name, argument))

        self.actor = ActorNetwork(self.alpha, self.input_dims, self.n_actions, self.hlOne, self.hlTwo, 'actor_online')
        self.critic = CriticNetwork(self.beta, self.input_dims, self.n_actions, self.hlOne, self.hlTwo, 'critic')
        self.actor_prime = ActorNetwork(self.alpha, self.input_dims, self.n_actions, self.hlOne, self.hlTwo, 'actor_target')
        self.critic_prime = CriticNetwork(self.beta, self.input_dims, self.n_actions, self.hlOne, self.hlTwo, 'target_critic')

        self.buffer = ReplayBuffer(self.buffer_size, self.input_dims, self.n_actions)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        # get policy
        state = torch.tensor(np.array([observation]), dtype=torch.float)
        state = self.actor.route_data(state)
        mu = self.actor(state)
        mu_prime = mu + self.actor.route_data(torch.Tensor(self.noise()))
        self.actor.train()

#         # get action from policy
#         mu = f.softmax(mu, dim=1)
#         action_probs = torch.distributions.Categorical(mu)
#         a = action_probs.sample()

#         self.log_prob = action_probs.log_prob(a)

        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, state_null, action, reward, state_prime, done):
        self.buffer.store_transition(state_null, action, reward, state_prime, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.actor_prime.save_checkpoint()
        self.critic_prime.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.actor_prime.load_checkpoint()
        self.critic_prime.load_checkpoint()

    def learn(self):
        if self.buffer.mem_cntr < self.batch_size:
            return

        null_states, actions, rewards, prime_states, terminal = self.buffer.sample_replay(self.batch_size)
        null_states = self.actor.route_data(torch.tensor(np.array([null_states]), dtype=torch.float))
        actions = self.actor.route_data(torch.tensor(np.array([actions]), dtype=torch.float))
        rewards = self.actor.route_data(torch.tensor(np.array([rewards]), dtype=torch.float))
        prime_states = self.actor.route_data(torch.tensor(np.array([prime_states]), dtype=torch.float))
        # print(terminal)
        terminal = self.actor.route_data(torch.tensor(np.array([terminal])))

        # target_critic_value_null = self.critic_prime(state_null)
        # target_critic_value_prime = self.critic_prime(state_prime)

        for n, a, r, p, t in zip(null_states, actions, rewards, prime_states, terminal):
            Q = self.critic(n, a)
            Q_prime = self.critic_prime(p, self.actor_prime(p))
            # use terminal tensor as a mask to modify respective rewards
            Q_prime[t] = 0.0
            Q_prime = Q_prime.view(-1)

            y = r + self.gamma * Q_prime
            y = y.view(self.batch_size, 1)

            self.actor.optimizer.zero_grad()
            actor_loss = -self.critic(n, self.actor(n))
            actor_loss = torch.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()

            self.critic.optimizer.zero_grad()
            critic_loss = F.mse_loss(y, Q)
            critic_loss.backward()
            self.critic.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        theta_mu = self.actor.state_dict()
        theta_Q = self.critic.state_dict()
        theta_mu_prime = self.actor_prime.state_dict()
        theta_Q_prime = self.critic_prime.state_dict()
        # theta_mu = {name:param for name,param in self.actor_online.named_parameters()}
        # theta_Q = {name:param for name,param in self.critic.named_parameters()}
        # theta_mu_prime = {name:param for name,param in self.actor_prime.named_parameters()}
        # theta_Q_prime = {name:param for name,param in self.critic_prime.named_parameters()}

        for target_network, null_network in [[theta_mu_prime,theta_mu],[theta_Q_prime,theta_Q]]:
            for param in null_network.keys():
                target_network[param] = (tau * null_network[param].clone()) +                 ((1 - tau) * target_network[param].clone())

        self.actor_prime.load_state_dict(theta_mu_prime)
        self.critic_prime.load_state_dict(theta_Q_prime)

#         L = (1 / batch_size) * ((y - Q) ** 2)
#         # target_actor_loss = -self.log_prob*delta
#         # target_critic_loss = delta**2

#         # (target_actor_loss + target_critic_loss).backward()
#         L.backward()

#         self.actor_online.optimizer.step()


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


import gym


# In[ ]:


n_games = 5000
actor_lr =1e-4
critic_lr = 1e-3
soft_target_update = 1e-3
# input_dimensions = [210, 160, 3]
# number_of_actions = 4
discount_factor = 0.99
hlOne = 400
hlTwo = 300
buffer_size = 2e4
minibatch_size = 64
# stuck_seconds = 5

# instantiate environment and agent
# env = gym.make('LunarLanderContinuous-v2') # <- good vizualizations
# env = gym.make('ALE/ElevatorAction-v5')#, render_mode='human')
# env = gym.make('ALE/Adventure-v5')
# env = gym.make('ALE/Asteroids-v5')
# env = gym.make('ALE/ChopperCommand-v5') # <- good vizualizations
# env = gym.make('ALE/Alien-v5') # <- good vizualizations
env = gym.make('ALE/BattleZone-v5') # <- good vizualizations
# print(env.observation_space.shape[0])
action_space_is_box = type(env.action_space) is gym.spaces.box.Box
number_of_actions = env.action_space.shape[0] if action_space_is_box else env.action_space.n
# print(type(env.action_space))
agent = Agent(actor_lr, critic_lr, soft_target_update, env.observation_space.shape,               number_of_actions, discount_factor,               hlOne, hlTwo, int(buffer_size), minibatch_size)
            # alpha, beta, tau, input_dims, n_actions, gamma=0.99, hlOne=400, hlTwo=300, buffer_size=1e6, batch_size=64
# print(agent.actor_online.n_actions)

figure_name = 'ACTOR_CRITIC-' + 'lunar_lander-%s' % str(agent.hlOne) + 'time%s' % str(agent.hlTwo) +     '-alpha_%s' % str(agent.alpha)  + '-beta_%s' % str(agent.beta)  +     '-tau_%s' % str(agent.tau) + '-buffer_%s' % str(agent.buffer_size)  +     '-batch_size_%s' % str(agent.batch_size)  + '-' +     str(n_games) + '_games'
figure_file = 'plots/' + figure_name + '.png'

best_score = env.reward_range[0]
score_history = []

for i in range(n_games):
    # initialize values
    null_obv = env.reset()
    agent.noise.reset()
    score = 0
    done = False

#     stuck_range = stuck_seconds * 60
#     # sticky_situation = [null_obv for i in range(stuck_range)]
#     sticky_situation = np.zeros(stuck_range).tolist()
#     stick_cnt = 0
#     stuck = True

    while not done:
        # # get unstuck if necessary
        # sticky_situation[stick_cnt % stuck_range]
        # stick_cnt += 1
        # for frame in sticky_situation:
        #     if null_obv is not frame: 
        #         stuck = False
        # if stuck: break

        # get action
        choice = agent.choose_action(null_obv)
        # prime_obv, reward, done, info = env.step(action.argmax(0))
        action = choice if action_space_is_box else choice.argmax(0)
        prime_obv, reward, done, info = env.step(action)
        env.render()
        # print(info)
        agent.remember(null_obv, action, reward, prime_obv, done)
        # update score
        score += reward
        # learn
        agent.learn()

        # prep for next (time)step
        # null_obv = prime_obv if info['lives'] > 0 else env.reset()
        if info['lives'] == 0: break
        null_obv = prime_obv

    # manage score
    score_history.append(score)
    running_avg_score = np.mean(score_history[-100:])

    if running_avg_score > best_score:
        best_score = running_avg_score
        agent.save_models()

    print("Episode: {}\t\tScore: {}\t\tAverage Score: {}".format(i, score, running_avg_score))

x = [i+1 for i in range(len(score_history))]
plot_learning_curve(score_history, x, figure_file)


# In[ ]:




