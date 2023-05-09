import random
import time
from collections import namedtuple, deque

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from torch.nn import init

# Hyperparameters
lr = 0.0001
gamma = 0.99
iteration = 200
EPISODES = 1000

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(torch.cuda.is_available())

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((torch.FloatTensor(np.array(state)),
                            torch.FloatTensor(np.array(action)),
                            torch.tensor(np.array([reward])),
                            torch.FloatTensor(np.array(next_state)),
                            torch.tensor(np.array([done]))))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.dropout = nn.Dropout(0.1)

        self.afc1 = nn.Linear(state_dim, 64)
        self.afc2 = nn.Linear(64, 128)
        self.afc3 = nn.Linear(128, 64)
        self.afc4 = nn.Linear(64, 32)
        self.action = nn.Linear(32, action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.afc1(state))
        x = F.relu(self.afc2(x))
        x = F.relu(self.afc3(x))
        x = F.relu(self.afc4(x))
        action = torch.tanh(self.action(x)) * self.max_action
        action = torch.round(action) # integer between -10 and 10
        return action


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.dropout = nn.Dropout(0.1)

        self.vfc1 = nn.Linear(state_dim + action_dim, 64)
        self.vfc2 = nn.Linear(64, 128)
        self.vfc3 = nn.Linear(128, 64)
        self.vfc4 = nn.Linear(64, 32)
        self.value = nn.Linear(32, 1)

    def forward(self, state, action):
        x = F.relu(self.vfc1(torch.cat([state, action], 1)))
        x = F.relu(self.vfc2(x))
        x = F.relu(self.vfc3(x))
        x = F.relu(self.vfc4(x))
        value = self.value(x)
        return value

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * self.mu
        self.reset()
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

class DDPG(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(DDPG, self).__init__()
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(device)
        self.actor_target = ActorNetwork(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.action_dim = action_dim
        self.max_action = max_action
        self.noise = OUNoise(action_dim)
        self.memory = ReplayMemory(100000)

    def select_action(self, state):
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        noise = self.noise.sample() # noise will be removed during execution
        return np.clip(action + noise, -self.max_action, self.max_action) # OUnoise

    def train_net(self, iterations, batch_size = 64, discount = 0.99, tau = 0.001):
        if len(self.memory) < batch_size:
            return

        for _ in range(iterations):
            # print(self.memory.sample(1))
            batch = self.memory.sample(batch_size)
            states, actions, rewards, next_states, done = zip(*batch)
            s =  torch.stack(states)
            a = torch.stack(actions)
            r = torch.stack(rewards)
            s_prime = torch.stack(next_states)
            done = torch.stack(done)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            # Critic loss
            current_Q = self.critic(s, a)
            target_Q = self.critic_target(s_prime, self.actor_target(s_prime))
            target_Q = r + (done * discount * target_Q).detach()
            critic_loss = F.smooth_l1_loss(target_Q, current_Q) # advantages.pow(2).mean()

            # Actor loss
            actor_loss = -self.critic(s, self.actor(s)).mean()

            critic_loss.backward()
            actor_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def main():
    env = gym.make('MountainCarContinuous-v0', render_mode="human")
    # print(env.observation_space.shape[0], env.action_space)
    model = DDPG(env.observation_space.shape[0], action_dim=1, max_action= 1)
    print_interval = 20
    print_score = 0.0
    scores = []
    reward_sum = 0

    for n_epi in range(EPISODES):
        step = 0
        done = False
        s, _ = env.reset()
        while not done:
            for t in range(iteration):
                a = model.select_action(torch.from_numpy(s).float().to(device))
                # a = a.cpu().detach().numpy()
                s_prime, r, done, debug, info = env.step(a)
                model.memory.memorize(s, a, r, s_prime, done)
                # print(type(s), type(a), type(r), type(s_prime), type(done))
                # print(s, a, r, s_prime, done)
                s = s_prime

                print_score += r
                reward_sum += r
                step += 1

                if done:
                    print(step)
                    break

                model.train_net(iterations=20)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, print_score / print_interval))
            print_score = 0.0

        scores.append(reward_sum)
        reward_sum = 0

    env.close()

    plt.figure(figsize=(10,6))
    plt.plot(scores)
    plt.plot(pd.Series(scores).rolling(100).mean())
    plt.title('DDPG_continuous')
    plt.xlabel('# of episodes')
    plt.ylabel('score')
    plt.savefig('DDPG_continuous.png')

    torch.save(model.state_dict(), 'DDPG_continuous.pth')

if __name__ == '__main__':
    main()
