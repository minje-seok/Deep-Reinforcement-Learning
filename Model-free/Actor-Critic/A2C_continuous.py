import time
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
EPISODES = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.dropout = nn.Dropout(0.1)

        self.afc1 = nn.Linear(state_dim, 64)
        self.afc2 = nn.Linear(64, 128)
        self.afc3 = nn.Linear(128, 64)
        self.afc4 = nn.Linear(64, 32)
        self.mean = nn.Linear(32, 1)
        self.std = nn.Linear(32, 1)

    def forward(self, state):
        x = F.relu(self.afc1(state))
        x = F.relu(self.afc2(x))
        x = F.relu(self.afc3(x))
        x = F.relu(self.afc4(x))
        mean = self.mean(x)
        std = torch.exp(self.std(x))
        return torch.distributions.Normal(mean, std)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.dropout = nn.Dropout(0.1)

        self.vfc1 = nn.Linear(state_dim, 64)
        self.vfc2 = nn.Linear(64, 128)
        self.vfc3 = nn.Linear(128, 64)
        self.vfc4 = nn.Linear(64, 32)
        self.value = nn.Linear(32, 1)

    def forward(self, state):
        x = F.relu(self.vfc1(state))
        x = F.relu(self.vfc2(x))
        x = F.relu(self.vfc3(x))
        x = F.relu(self.vfc4(x))
        value = self.value(x)
        return value


class A2C(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A2C, self).__init__()
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.data = []

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(np.array(s_lst), dtype=torch.float).to(device), \
                                                               torch.tensor(np.array(a_lst)).to(device), \
                                                               torch.tensor(np.array(r_lst), dtype=torch.float).to(device), \
                                                               torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(device), \
                                                               torch.tensor(np.array(done_lst), dtype=torch.float).to(device)
        self.data = []

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def select_action(self, state):
        dist = self.actor(state)
        action = dist.sample()
        return action

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # Critic loss
        values = self.critic(s)
        next_values = self.critic(s_prime).detach()
        target_values = r + gamma * next_values * (1 - done) # target
        critic_loss = F.smooth_l1_loss(target_values, values) # advantages.pow(2).mean()

        advantages = target_values - values

        # Actor loss
        dist = self.actor(s)
        log_prob = dist.log_prob(a)
        actor_loss = -(log_prob * advantages.detach()).mean()

        critic_loss.backward()
        actor_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()


def main():
    env = gym.make('MountainCarContinuous-v0', render_mode="human")
    # print(env.observation_space.shape[0], env.action_space)
    model = A2C(env.observation_space.shape[0], 1)
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
                a = a.cpu().detach().numpy()
                s_prime, r, done, _, info = env.step(a)
                model.put_data((s, a, r, s_prime, done))
                # print(type(s), type(a), type(r), type(s_prime), type(done))
                # print(s, a, r, s_prime, done)
                s = s_prime

                print_score += r
                reward_sum += r
                step += 1

                if done:
                    print(step)
                    break

                model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, print_score / print_interval))
            print_score = 0.0

        scores.append(reward_sum)
        reward_sum = 0

    env.close()

    plt.figure(figsize=(10,6))
    plt.plot(scores)
    plt.plot(pd.Series(scores).rolling(100).mean())
    plt.title('A2C_continuous')
    plt.xlabel('# of episodes')
    plt.ylabel('score')
    plt.savefig('A2C_continuous.png')

    # torch.save(model.state_dict(), 'A2C_continuous.pth')

if __name__ == '__main__':
    main()
