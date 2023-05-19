import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
env = gym.make('CartPole-v1')
score_history = [] # array to store reward

# hyperparameters definition
EPISODES = 200
EPS_START = 0.8
EPS_END = 0.05
EPS_DECAY = 200
GAMMA = 0.8
LR = 0.001
BATCH_SIZE = 32
TARGET_UPDATE = 10

Transition = namedtuple('Transition',
                                    ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    # def push(self, *args):
    #     self.memory.append(Transition(*args))

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, 
                            torch.tensor([action]),
                            torch.FloatTensor([reward]),
                            torch.FloatTensor([next_state])))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def choose_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if random.random() > eps_threshold: # if random value > epsilon value
        with torch.no_grad():
            return policy_net(state).data.max(-1)[1].item() # neural network result
    else:
        return np.random.randint(action_space) # random integer result


def learn():
    if len(memory) < BATCH_SIZE:
        return

    batch = memory.sample(BATCH_SIZE) # random batch sampling
    states, actions, rewards, next_states = zip(*batch) # separate batch by element list

    # Tensor list
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    next_states = torch.stack(next_states)

    current_q = policy_net(states).gather(1, actions)# 1 dim's actions value, size[32, 1]
    ''' DQN on policy_net'''
    # max_next_q = policy_net(next_states).max(1)[0].unsqueeze(1) # get max_next_q at poicy_net, size[64]
    # unsqueeze(): create 1 dim
    # squeeze(): remove 1 dim ex) [3, 1, 20, 128] -> [3, 20, 128]

    ''' DQN on target_net'''
    # max_next_q = target_net(next_states).max(1)[0] # get max_next_q at targety_net, size[64]

    ''' DDQN'''
    a = policy_net(states).data.max(-1)[1].unsqueeze(1) # size[32, 1]
    max_next_q = target_net(next_states).squeeze().gather(1, a)

    expected_q = rewards + (GAMMA * max_next_q) # rewards + future value

    loss = F.mse_loss(current_q, expected_q)
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train_dqn():
    for e in range(1, EPISODES + 1):
        state = env.reset() # state = [-0.01508306 -0.04199532  0.00379785  0.01508063], <class 'numpy.ndarray'>
        steps = 0
        while True:
            state = torch.FloatTensor(state) # tensorize state
            action = choose_action(state) # integer

            next_state, reward, done, _ = env.step(action)
        
            if done:
                reward = -1

            memory.memorize(state, action, reward, next_state) # memory experience
            learn()

            state = next_state
            steps += 1

            if done or steps == 200:
                print("Episode:{0} step: {1} ".format(e, steps))
                score_history.append(steps)
                break

        if e % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            # print("Policy_net to Target_net")


if __name__ == '__main__':
    # get env's state & action spaces
    state_space = 4 
    action_space = 2  

    policy_net = DQN(state_space, action_space).to(device) # policy net = main net
    target_net = DQN(state_space, action_space).to(device) # target net
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters(), LR)
    memory = ReplayMemory(10000)
    steps_done = 0

    train_dqn()

    # np.save('DQN.npy', score_history)
    plt.figure()
    plt.plot(score_history)
    plt.xlabel('Episode')
    plt.ylabel('steps')
    plt.title('DQN')
    plt.show()
