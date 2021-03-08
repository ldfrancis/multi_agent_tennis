import torch
import numpy as np
from torch import FloatTensor
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal

from config import NUM_OBS, NUM_ACT, ACTOR_HIDDEN_DIM, CRITIC_HIDDEN_DIM

class Actor(nn.Module):
    def __init__(self, seed=0):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(NUM_OBS, ACTOR_HIDDEN_DIM[0])
        self.fc2 = nn.Linear(ACTOR_HIDDEN_DIM[0], ACTOR_HIDDEN_DIM[1])
        self.fc3 = nn.Linear(ACTOR_HIDDEN_DIM[1], NUM_ACT)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, seed=0):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(NUM_OBS, CRITIC_HIDDEN_DIM[0])
        self.fc2 = nn.Linear(CRITIC_HIDDEN_DIM[0] + NUM_ACT, CRITIC_HIDDEN_DIM[1])
        self.fc3 = nn.Linear(CRITIC_HIDDEN_DIM[1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim
