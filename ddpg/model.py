import torch
import numpy as np
from torch import FloatTensor
import torch.nn.functional as F
from torch.distributions import Normal

from config import NUM_OBS, NUM_ACT, ACTOR_HIDDEN_DIM, CRITIC_HIDDEN_DIM


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = torch.nn.Linear(NUM_OBS, ACTOR_HIDDEN_DIM[0])
        self.hiddens = torch.nn.ModuleList([torch.nn.Linear(ACTOR_HIDDEN_DIM[i], ACTOR_HIDDEN_DIM[i+1])
                                            for i in range(len(ACTOR_HIDDEN_DIM)-1)])
        self.out = torch.nn.Linear(ACTOR_HIDDEN_DIM[-1], NUM_ACT)

        init_layer(self.hidden1)
        for hidden in self.hiddens:
            init_layer(hidden)
        init_layer(self.out, [-3e-3, 3e-3])
        self.out.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, x: FloatTensor):
        """Forward pass through the model, input is x"""
        x = self.hidden1(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = F.relu(x)
        action = torch.tanh(self.mu_layer(x))

        return action


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = torch.nn.Linear(NUM_OBS+NUM_ACT, CRITIC_HIDDEN_DIM[0])
        self.hiddens = torch.nn.ModuleList([torch.nn.Linear(CRITIC_HIDDEN_DIM[i], CRITIC_HIDDEN_DIM[i + 1])
                                            for i in range(len(CRITIC_HIDDEN_DIM) - 1)])
        self.value = torch.nn.Linear(CRITIC_HIDDEN_DIM[-1], 1)

        init_layer(self.hidden1)
        for hidden in self.hiddens:
            init_layer(hidden)
        init_layer(self.value)

    def forward(self, x: FloatTensor, a: FloatTensor):
        """Forward pass through the model, input is x"""
        x = torch.cat([x, a], dim=-1)
        x = self.hidden1(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = F.relu(x)
        val = self.value(x)

        return val


def init_layer(layer, range_=None):
    if range_ is None:
        num_inputs = layer.weight.data.size()[0]
        temp = 1.0/np.sqrt(num_inputs)
        min_ = -num_inputs
        max_ = num_inputs
    else:
        min_, max_ = range_
    layer.weight.data.uniform_(min_, max_)
