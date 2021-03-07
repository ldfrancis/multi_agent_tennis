import torch
import numpy as np
from torch import FloatTensor
import torch.nn.functional as F
from torch.distributions import Normal

from config import NUM_OBS, NUM_ACT, POLICY_HIDDEN_DIM, MAX_LOG_STD, MIN_LOG_STD, CRITIC_HIDDEN_DIM


class Policy(torch.nn.Module):
    """Model used for the ippo policy. produces a distribution over actions given a state
    """

    def __init__(self):
        super().__init__()
        self.hidden1 = torch.nn.Linear(NUM_OBS, POLICY_HIDDEN_DIM[0])
        self.hiddens = torch.nn.ModuleList([torch.nn.Linear(POLICY_HIDDEN_DIM[i], POLICY_HIDDEN_DIM[i+1])
                                            for i in range(len(POLICY_HIDDEN_DIM)-1)])
        self.mu_layer = torch.nn.Linear(POLICY_HIDDEN_DIM[-1], NUM_ACT)
        self.log_std_layer = torch.nn.Linear(POLICY_HIDDEN_DIM[-1], NUM_ACT)

        init_layer(self.hidden1)
        for hidden in self.hiddens:
            init_layer(hidden)
        init_layer(self.mu_layer)
        init_layer(self.log_std_layer)

    def forward(self, x: FloatTensor):
        """Forward pass through the model, input is x"""
        x = self.hidden1(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = F.relu(x)
        mu = torch.tanh(self.mu_layer(x) )
        log_std = MIN_LOG_STD + (torch.tanh(self.log_std_layer(x))+1)*(MAX_LOG_STD-MIN_LOG_STD)/2
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist


class Critic(torch.nn.Module):
    """Model used for the critic of ippo. computes the value of a state/observation
    """

    def __init__(self):
        super().__init__()
        self.hidden1 = torch.nn.Linear(NUM_OBS, CRITIC_HIDDEN_DIM[0])
        self.hiddens = torch.nn.ModuleList([torch.nn.Linear(CRITIC_HIDDEN_DIM[i], CRITIC_HIDDEN_DIM[i + 1])
                                            for i in range(len(CRITIC_HIDDEN_DIM) - 1)])
        self.value = torch.nn.Linear(CRITIC_HIDDEN_DIM[-1], 1)

        init_layer(self.hidden1)
        for hidden in self.hiddens:
            init_layer(hidden)
        init_layer(self.value)

    def forward(self, x: FloatTensor):
        """Forward pass through the model, input is x"""
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
