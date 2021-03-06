import torch
from torch import FloatTensor
import torch.nn.functional as F
from torch.distributions import Normal

from config import NUM_OBS, NUM_ACT, POLICY_HIDDEN_DIM, MAX_LOG_STD, MIN_LOG_STD, CRITIC_HIDDEN_DIM


class Policy(torch.nn.Module):
    """Model used for the ppo policy. produces a distribution over actions given a state
    """

    def __init__(self):
        super().__init__()
        self.hidden1 = torch.nn.Linear(NUM_OBS, POLICY_HIDDEN_DIM[0])
        self.hidden = torch.nn.Sequential(*[torch.nn.Linear(POLICY_HIDDEN_DIM[i], POLICY_HIDDEN_DIM[i+1])
                                            for i in range(len(POLICY_HIDDEN_DIM)-1)])
        self.mu_layer = torch.nn.Linear(POLICY_HIDDEN_DIM[-1], NUM_ACT)
        self.log_std_layer = torch.nn.Linear(POLICY_HIDDEN_DIM[-1], NUM_ACT)

    def forward(self, x: FloatTensor):
        """Forward pass through the model, input is x"""
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden(x)
        x = F.relu(x)
        mu = torch.tanh(self.mu_layer(x) )
        log_std = MIN_LOG_STD + (torch.tanh(self.log_std_layer(x))+1)*(MAX_LOG_STD-MIN_LOG_STD)/2
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist


class Critic(torch.nn.Module):
    """Model used for the critic of ppo. computes the value of a state/observation
    """

    def __init__(self):
        super().__init__()
        self.hidden1 = torch.nn.Linear(NUM_OBS, CRITIC_HIDDEN_DIM[0])
        self.hidden = torch.nn.Sequential(*[torch.nn.Linear(CRITIC_HIDDEN_DIM[i], CRITIC_HIDDEN_DIM[i + 1])
                                            for i in range(len(CRITIC_HIDDEN_DIM) - 1)])
        self.value = torch.nn.Linear(CRITIC_HIDDEN_DIM[-1], 1)

    def forward(self, x: FloatTensor):
        """Forward pass through the model, input is x"""
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden(x)
        x = F.relu(x)
        val = self.value(x)

        return val



