import numpy as np
import torch

import os

from .model import Policy, Critic
from config import POLICY_LR, CRITIC_LR, CLIP_EPSILON, ENTROPY_WEIGHT
from pathlib import Path


class PPO:
    """PPO Agent"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        self.policy = Policy()
        self.critic = Critic()
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=POLICY_LR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        # restore if checkpoint exists
        if Path("./checkpoint/checkpoint.pt").exists():
            self.restore("./checkpoint/checkpoint.pt")

    def take_action(self, state:np.ndarray):
        state = torch.FloatTensor(state[None, :]).to(PPO.device)
        self.policy.eval()
        action, dist = self.policy(state)

        return action, dist

    def compute_value(self, state:np.ndarray):
        state = torch.FloatTensor(state[None, :]).to(PPO.device)
        value = self.critic(state)
        return value

    def update(self, state:np.ndarray, action:np.ndarray, value:np.ndarray, log_prob:np.ndarray, return_:np.ndarray,
               advantage:np.ndarray):
        # convert to tensor
        state = torch.FloatTensor(state).to(PPO.device)
        action = torch.FloatTensor(action).to(PPO.device)
        value = torch.FloatTensor(value).to(PPO.device)
        log_prob = torch.FloatTensor(log_prob).to(PPO.device)
        return_ = torch.FloatTensor(return_).to(PPO.device)
        advantage = torch.FloatTensor(advantage).to(PPO.device)

        # policy ratio
        _, dist = self.policy(state)
        _log_prob = dist.log_prob(action)
        ratio = (_log_prob-log_prob).exp()

        # policy loss
        objective = ratio * advantage
        clipped_objective = torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * advantage
        ppo_clip_objective = torch.min(objective,clipped_objective).mean()
        entropy = dist.entropy().mean()
        policy_loss = -(ppo_clip_objective + entropy*ENTROPY_WEIGHT)

        # critic loss
        _value = self.critic(state)
        critic_loss = (return_ - _value).pow(2).mean()

        # optimize
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item(), policy_loss.item()

    def save(self, path:str="./checkpoint"):
        os.makedirs(path, exist_ok=True)
        checkpoint_path_file = f"{path}/checkpoint.pt"
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            }, checkpoint_path_file )

    def restore(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])





