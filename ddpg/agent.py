import copy
from random import random

import numpy as np
import torch
import torch.functional as F

import os

from .model import Actor, Critic
from .buffer import Buffer
from config import ACTOR_LR, CRITIC_LR, TAU, NUM_ACT, GAMMA
from pathlib import Path


class DDPG:
    """PPO Agent"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.actor_target = Actor()
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.buffer = Buffer()
        self.steps = 0

        # noise
        self.noise = OUNoise(
            NUM_ACT,
            theta=OU_NOISE_THETA,
            sigma=OU_NOISE_SIGMA,
        )

        # restore if checkpoint exists
        if Path("./checkpoint/checkpoint.pt").exists():
            self.restore("./checkpoint/checkpoint.pt")

    def take_action(self, state:np.ndarray, train=True):
        state = torch.FloatTensor(state[None, :]).to(DDPG.device)
        self.actor.eval()
        action = self.actor(state).detach().cpu().numpy()
        if train:
            action += np.clip(self.noise.sample(),-1,1)
        self.steps +=1

        return action

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample()
        states = torch.FloatTensor(states).to(DDPG.device)
        actions = torch.FloatTensor(actions).to(DDPG.device)
        rewards = torch.FloatTensor(rewards).to(DDPG.device)
        next_states = torch.FloatTensor(next_states).to(DDPG.device)
        dones = torch.FloatTensor(dones).to(DDPG.device)

        masks = 1-dones
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(next_states, next_actions)
        curr_return = rewards + GAMMA * next_values * masks

        # train critic
        values = self.critic(states, actions)
        critic_loss = F.mse_loss(values, curr_return)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # train actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # target update
        self._target_soft_update()

    def _target_soft_update(self):
        for t_param, l_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            t_param.data.copy_(TAU * l_param.data + (1.0 - TAU) * t_param.data)

        for t_param, l_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            t_param.data.copy_(TAU * l_param.data + (1.0 - TAU) * t_param.data)

    def save(self, path:str="./checkpoint"):
        os.makedirs(path, exist_ok=True)
        checkpoint_path_file = f"{path}/checkpoint.pt"
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            }, checkpoint_path_file )

    def restore(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


class OUNoise:
    """Ornstein-Uhlenbeck process.
    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.state


