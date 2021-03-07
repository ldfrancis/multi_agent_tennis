from typing import List

import numpy as np
import torch
from torch import Tensor


from config import GAMMA, LAMBDA


class TrajectoryProcessor:
    """Processes a trajectory to compute the returns and advantages"""

    def __init__(self, ppo_agent, num_envs: int = 2):
        self.num_envs = num_envs
        self.states: np.ndarray = None
        self.actions: np.ndarray = None
        self.values: np.ndarray = None
        self.log_probs: np.ndarray = None
        self.returns: np.ndarray = None
        self.advantages: np.ndarray = None
        self.agent = ppo_agent

    def process(self, states: List[List[np.ndarray]], actions: List[List[float]], rewards: List[List[float]],
                dones: List[List[bool]], next_state: List[List[np.ndarray]], epochs: int=10):
        """Perfroms the processing of collected trajectories
        """

        for _ in range(epochs):
            returns = [[] for _ in range(self.num_envs)]
            advantages = [[] for _ in range(self.num_envs)]
            values = [[] for _ in range(self.num_envs)]
            log_probs = [[] for _ in range(self.num_envs)]

            for i in range(self.num_envs):
                dists = [self.agent.take_action(ob)[-1] for ob in states[i]]
                values[i] = [self.agent.compute_value(ob).detach().cpu().numpy() for ob in states[i]]
                log_probs[i] = [dist.log_prob(torch.FloatTensor(action)).detach().cpu().numpy() for action, dist in zip(actions[i], dists)]

            next_value = [self.agent.compute_value(ob).detach().cpu().numpy() for ob in next_state]

            for i in range(self.num_envs):
                returns[i] = compute_gae(next_value[i], rewards[i], dones[i], values[i], GAMMA, LAMBDA)
                advantages[i] = [r - v for r, v in zip(returns[i], values[i])]

            self.states = np.concatenate([s[None, :] for state in states for s in state], axis=0)
            self.actions = np.concatenate([a for action in actions for a in action], axis=0)
            self.values = np.concatenate([v for value in values for v in value], axis=0)
            self.log_probs = np.concatenate([l for l_prob in log_probs for l in l_prob], axis=0)
            self.returns = np.concatenate([r for return_ in returns for r in return_], axis=0)
            self.advantages = np.concatenate([ad for advantage in advantages for ad in advantage], axis=0)

            size = len(self.states)

            rand_ids = np.random.choice(size, size, replace=False)
            yield self.states[rand_ids], self.actions[rand_ids], self.values[rand_ids], self.log_probs[rand_ids], \
                  self.returns[rand_ids], self.advantages[rand_ids]


class TrajectoryPrep:
    """Prepares a trajectory for processing. collects experiences till the end of a trajectory"""

    def __init__(self, num_envs: int = 2):
        self.num_envs = num_envs
        self.states = [[] for _ in range(num_envs)]
        self.actions = [[] for _ in range(num_envs)]
        self.rewards = [[] for _ in range(num_envs)]
        self.dones = [[] for _ in range(num_envs)]

    def reset(self):
        self.states = [[] for _ in range(self.num_envs)]
        self.actions = [[] for _ in range(self.num_envs)]
        self.rewards = [[] for _ in range(self.num_envs)]
        self.dones = [[] for _ in range(self.num_envs)]

    def add(self, states: List[np.ndarray], actions: List[np.ndarray], rewards: List[float],
            dones: List[bool]):
        for i in range(self.num_envs):
            self.states[i] += [states[i]]
            self.actions[i] += [actions[i]]
            self.rewards[i] += [rewards[i]]
            self.dones[i] += [dones[i]]

    def get(self):
        states = self.states
        actions = self.actions
        rewards = self.rewards
        dones = self.dones
        self.reset()
        return states, actions, rewards, dones


def compute_gae(next_value: float, rewards: List[float], dones: List[bool], values: List[float], gamma: float,
                lmbd: float):
    values += [next_value]
    gae = 0
    returns = []
    for i in reversed(range(len(rewards))):
        adv = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
        gae = adv + gamma * lmbd * (1 - dones[i]) * gae
        returns = [gae] + returns

    return returns
