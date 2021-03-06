from typing import List

import numpy as np
from torch import Tensor

from config import GAMMA, LAMBDA


class TrajectoryProcessor:
    """Processes a trajectory to compute the returns and advantages"""

    def __init__(self, num_envs: int = 20):
        self.num_envs = num_envs
        self.states: np.ndarray = None
        self.actions: np.ndarray = None
        self.values: np.ndarray = None
        self.log_probs: np.ndarray = None
        self.returns: np.ndarray = None
        self.advantages: np.ndarray = None

    def process(self, states: List[List[np.ndarray]], actions: List[List[float]], rewards: List[List[float]],
                values: List[List[float]], dones: List[List[bool]], log_probs: List[List[float]],
                next_value: List[List[float]]):
        """Perfroms the processing of collected trajectories
        """
        returns = [[] for _ in range(self.num_envs)]
        advantages = [[] for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            returns[i] = compute_gae(next_value[i], rewards[i], dones[i], values[i], GAMMA, LAMBDA)
            advantages[i] = [r - v for r, v in zip(returns[i], values[i])]

        self.states = np.concatenate([s[None, :] for state in states for s in state], axis=0)
        self.actions = np.concatenate([a for action in actions for a in action], axis=0)
        self.values = np.concatenate([v for value in values for v in value], axis=0)
        self.log_probs = np.concatenate([l for l_prob in log_probs for l in l_prob], axis=0)
        self.returns = np.concatenate([r for return_ in returns for r in return_], axis=0)
        self.advantages = np.concatenate([ad for advantage in advantages for ad in advantage], axis=0)

    def iterate(self, epochs: int = 10, batch_size: int = 32):
        size = len(self.states)
        iters = size // batch_size
        for _ in range(iters * epochs):
            rand_ids = np.random.choice(size, batch_size)
            yield self.states[rand_ids], self.actions[rand_ids], self.values[rand_ids], self.log_probs[rand_ids], \
                  self.returns[rand_ids], self.advantages[rand_ids]


class TrajectoryPrep:
    """Prepares a trajectory for processing. collects experiences till the end of a trajectory"""

    def __init__(self, num_envs: int = 20):
        self.num_envs = num_envs
        self.states = [[] for _ in range(num_envs)]
        self.actions = [[] for _ in range(num_envs)]
        self.rewards = [[] for _ in range(num_envs)]
        self.values = [[] for _ in range(num_envs)]
        self.dones = [[] for _ in range(num_envs)]
        self.log_probs = [[] for _ in range(num_envs)]

    def reset(self):
        self.states = [[] for _ in range(self.num_envs)]
        self.actions = [[] for _ in range(self.num_envs)]
        self.rewards = [[] for _ in range(self.num_envs)]
        self.values = [[] for _ in range(self.num_envs)]
        self.dones = [[] for _ in range(self.num_envs)]
        self.log_probs = [[] for _ in range(self.num_envs)]

    def add(self, states: List[np.ndarray], actions: List[np.ndarray], rewards: List[float], values: List[float],
            dones: List[bool], log_probs: List[float]):
        for i in range(self.num_envs):
            self.states[i] += [states[i]]
            self.actions[i] += [actions[i]]
            self.rewards[i] += [rewards[i]]
            self.values[i] += [values[i]]
            self.dones[i] += [dones[i]]
            self.log_probs[i] += [log_probs[i]]

    def get(self):
        states = self.states
        actions = self.actions
        rewards = self.rewards
        values = self.values
        dones = self.dones
        log_probs = self.log_probs
        self.reset()
        return states, actions, rewards, values, dones, log_probs


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
