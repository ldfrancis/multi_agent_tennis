# based on https://github.com/MrSyee/pg-is-all-you-need/blob/master/03.DDPG.ipynb
import random
from collections import namedtuple, deque
from typing import Dict

import numpy as np
import torch

from config import BUFFER_SIZE, BATCH_SIZE, NUM_OBS, NUM_ACT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Buffer:

    def __init__(self, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=0):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self.seed = random.seed(seed)

    def store(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample_batch(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
