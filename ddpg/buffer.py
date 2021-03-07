# based on https://github.com/MrSyee/pg-is-all-you-need/blob/master/03.DDPG.ipynb
from typing import Dict

import numpy as np
from config import BUFFER_SIZE, BATCH_SIZE, NUM_OBS, NUM_ACT


class Buffer:
    def __init__(self):
        self.obs_buf = np.zeros([BUFFER_SIZE, NUM_OBS], dtype=np.float32)
        self.next_obs_buf = np.zeros([BUFFER_SIZE, NUM_OBS], dtype=np.float32)
        self.acts_buf = np.zeros([BUFFER_SIZE], dtype=np.float32)
        self.rews_buf = np.zeros([BUFFER_SIZE], dtype=np.float32)
        self.done_buf = np.zeros([BUFFER_SIZE], dtype=np.float32)
        self.max_size, self.batch_size = BUFFER_SIZE, BATCH_SIZE
        self.ptr, self.size, = 0, 0

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size