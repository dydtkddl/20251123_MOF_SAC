# utils/replay_buffer.py

import numpy as np


class ReplayBuffer:

    def __init__(self,
                 obs_dim,
                 act_dim,
                 max_size=1_000_000):

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.nobs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((max_size, ), dtype=np.float32)
        self.done_buf = np.zeros((max_size, ), dtype=np.float32)


    # ----------------------------------------------------------

    def store(self, obs, act, rew, nobs, done):

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.nobs_buf[self.ptr] = nobs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    # ----------------------------------------------------------

    def sample(self, batch_size=256):

        idxs = np.random.choice(self.size, size=batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            nobs=self.nobs_buf[idxs],
            done=self.done_buf[idxs]
        )

