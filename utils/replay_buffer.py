# utils/replay_buffer.py

import numpy as np


class ReplayBuffer:
    """
    MACS-style per-atom replay buffer.
    Stores: <obs, act, reward, next_obs, done>
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        max_size=10_000_000,  # MACS spec scale
    ):

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # obs / next obs
        self.obs_buf  = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.nobs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)

        # act
        self.act_buf  = np.zeros((max_size, act_dim), dtype=np.float32)

        # reward
        self.rew_buf  = np.zeros((max_size,), dtype=np.float32)

        # done flag
        self.done_buf = np.zeros((max_size,), dtype=np.bool_)



    ###################################################################
    # STORE ONE ATOM TRANSITION
    ###################################################################
    def store(self, obs, act, rew, next_obs, done):

        self.obs_buf[self.ptr]  = obs
        self.act_buf[self.ptr]  = act
        self.rew_buf[self.ptr]  = rew
        self.nobs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)



    ###################################################################
    # SAMPLE MINI-BATCH
    ###################################################################
    def sample(self, batch_size):

        idxs = np.random.randint(0, self.size, size=batch_size)

        return dict(
            obs  = self.obs_buf[idxs],
            act  = self.act_buf[idxs],
            rew  = self.rew_buf[idxs],
            nobs = self.nobs_buf[idxs],
            done = self.done_buf[idxs],
        )



    ###################################################################
    def __len__(self):
        return self.size
