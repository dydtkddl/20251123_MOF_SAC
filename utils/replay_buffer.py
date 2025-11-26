# utils/replay_buffer.py

import numpy as np


class ReplayBuffer:
    """
    Stable MACS-style ReplayBuffer
    ---------------------------------------------------------
    Stores *per-atom transitions*:
        obs_i       : (obs_dim,)
        act_i       : (3,)          # dx, dy, dz
        reward_i    : float
        next_obs_i  : (obs_dim,)
        done        : bool
    ---------------------------------------------------------
    """

    def __init__(
        self,
        obs_dim: int,
        max_size: int = 5_000_000,
    ):
        """
        act_dim is fixed to 3 for stable MACS-style SAC.
        """

        self.obs_dim = obs_dim
        self.act_dim = 3                # ALWAYS dx, dy, dz
        self.max_size = max_size

        self.ptr = 0
        self.size = 0

        # Buffers
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.nobs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)

        self.act_buf = np.zeros((max_size, 3), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.bool_)


    # ============================================================
    # STORE ONE ATOM TRANSITION
    # ============================================================
    def store(self, obs_i, act_i, rew_i, next_obs_i, done_i):
        """
        Parameters
        ----------
        obs_i : np.ndarray (obs_dim,)
        act_i : np.ndarray (3,)
        rew_i : float
        next_obs_i : np.ndarray (obs_dim,)
        done_i : bool
        """

        self.obs_buf[self.ptr] = obs_i
        self.act_buf[self.ptr] = act_i
        self.rew_buf[self.ptr] = rew_i
        self.nobs_buf[self.ptr] = next_obs_i
        self.done_buf[self.ptr] = done_i

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    # ============================================================
    # SAMPLE MINI-BATCH
    # ============================================================
    def sample(self, batch_size):
        """
        Randomly sample per-atom transitions
        """

        idxs = np.random.randint(0, self.size, size=batch_size)

        return dict(
            obs=self.obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            nobs=self.nobs_buf[idxs],
            done=self.done_buf[idxs],
        )


    def __len__(self):
        return self.size
