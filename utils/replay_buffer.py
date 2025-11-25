import logging
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Multi-agent per-atom transition을 flatten해서 저장하는 일반 SAC 버퍼.
    obs:  (N, obs_dim)
    act:  (N, act_dim)
    rew:  scalar (global), 모든 atom에 동일하게 할당
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        size: int,
        device: torch.device,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.size_max = size
        self.device = device

        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

        logger.info(
            f"[ReplayBuffer] Initialized: max_size={size}, obs_dim={obs_dim}, act_dim={act_dim}"
        )

    def __len__(self):
        return self.size

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.acts_buf[idx] = act
        self.rew_buf[idx] = rew
        self.next_obs_buf[idx] = next_obs
        self.done_buf[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.size_max
        self.size = min(self.size + 1, self.size_max)

    def store_batch(
        self,
        obs_batch: np.ndarray,   # (N, obs_dim)
        act_batch: np.ndarray,   # (N, act_dim)
        rew: float,
        next_obs_batch: np.ndarray,  # (N, obs_dim)
        done: bool,
    ):
        N = obs_batch.shape[0]
        for i in range(N):
            self.store(
                obs_batch[i],
                act_batch[i],
                rew,
                next_obs_batch[i],
                done,
            )

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        if self.size == 0:
            raise RuntimeError("ReplayBuffer is empty")

        batch_size = min(batch_size, self.size)
        idx = np.random.randint(0, self.size, size=batch_size)

        obs = torch.as_tensor(self.obs_buf[idx], dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(self.acts_buf[idx], dtype=torch.float32, device=self.device)
        rews = torch.as_tensor(self.rew_buf[idx], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(
            self.next_obs_buf[idx], dtype=torch.float32, device=self.device
        )
        done = torch.as_tensor(self.done_buf[idx], dtype=torch.float32, device=self.device)

        return obs, acts, rews, next_obs, done
