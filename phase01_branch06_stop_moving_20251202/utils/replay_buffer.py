# utils/replay_buffer.py

import numpy as np
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    # 기본 핸들러 (상위에서 로깅 설정한 경우 중복 방지)
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class ReplayBuffer:
    """
    Stable MACS-style ReplayBuffer (Per-Atom, 4D action)
    ---------------------------------------------------------
    Stores *per-atom transitions*:
        obs_i       : (obs_dim,)
        act_i       : (4,)          # [gate, dx, dy, dz]
        reward_i    : float
        next_obs_i  : (obs_dim,)
        done        : bool
    ---------------------------------------------------------
    Notes
    -----
    - This buffer is designed for MACS-style per-atom SAC.
    - We fix act_dim=4 (gate + 3D displacement) for stability.
    """

    def __init__(
        self,
        obs_dim: int,
        max_size: int = 5_000_000,
        log_interval: int = 500_000,
    ):
        """
        Parameters
        ----------
        obs_dim : int
            Dimension of per-atom observation.
        max_size : int
            Maximum number of per-atom transitions to store.
        log_interval : int
            How often (in number of stored transitions) to emit a size log.
        """

        self.obs_dim = obs_dim
        self.act_dim = 4                # [gate, dx, dy, dz]
        self.max_size = max_size
        self.log_interval = max(1, int(log_interval))

        self.ptr = 0
        self.size = 0

        # Buffers
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.nobs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)

        self.act_buf = np.zeros((max_size, self.act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.bool_)

        logger.info(
            f"[ReplayBuffer] Initialized "
            f"(obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
            f"max_size={self.max_size:,})"
        )

    # ============================================================
    # STORE ONE ATOM TRANSITION
    # ============================================================
    def store(self, obs_i, act_i, rew_i, next_obs_i, done_i):
        """
        Parameters
        ----------
        obs_i : np.ndarray, shape (obs_dim,)
            Current per-atom observation.
        act_i : np.ndarray, shape (4,)
            Per-atom action = [gate, dx, dy, dz].
        rew_i : float
            Scalar reward for this atom.
        next_obs_i : np.ndarray, shape (obs_dim,)
            Next per-atom observation.
        done_i : bool
            Episode termination flag.
        """

        # 기본 shape 체크 (필요 시 디버깅용)
        if act_i.shape[-1] != self.act_dim:
            raise ValueError(
                f"[ReplayBuffer] act_i has wrong dim: "
                f"expected {self.act_dim}, got {act_i.shape[-1]}"
            )
        if obs_i.shape[-1] != self.obs_dim or next_obs_i.shape[-1] != self.obs_dim:
            raise ValueError(
                f"[ReplayBuffer] obs dim mismatch: "
                f"expected {self.obs_dim}, got "
                f"{obs_i.shape[-1]}, {next_obs_i.shape[-1]}"
            )

        self.obs_buf[self.ptr] = obs_i
        self.act_buf[self.ptr] = act_i
        self.rew_buf[self.ptr] = rew_i
        self.nobs_buf[self.ptr] = next_obs_i
        self.done_buf[self.ptr] = done_i

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        # 간헐적인 로깅 (과도한 I/O 방지)
        if self.size % self.log_interval == 0:
            logger.info(
                f"[ReplayBuffer] size={self.size:,} "
                f"(max_size={self.max_size:,})"
            )
        # 한 번이라도 래핑되면 알려줌
        if self.ptr == 0 and self.size == self.max_size:
            logger.warning(
                "[ReplayBuffer] Buffer is full; overwriting oldest samples from now on."
            )

    # ============================================================
    # SAMPLE MINI-BATCH
    # ============================================================
    def sample(self, batch_size):
        """
        Randomly sample per-atom transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        batch : dict of np.ndarray
            {
                'obs'  : (batch, obs_dim),
                'act'  : (batch, 4),
                'rew'  : (batch,),
                'nobs' : (batch, obs_dim),
                'done' : (batch,),
            }
        """
        if self.size == 0:
            raise RuntimeError(
                "[ReplayBuffer] Cannot sample: buffer is empty (size=0)."
            )

        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(
            obs=self.obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            nobs=self.nobs_buf[idxs],
            done=self.done_buf[idxs],
        )

        return batch

    # ============================================================
    def __len__(self):
        return self.size
