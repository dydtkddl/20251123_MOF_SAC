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
    # BULK LOAD (from arrays / dict)
    # ============================================================
    def load_from_dict(
        self,
        data_dict,
        max_samples: int = None,
        shuffle: bool = True,
        reset_buffer: bool = True,
    ):
        """
        Bulk-load transitions from a dict into the buffer.

        Parameters
        ----------
        data_dict : dict
            Must contain:
                'obs'  : (N, obs_dim)
                'act'  : (N, act_dim) or (N, 3)  # 3D는 여기서 에러; expert 유틸에서 처리
                'rew'  : (N,)
                'nobs' : (N, obs_dim)
                'done' : (N,)
        max_samples : int, optional
            If provided, at most this many samples will be loaded.
        shuffle : bool
            Whether to shuffle indices before loading.
        reset_buffer : bool
            Whether to reset ptr and size before bulk loading.
        """
        obs = data_dict["obs"]
        act = data_dict["act"]
        rew = data_dict["rew"]
        nobs = data_dict["nobs"]
        done = data_dict["done"]

        N = obs.shape[0]
        if obs.shape != nobs.shape:
            raise ValueError(
                f"[ReplayBuffer.load_from_dict] obs shape {obs.shape} "
                f"!= nobs shape {nobs.shape}"
            )

        if obs.shape[1] != self.obs_dim:
            raise ValueError(
                f"[ReplayBuffer.load_from_dict] obs_dim mismatch: "
                f"expert={obs.shape[1]}, replay={self.obs_dim}"
            )

        if act.shape[1] != self.act_dim:
            raise ValueError(
                f"[ReplayBuffer.load_from_dict] act_dim mismatch: "
                f"expert={act.shape[1]}, replay={self.act_dim} "
                f"(3D expert는 utils.expert_replay.seed_replay_from_expert에서 "
                f"padding 처리 후 사용해야 함)"
            )

        if max_samples is None:
            num = min(N, self.max_size)
        else:
            num = min(N, max_samples, self.max_size)

        if num <= 0:
            logger.warning(
                "[ReplayBuffer.load_from_dict] No samples to load (num=%d).", num
            )
            return

        idxs = np.arange(N, dtype=np.int64)
        if shuffle:
            np.random.shuffle(idxs)
        idxs = idxs[:num]

        if reset_buffer:
            logger.info(
                "[ReplayBuffer.load_from_dict] Resetting buffer before load "
                "(prev size=%d, ptr=%d)",
                self.size, self.ptr,
            )
            self.size = 0
            self.ptr = 0

        logger.info(
            "[ReplayBuffer.load_from_dict] Bulk loading %d samples "
            "(N=%d, max_size=%d, shuffle=%s)",
            num, N, self.max_size, shuffle,
        )

        # 여기서는 store를 호출하지 않고 바로 배열 복사로 빠르게 채운다
        self.obs_buf[:num] = obs[idxs]
        self.act_buf[:num] = act[idxs]
        self.rew_buf[:num] = rew[idxs]
        self.nobs_buf[:num] = nobs[idxs]
        self.done_buf[:num] = done[idxs]

        self.ptr = num % self.max_size
        self.size = num

        logger.info(
            "[ReplayBuffer.load_from_dict] Done. size=%d, ptr=%d",
            self.size, self.ptr,
        )

    # ============================================================
    def __len__(self):
        return self.size
