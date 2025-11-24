import numpy as np


class ReplayBuffer:
    """
    Priority-weighted ReplayBuffer for MACS-style per-atom SAC
    --------------------------------------------------------------------
    Stores *per-atom transitions*:
        obs_i       : (obs_dim,)
        act_i       : (3,)
        reward_i    : float
        next_obs_i  : (obs_dim,)
        done        : bool
    --------------------------------------------------------------------
    Features:
        - reward-weighted storage (priority-like)
        - reward clipping ready
        - smaller buffer (200k)
        - warm-up support
    --------------------------------------------------------------------
    """

    def __init__(
        self,
        obs_dim: int,
        max_size: int = 200_000,
        reward_weight: float = 0.2,     # ★ priority strength reduced from 2.0 → 0.2
        warmup: int = 10_000,           # minimum samples before training
    ):
        self.obs_dim = obs_dim
        self.act_dim = 3
        self.max_size = max_size

        self.reward_weight = reward_weight
        self.warmup = warmup

        self.ptr = 0
        self.size = 0

        # Buffers
        self.obs_buf  = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.nobs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf  = np.zeros((max_size, 3), dtype=np.float32)
        self.rew_buf  = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.bool_)


    # ============================================================
    # Reward-weighted storage (priority-like sampling)
    # ============================================================
    def store(self, obs_i, act_i, rew_i, next_obs_i, done_i):
        """
        Reward-weighted storage rule:
            p_store = sigmoid(|reward| * reward_weight)

        ★ reward_weight이 0.2로 약해져서
          거의 대부분의 transition이 저장됨 (좋은 안정성)
        """

        # reward magnitude determines priority
        mag = abs(float(rew_i))

        # Sigmoid priority probability
        p = 1.0 / (1.0 + np.exp(- self.reward_weight * mag))

        # Decide store or skip
        if np.random.rand() > p:
            return False

        # Store transition
        self.obs_buf[self.ptr]  = obs_i
        self.act_buf[self.ptr]  = act_i
        self.rew_buf[self.ptr]  = rew_i
        self.nobs_buf[self.ptr] = next_obs_i
        self.done_buf[self.ptr] = done_i

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return True


    # ============================================================
    # Random mini-batch sampling
    # ============================================================
    def sample(self, batch_size):
        assert (
            self.size >= self.warmup
        ), f"[ReplayBuffer] Need warmup={self.warmup}, current={self.size}"

        idxs = np.random.randint(0, self.size, size=batch_size)

        return dict(
            obs  = self.obs_buf[idxs],
            act  = self.act_buf[idxs],
            rew  = self.rew_buf[idxs],
            nobs = self.nobs_buf[idxs],
            done = self.done_buf[idxs],
        )


    # ============================================================
    def ready(self):
        return self.size >= self.warmup

    def __len__(self):
        return self.size
