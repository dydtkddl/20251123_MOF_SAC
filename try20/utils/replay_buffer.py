import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Final Upgraded ReplayBuffer for MACS–MOF RL
    ==========================================================
    Features:
    - PER (Prioritized Experience Replay)
    - Sliding window (keep recent MAX only)
    - Episode-balanced sampling
    - N-step return
    - reward-based priority
    - episode-level discard (COM/bond termination)
    ==========================================================
    Stores *per-atom transitions*:
        obs       (obs_dim,)
        act       (3,)
        reward    float
        next_obs  (obs_dim,)
        done      bool
        priority  float
    """

    def __init__(
        self,
        obs_dim,
        max_size=2_000_000,
        alpha=0.6,      # PER exponent
        beta=0.4,       # PER IS weight exponent
        n_step=3,
        gamma=0.995,
    ):
        self.obs_dim = obs_dim
        self.act_dim = 3

        # sliding window
        self.max_size = max_size

        # PER
        self.alpha = alpha
        self.beta = beta

        # N-step
        self.n_step = n_step
        self.gamma = gamma
        self.n_queue = deque(maxlen=n_step)

        # flat memory
        self.obs_buf = np.zeros((max_size, obs_dim), np.float32)
        self.nobs_buf = np.zeros((max_size, obs_dim), np.float32)
        self.act_buf = np.zeros((max_size, 3), np.float32)
        self.rew_buf = np.zeros(max_size, np.float32)
        self.done_buf = np.zeros(max_size, np.bool_)

        self.prior_buf = np.zeros(max_size, np.float32) + 1e-6

        self.ptr = 0
        self.size = 0

        # episode tracking
        self.ep_counter = 0
        self.current_ep_indices = []
        self.episode_track = []      # list of lists

    # ==========================================================
    # EPISODE CONTROL
    # ==========================================================
    def new_episode(self):
        """Call this at EP start."""
        self.ep_counter += 1
        self.current_ep_indices = []
        self.n_queue.clear()

    def end_episode(self, keep: bool = True):
        """Keep=False → discard by zeroing priority."""
        if not keep:
            for idx in self.current_ep_indices:
                self.prior_buf[idx] = 0.0
            self.current_ep_indices = []
            return

        # valid ep
        if len(self.current_ep_indices) > 0:
            self.episode_track.append(list(self.current_ep_indices))
        self.current_ep_indices = []

    # ==========================================================
    # N-step helper
    # ==========================================================
    def _convert_n_step(self, s, a, r, ns, d):
        """
        Convert to n-step transition.
        Returns None until queue is filled.
        """
        self.n_queue.append((s, a, r, ns, d))

        if len(self.n_queue) < self.n_step:
            return None

        R = 0
        discount = 1
        for (_, _, r_i, _, _) in self.n_queue:
            R += r_i * discount
            discount *= self.gamma

        s0, a0, _, _, _ = self.n_queue[0]
        _, _, _, ns_n, d_n = self.n_queue[-1]

        return s0, a0, R, ns_n, d_n

    # ==========================================================
    # STORE
    # ==========================================================
    def store(self, s, a, r, ns, d):
        """
        Add transition using n-step conversion + assign priority.
        """

        nstep = self._convert_n_step(s, a, r, ns, d)
        if nstep is None:
            return

        s0, a0, Rn, ns_n, d_n = nstep

        idx = self.ptr

        self.obs_buf[idx] = s0
        self.act_buf[idx] = a0
        self.rew_buf[idx] = Rn
        self.nobs_buf[idx] = ns_n
        self.done_buf[idx] = d_n

        # reward-based priority
        self.prior_buf[idx] = abs(float(Rn)) + 1e-6

        self.current_ep_indices.append(idx)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # ==========================================================
    # Sliding window clean
    # ==========================================================
    def _sliding_window_cleanup(self):
        if self.size < self.max_size:
            return

        # remove low priority transitions
        remove_n = 200_000
        lowest = np.argsort(self.prior_buf)[:remove_n]
        for idx in lowest:
            self.prior_buf[idx] = 0.0

    # ==========================================================
    # SAMPLE
    # ==========================================================
    def sample(self, batch_size):
        """
        Episode-balanced + PER sampling
        """

        if len(self.episode_track) == 0:
            # fallback random sample
            ids = np.random.randint(0, self.size, batch_size)
            w = np.ones(batch_size, np.float32)
            return self._package(ids, w)

        # choose episodes uniformly
        chosen_eps = random.sample(
            self.episode_track,
            k=min(len(self.episode_track), batch_size)
        )

        indices = []
        for ep_idxs in chosen_eps:
            prios = self.prior_buf[ep_idxs]
            if prios.sum() < 1e-9:
                continue

            probs = prios ** self.alpha
            probs /= probs.sum()

            pick = np.random.choice(ep_idxs, p=probs)
            indices.append(pick)

        # fill if needed
        if len(indices) < batch_size:
            remain = batch_size - len(indices)
            extra = np.random.randint(0, self.size, remain)
            indices.extend(extra.tolist())

        indices = np.array(indices[:batch_size])

        # PER IS weights
        prios = self.prior_buf[indices]
        probs = prios / prios.sum()
        weights = (len(self.prior_buf) * probs) ** (-self.beta)
        weights /= weights.max()

        return self._package(indices, weights.astype(np.float32))

    # ==========================================================
    def _package(self, indices, weights):
        """Return dict batch."""
        return dict(
            obs=self.obs_buf[indices],
            act=self.act_buf[indices],
            rew=self.rew_buf[indices],
            nobs=self.nobs_buf[indices],
            done=self.done_buf[indices],
            weights=weights,
            idx=indices,
        )

    # ==========================================================
    def update_priority(self, idx, td_err):
        """Agent can update priority after TD-error."""
        self.prior_buf[idx] = abs(float(td_err)) + 1e-6

    def __len__(self):
        return self.size
