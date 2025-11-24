import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    ReplayBuffer for MOF structure optimization (scale-factor RL):

    Features:
    -------------------------------------------------------
    ✓ PER (Prioritized Experience Replay)
    ✓ TD-error-based priority update
    ✓ Importance sampling weights
    ✓ Episode-balanced sampling (stabilizes per-MOF training)
    ✓ n-step return (default = 1)
    ✓ Sliding window FIFO memory
    ✓ act_dim = 1 (scalar action only)
    -------------------------------------------------------
    """

    def __init__(
        self,
        obs_dim,
        max_size=2_000_000,
        alpha=0.6,
        beta=0.4,
        n_step=1,
        gamma=0.995
    ):
        self.obs_dim = obs_dim
        self.act_dim = 1               # scale-factor only

        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta

        self.n_step = n_step
        self.gamma = gamma
        self.n_queue = deque(maxlen=n_step)   # for n-step accumulation

        # -------------------------------
        # Buffers
        # -------------------------------
        self.obs_buf = np.zeros((max_size, obs_dim), np.float32)
        self.nobs_buf = np.zeros((max_size, obs_dim), np.float32)
        self.act_buf = np.zeros((max_size, 1), np.float32)
        self.rew_buf = np.zeros(max_size, np.float32)
        self.done_buf = np.zeros(max_size, np.bool_)

        # PER priorities
        self.prior_buf = np.zeros(max_size, np.float32) + 1e-6

        self.ptr = 0
        self.size = 0

        # Episode management
        self.ep_counter = 0
        self.current_ep_indices = []
        self.episode_track = []


    # ======================================================================
    # Episode management
    # ======================================================================
    def new_episode(self):
        """Called at the beginning of each environment episode"""
        self.ep_counter += 1
        self.current_ep_indices = []
        self.n_queue.clear()


    def end_episode(self, keep=True):
        """
        Store current episode indices into episode_track
        or discard them (if BAD termination, com/bond crash)
        """
        if not keep:
            # bad episode: set priority to zero
            for idx in self.current_ep_indices:
                self.prior_buf[idx] = 0.0
            self.current_ep_indices = []
            return

        # good episode: store for sampling
        if len(self.current_ep_indices) > 0:
            self.episode_track.append(list(self.current_ep_indices))

        self.current_ep_indices = []


    # ======================================================================
    # N-step return assembly
    # ======================================================================
    def _convert_n_step(self, s, a, r, ns, d):
        """
        n-step transition converter.
        Returns None until the internal queue is full (len == n_step).
        """
        self.n_queue.append((s, a, r, ns, d))

        if len(self.n_queue) < self.n_step:
            return None

        # cumulative discounted reward
        R = 0
        discount = 1.0
        for (_, _, r_i, _, _) in self.n_queue:
            R += r_i * discount
            discount *= self.gamma

        s0, a0, _, _, _ = self.n_queue[0]
        _, _, _, ns_n, d_n = self.n_queue[-1]

        return s0, a0, R, ns_n, d_n


    # ======================================================================
    # Store transition (with n-step)
    # ======================================================================
    def store(self, s, a, r, ns, d):
        """
        s: obs_dim vector
        a: scalar action [1]
        r: reward per-atom
        ns: next obs
        d: done flag
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

        # initial priority = |reward|
        self.prior_buf[idx] = abs(float(Rn)) + 1e-6

        # record for episode-balanced sampling
        self.current_ep_indices.append(idx)

        # FIFO sliding window
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    # ======================================================================
    # PER Sampling (episode-balanced)
    # ======================================================================
    def sample(self, batch_size):
        """
        Episode-balanced prioritized sampling:
        1) randomly choose episodes
        2) within each episode select 1 or more transitions using PER
        """
        if self.size == 0:
            raise ValueError("ReplayBuffer is empty!")

        # no episode stored yet → fallback random sampling
        if len(self.episode_track) == 0:
            ids = np.random.randint(0, self.size, batch_size)
            w = np.ones(batch_size, np.float32)
            return self._package(ids, w)

        # sample episodes
        chosen_eps = random.sample(
            self.episode_track,
            k=min(len(self.episode_track), batch_size)
        )

        indices = []
        for ep_idxs in chosen_eps:

            prios = self.prior_buf[ep_idxs]
            if prios.sum() < 1e-9:
                continue

            # PER probability inside episode
            probs = prios ** self.alpha
            probs /= probs.sum()

            pick = np.random.choice(ep_idxs, p=probs)
            indices.append(pick)

        # if still under batch_size, fill with global random samples
        if len(indices) < batch_size:
            remain = batch_size - len(indices)
            extra = np.random.randint(0, self.size, remain)
            indices.extend(extra.tolist())

        indices = np.array(indices[:batch_size])

        # calculate weights = IS-correction
        prios = self.prior_buf[indices]
        probs = prios / prios.sum()
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()  # normalize

        return self._package(indices, weights.astype(np.float32))


    # ======================================================================
    def _package(self, ids, weights):
        return dict(
            obs=self.obs_buf[ids],
            act=self.act_buf[ids],
            rew=self.rew_buf[ids],
            nobs=self.nobs_buf[ids],
            done=self.done_buf[ids],
            weights=weights,
            idx=ids,
        )


    # ======================================================================
    # TD-error priority update
    # ======================================================================
    def update_priority(self, idx, td_err):
        """
        td_err: numpy float or array
        """
        self.prior_buf[idx] = abs(float(td_err)) + 1e-6


    # ======================================================================
    def __len__(self):
        return self.size
