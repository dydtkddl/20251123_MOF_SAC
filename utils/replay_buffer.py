import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    ReplayBuffer for MOF scale-factor SAC:

    Features:
    -------------------------------------------------------
    ✓ PER (Prioritized Experience Replay)
    ✓ TD-error priority update
    ✓ Importance sampling weights
    ✓ Episode-balanced sampling (reduces collapse)
    ✓ n-step return support
    ✓ Sliding FIFO memory
    ✓ Supports scalar action only (act_dim = 1)
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
        self.act_dim = 1

        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta

        self.n_step = n_step
        self.gamma = gamma

        # queue for n-step staging
        self.n_queue = deque(maxlen=n_step)

        # Buffers
        self.obs_buf = np.zeros((max_size, obs_dim), np.float32)
        self.nobs_buf = np.zeros((max_size, obs_dim), np.float32)
        self.act_buf = np.zeros((max_size, 1), np.float32)
        self.rew_buf = np.zeros(max_size, np.float32)
        self.done_buf = np.zeros(max_size, np.bool_)

        self.prior_buf = np.zeros(max_size, np.float32) + 1e-6

        # pointers
        self.ptr = 0
        self.size = 0

        # episode management
        self.current_ep_indices = []
        self.episode_track = []


    # ============================================================
    # Episode management
    # ============================================================
    def new_episode(self):
        self.current_ep_indices = []
        self.n_queue.clear()

    def end_episode(self, keep=True):
        """keep=False → BAD episode (bond/com crash) → remove priorities"""
        if not keep:
            for idx in self.current_ep_indices:
                self.prior_buf[idx] = 0.0
            self.current_ep_indices = []
            return

        # good episode → store track
        if len(self.current_ep_indices) > 0:
            self.episode_track.append(list(self.current_ep_indices))

        self.current_ep_indices = []


    # ============================================================
    # n-step converter
    # ============================================================
    def _convert_n_step(self, s, a, r, ns, d):
        """
        Returns None until the internal queue fills (len = n_step)
        """
        self.n_queue.append((s, a, r, ns, d))

        if len(self.n_queue) < self.n_step:
            return None

        R = 0.0
        discount = 1.0
        for (_, _, r_i, _, _) in self.n_queue:
            R += r_i * discount
            discount *= self.gamma

        s0, a0, _, _, _ = self.n_queue[0]
        _, _, _, nobs_n, done_n = self.n_queue[-1]

        return s0, a0, R, nobs_n, done_n


    # ============================================================
    # Store transition
    # ============================================================
    def store(self, s, a, r, ns, d):
        """
        Stores one transition after n-step conversion.
        """
        nstep_out = self._convert_n_step(s, a, r, ns, d)

        if nstep_out is None:
            return

        s0, a0, Rn, ns_n, d_n = nstep_out

        idx = self.ptr

        self.obs_buf[idx] = s0
        self.act_buf[idx] = a0
        self.rew_buf[idx] = Rn
        self.nobs_buf[idx] = ns_n
        self.done_buf[idx] = d_n

        # priority initialization
        self.prior_buf[idx] = abs(float(Rn)) + 1e-6

        # record this index to episode history
        self.current_ep_indices.append(idx)

        # FIFO pointer update
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    # ============================================================
    # PER Sampling (episode-balanced)
    # ============================================================
    def sample(self, batch_size):

        if self.size == 0:
            raise ValueError("ReplayBuffer empty!")

        # if no episode added yet → fallback
        if len(self.episode_track) == 0:
            ids = np.random.randint(0, self.size, batch_size)
            weights = np.ones(batch_size, np.float32)
            return self._package(ids, weights)

        # sample episodes
        chosen_eps = random.sample(
            self.episode_track,
            k=min(len(self.episode_track), batch_size)
        )

        ids = []
        for ep_idxs in chosen_eps:

            pr = self.prior_buf[ep_idxs]
            pr_sum = pr.sum()

            if pr_sum < 1e-12:
                continue

            probs = pr ** self.alpha
            probs /= probs.sum()

            pick = np.random.choice(ep_idxs, p=probs)
            ids.append(pick)

        # fill leftovers
        if len(ids) < batch_size:
            remain = batch_size - len(ids)
            extra = np.random.randint(0, self.size, remain)
            ids.extend(extra.tolist())

        ids = np.array(ids[:batch_size])

        # importance sampling weights
        pr = self.prior_buf[ids]
        pr = pr + 1e-12
        P = pr ** self.alpha
        P /= P.sum()

        weights = (self.size * P) ** (-self.beta)
        weights /= weights.max()

        return self._package(ids, weights.astype(np.float32))


    # ============================================================
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


    # ============================================================
    # TD-error priority update
    # ============================================================
    def update_priority(self, idx, td_err):
        """
        idx: int or array
        td_err: scalar or same-shape array
        """
        if np.isscalar(idx):
            self.prior_buf[idx] = abs(float(td_err)) + 1e-6
        else:
            for i, e in zip(idx, np.atleast_1d(td_err)):
                self.prior_buf[i] = abs(float(e)) + 1e-6


    # ============================================================
    def __len__(self):
        return self.size
