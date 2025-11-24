###############################################################
# utils/replay_buffer.py — MACS 3D-Action Fully Compatible Version
###############################################################

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Hybrid-MACS ReplayBuffer with 3D action support.

    Features:
    ----------------------------------------------------
    ✓ obs_dim dynamic
    ✓ act_dim = 3 (MACS vector action)
    ✓ PER (Prioritized Experience Replay)
    ✓ TD-error priority update
    ✓ Episode-aware sampling
    ✓ n-step return
    ✓ 10M-scale memory
    ----------------------------------------------------
    """

    def __init__(
        self,
        obs_dim,
        max_size=10_000_000,
        alpha=0.6,
        beta=0.4,
        n_step=1,
        gamma=0.995
    ):
        self.obs_dim = obs_dim
        self.act_dim = 3                 # ★ MACS 3D action

        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta

        self.n_step = n_step
        self.gamma = gamma
        self.n_queue = deque(maxlen=n_step)

        ########################################################
        # Buffers (10M scale)
        ########################################################
        self.obs_buf  = np.zeros((max_size, obs_dim), np.float32)
        self.nobs_buf = np.zeros((max_size, obs_dim), np.float32)

        # ★ 3D action buf
        self.act_buf  = np.zeros((max_size, self.act_dim), np.float32)

        self.rew_buf  = np.zeros(max_size, np.float32)
        self.done_buf = np.zeros(max_size, np.bool_)

        self.prior_buf = np.zeros(max_size, np.float32) + 1e-6

        self.ptr = 0
        self.size = 0

        self.current_ep_indices = []
        self.episode_track = []


    ###########################################################
    # Episode Lifecycle
    ###########################################################
    def new_episode(self):
        self.current_ep_indices = []
        self.n_queue.clear()

    def end_episode(self, keep=True):
        if not keep:
            for idx in self.current_ep_indices:
                self.prior_buf[idx] = 0.0
            self.current_ep_indices = []
            self.n_queue.clear()
            return

        if len(self.current_ep_indices) > 0:
            self.episode_track.append(list(self.current_ep_indices))

        self.current_ep_indices = []
        self.n_queue.clear()


    ###########################################################
    # n-step Return
    ###########################################################
    def _convert_n_step(self, s, a, r, ns, d):
        """
        a must be shape (3,) float32
        """

        # store sequence
        self.n_queue.append((s, a, r, ns, d))

        if len(self.n_queue) < self.n_step:
            return None

        # discounted reward
        R = 0.0
        discount = 1.0
        for (_, _, r_i, _, _) in self.n_queue:
            R += r_i * discount
            discount *= self.gamma

        # s0, a0 from first element
        s0, a0, _, _, _ = self.n_queue[0]

        # final next_state
        _, _, _, ns_n, d_n = self.n_queue[-1]

        return s0, a0, R, ns_n, d_n


    ###########################################################
    # Store Transition
    ###########################################################
    def store(self, s, a, r, ns, d):
        """
        s : (obs_dim,)
        a : (3,)          ★ 3D action
        ns: (obs_dim,)
        r : scalar
        d : bool
        """
        out = self._convert_n_step(s, a, r, ns, d)
        if out is None:
            return

        s0, a0, Rn, ns_n, done_n = out

        idx = self.ptr

        self.obs_buf[idx]  = s0
        self.act_buf[idx]  = a0      # ★ store vector action
        self.rew_buf[idx]  = Rn
        self.nobs_buf[idx] = ns_n
        self.done_buf[idx] = done_n

        # initial PER priority
        self.prior_buf[idx] = abs(float(Rn)) + 1e-6

        # save index
        self.current_ep_indices.append(idx)

        # FIFO move
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    ###########################################################
    # Episode-Balanced PER Sampling
    ###########################################################
    def sample(self, batch_size):

        if self.size == 0:
            raise RuntimeError("ReplayBuffer is empty")

        # fallback: uniform sampling
        if len(self.episode_track) == 0:
            ids = np.random.randint(0, self.size, size=batch_size)
            w = np.ones(batch_size, np.float32)
            return self._package(ids, w)

        chosen_eps = random.sample(
            self.episode_track,
            k=min(len(self.episode_track), batch_size)
        )

        ids = []

        # PER inside episode
        for ep_idxs in chosen_eps:
            p = self.prior_buf[ep_idxs]
            if p.sum() < 1e-12:
                continue

            prob = p ** self.alpha
            prob /= prob.sum()

            idx = np.random.choice(ep_idxs, p=prob)
            ids.append(idx)

        # fill rest
        if len(ids) < batch_size:
            remain = batch_size - len(ids)
            extra = np.random.randint(0, self.size, remain)
            ids.extend(extra.tolist())

        ids = np.array(ids[:batch_size])

        # PER weights
        pr = self.prior_buf[ids] + 1e-12
        prob = pr ** self.alpha
        prob /= prob.sum()

        w = (self.size * prob) ** (-self.beta)
        w /= w.max()

        return self._package(ids, w.astype(np.float32))


    ###########################################################
    # Packaging for SACAgent
    ###########################################################
    def _package(self, ids, weights):
        """
        Returning shapes:
          obs   → (B, obs_dim)
          act   → (B, 3)
          rew   → (B,)
          nobs  → (B, obs_dim)
          done  → (B,)
          weights → (B,)
        """
        return dict(
            obs   = self.obs_buf[ids],
            act   = self.act_buf[ids],
            rew   = self.rew_buf[ids],
            nobs  = self.nobs_buf[ids],
            done  = self.done_buf[ids],
            weights = weights,
            idx = ids,
        )


    ###########################################################
    # PER priority update
    ###########################################################
    def update_priority(self, idx, td_err):
        if np.isscalar(idx):
            self.prior_buf[idx] = abs(float(td_err)) + 1e-6
            return

        for i, e in zip(idx, np.atleast_1d(td_err)):
            self.prior_buf[i] = abs(float(e)) + 1e-6


    ###########################################################
    def __len__(self):
        return self.size
