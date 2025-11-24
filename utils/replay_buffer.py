###############################################################
# utils/replay_buffer.py — Hybrid-MACS Final Complete Version
# Features:
#   ✓ obs_dim dynamic
#   ✓ 10M-scale memory supported
#   ✓ PER (α=0.6, β=0.4)
#   ✓ n-step return (γ=0.995)
#   ✓ Episode-balanced sampling
#   ✓ Good/Bad episode priority control
#   ✓ TD-error priority update
#   ✓ Compatible with SACAgent & main_train.py
###############################################################

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Hybrid-MACS ReplayBuffer

    Supports:
    -------------------------------------------------------
    ✓ Dynamic obs_dim (from env.reset())
    ✓ PER (Prioritized Experience Replay)
    ✓ TD-error priority update
    ✓ Importance sampling weights
    ✓ Episode-balanced sampling
    ✓ n-step return support
    ✓ 10M-scale FIFO storage
    ✓ Action_dim = 1 (scale action)
    -------------------------------------------------------
    """

    def __init__(
        self,
        obs_dim,
        max_size=10_000_000,   # ★ MACS-level large-scale buffer
        alpha=0.6,             # PER exponent
        beta=0.4,              # IS weight exponent
        n_step=1,
        gamma=0.995
    ):
        self.obs_dim = obs_dim
        self.act_dim = 1

        # capacity
        self.max_size = max_size

        # PER parameters
        self.alpha = alpha
        self.beta = beta

        # n-step
        self.n_step = n_step
        self.gamma = gamma
        self.n_queue = deque(maxlen=n_step)

        # allocate buffers
        self.obs_buf  = np.zeros((max_size, obs_dim), np.float32)
        self.nobs_buf = np.zeros((max_size, obs_dim), np.float32)
        self.act_buf  = np.zeros((max_size, self.act_dim), np.float32)
        self.rew_buf  = np.zeros(max_size, np.float32)
        self.done_buf = np.zeros(max_size, np.bool_)

        # PER priority buffer
        self.prior_buf = np.zeros(max_size, np.float32) + 1e-6

        # pointers
        self.ptr = 0
        self.size = 0

        # episode tracking
        self.current_ep_indices = []
        self.episode_track = []


    ###########################################################
    # Episode Management
    ###########################################################
    def new_episode(self):
        """Called whenever env.reset() happens."""
        self.current_ep_indices = []
        self.n_queue.clear()

    def end_episode(self, keep=True):
        """
        keep=True  → good episode → keep priorities  
        keep=False → bad episode(bond break, COM crash) → priority=0
        """
        if not keep:
            for idx in self.current_ep_indices:
                self.prior_buf[idx] = 0.0
            self.current_ep_indices = []
            self.n_queue.clear()
            return

        # good episode → record
        if len(self.current_ep_indices) > 0:
            self.episode_track.append(list(self.current_ep_indices))

        self.current_ep_indices = []
        self.n_queue.clear()


    ###########################################################
    # n-step Return Converter
    ###########################################################
    def _convert_n_step(self, s, a, r, ns, d):
        """
        Accumulates transitions until we have n items.
        """
        self.n_queue.append((s, a, r, ns, d))

        # insufficient length → return nothing
        if len(self.n_queue) < self.n_step:
            return None

        # compute discounted n-step return
        R = 0.0
        discount = 1.0
        for (_, _, r_i, _, _) in self.n_queue:
            R += r_i * discount
            discount *= self.gamma

        # extract (state_0, action_0)
        s0, a0, _, _, _ = self.n_queue[0]

        # final next_state, done
        _, _, _, ns_n, done_n = self.n_queue[-1]

        return s0, a0, R, ns_n, done_n


    ###########################################################
    # Store Transition
    ###########################################################
    def store(self, s, a, r, ns, d):
        """
        Insert a new transition after converting to n-step form.
        """

        out = self._convert_n_step(s, a, r, ns, d)
        if out is None:
            return

        s0, a0, Rn, ns_n, d_n = out

        idx = self.ptr

        # write data
        self.obs_buf[idx]  = s0
        self.act_buf[idx]  = a0
        self.rew_buf[idx]  = Rn
        self.nobs_buf[idx] = ns_n
        self.done_buf[idx] = d_n

        # initialize PER priority
        self.prior_buf[idx] = abs(float(Rn)) + 1e-6

        # record this index inside current episode
        self.current_ep_indices.append(idx)

        # FIFO pointer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    ###########################################################
    # Episode-Balanced PER Sampling
    ###########################################################
    def sample(self, batch_size):

        if self.size == 0:
            raise RuntimeError("ReplayBuffer is empty!")

        # no episode yet → uniform fallback sampling
        if len(self.episode_track) == 0:
            ids = np.random.randint(0, self.size, size=batch_size)
            w = np.ones(batch_size, np.float32)
            return self._package(ids, w)

        # sample episodes first
        chosen_eps = random.sample(
            self.episode_track,
            k=min(len(self.episode_track), batch_size)
        )

        ids = []

        # pick one transition per episode with PER
        for ep_idxs in chosen_eps:

            # priorities inside an episode
            p = self.prior_buf[ep_idxs]
            if p.sum() < 1e-12:
                continue  # skip fully-zero-episode

            prob = p ** self.alpha
            prob /= prob.sum()

            chosen = np.random.choice(ep_idxs, p=prob)
            ids.append(chosen)

        # fill remaining batch slots with uniform random
        if len(ids) < batch_size:
            remain = batch_size - len(ids)
            extra = np.random.randint(0, self.size, remain)
            ids.extend(extra.tolist())

        ids = np.array(ids[:batch_size])

        # PER importance weights
        pr = self.prior_buf[ids] + 1e-12
        P = pr ** self.alpha
        P /= P.sum()

        w = (self.size * P) ** (-self.beta)
        w /= w.max()

        return self._package(ids, w.astype(np.float32))


    ###########################################################
    # Build Return Package for SACAgent
    ###########################################################
    def _package(self, ids, weights):
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
    # TD-error priority update
    ###########################################################
    def update_priority(self, idx, td_err):

        if np.isscalar(idx):
            self.prior_buf[idx] = abs(float(td_err)) + 1e-6
            return

        for i, te in zip(idx, np.atleast_1d(td_err)):
            self.prior_buf[i] = abs(float(te)) + 1e-6


    ###########################################################
    def __len__(self):
        return self.size
