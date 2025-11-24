###############################################################
# utils/replay_buffer.py — MACS 3D-Action Fully Compatible Version
# - Supports 3D vector action (act_dim = 3)
# - Prioritized Experience Replay (PER)
# - n-step returns
# - Episode-aware sampling (to avoid bad episode imbalance)
# - Designed to handle 10M-scale buffers
###############################################################

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Hybrid MACS Replay Buffer with:
        ✓ act_dim = 3  (for 3D vector actions)
        ✓ n-step return
        ✓ Prioritized Experience Replay (PER)
        ✓ Episode-aware sampling
        ✓ 10M-scale memory safety

    This buffer is compatible with:
        SACAgent / main_train / actor / critic
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
        self.act_dim = 3        # ★ MACS: ALWAYS 3D action

        self.max_size = max_size

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.n_step = n_step
        self.n_queue = deque(maxlen=n_step)

        ###############################################################
        # Main Buffers (huge: up to 10M entries)
        ###############################################################
        self.obs_buf  = np.zeros((max_size, obs_dim), np.float32)
        self.act_buf  = np.zeros((max_size, 3), np.float32)    # ★ 3D ACTION
        self.rew_buf  = np.zeros((max_size,), np.float32)
        self.nobs_buf = np.zeros((max_size, obs_dim), np.float32)
        self.done_buf = np.zeros((max_size,), np.bool_)

        # PER priority buffer
        self.prior_buf = np.zeros((max_size,), np.float32) + 1e-6

        # Pointers
        self.ptr = 0
        self.size = 0

        # Episode storage for balanced sampling
        self.current_ep_indices = []
        self.episode_track = []


    ###################################################################
    # Episode lifecycle helpers
    ###################################################################
    def new_episode(self):
        self.current_ep_indices = []
        self.n_queue.clear()

    def end_episode(self, keep=True):
        """
        keep = False:
            remove priorities for this episode (training failed episode)
        """
        if not keep:
            for idx in self.current_ep_indices:
                self.prior_buf[idx] = 0.0
            self.current_ep_indices.clear()
            self.n_queue.clear()
            return

        # Otherwise store successful episode index list
        if len(self.current_ep_indices) > 0:
            self.episode_track.append(list(self.current_ep_indices))

        self.current_ep_indices.clear()
        self.n_queue.clear()


    ###################################################################
    # n-step merging
    ###################################################################
    def _convert_n_step(self, s, a, r, ns, d):
        """
        Input:
            s  : (obs_dim,)
            a  : (3,)  vector action
            r  : reward
            ns : next state
            d  : done

        If not enough history yet → return None
        If enough steps → return merged memory tuple
        """

        # Store temporarily
        self.n_queue.append((s, a, r, ns, d))

        # If insufficient n-step history → wait
        if len(self.n_queue) < self.n_step:
            return None

        # Merge reward: R = r0 + γ r1 + γ^2 r2 ...
        R = 0.0
        g = 1.0
        for (_, _, r_i, _, _) in self.n_queue:
            R += r_i * g
            g *= self.gamma

        # Use first state's s0, a0
        s0, a0, _, _, _ = self.n_queue[0]
        # Use last state's next state and done
        _, _, _, ns_n, d_n = self.n_queue[-1]

        return s0, a0, R, ns_n, d_n


    ###################################################################
    # Store one transition (with n-step support)
    ###################################################################
    def store(self, s, a, r, ns, d):
        """
        s : (obs_dim,)
        a : (3,) vector action ★
        r : float
        ns: (obs_dim,)
        d : bool
        """

        converted = self._convert_n_step(s, a, r, ns, d)
        if converted is None:
            return

        s0, a0, Rn, ns_n, d_n = converted

        idx = self.ptr

        # Store into circular buffer
        self.obs_buf[idx]  = s0
        self.act_buf[idx]  = a0
        self.rew_buf[idx]  = Rn
        self.nobs_buf[idx] = ns_n
        self.done_buf[idx] = d_n

        # Initial PER priority
        self.prior_buf[idx] = abs(float(Rn)) + 1e-6

        # Episode indexing
        self.current_ep_indices.append(idx)

        # Move ptr
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    ###################################################################
    # Episode-aware PER sample
    ###################################################################
    def sample(self, batch_size):

        if self.size == 0:
            raise RuntimeError("ReplayBuffer is empty!")

        # If no episodes yet → fallback to uniform sampling
        if len(self.episode_track) == 0:
            ids = np.random.randint(0, self.size, batch_size)
            w = np.ones(batch_size, np.float32)
            return self._package(ids, w)

        # 1) Choose episodes uniformly
        chosen_eps = random.sample(
            self.episode_track,
            k=min(batch_size, len(self.episode_track))
        )

        ids = []

        # 2) Sample 1 transition per selected episode (with PER)
        for ep_idxs in chosen_eps:
            p = self.prior_buf[ep_idxs]
            if np.sum(p) < 1e-12:   # avoid degenerate
                continue

            prob = (p ** self.alpha)
            prob /= prob.sum()

            idx = np.random.choice(ep_idxs, p=prob)
            ids.append(idx)

        # 3) If fewer than batch_size, fill remainder
        if len(ids) < batch_size:
            remain = batch_size - len(ids)
            extra = np.random.randint(0, self.size, remain)
            ids.extend(extra.tolist())

        ids = np.array(ids[:batch_size])

        # 4) PER importance sampling weight
        pr = self.prior_buf[ids] + 1e-12
        prob = pr ** self.alpha
        prob /= prob.sum()

        w = (self.size * prob) ** (-self.beta)
        w /= w.max()

        return self._package(ids, w.astype(np.float32))


    ###################################################################
    # Pack results for SACAgent.update()
    ###################################################################
    def _package(self, ids, weights):
        return dict(
            obs   = self.obs_buf[ids],      # (B, obs_dim)
            act   = self.act_buf[ids],      # (B, 3) ★
            rew   = self.rew_buf[ids],      # (B,)
            nobs  = self.nobs_buf[ids],     # (B, obs_dim)
            done  = self.done_buf[ids],     # (B,)
            weights = weights,              # (B,)
            idx    = ids
        )


    ###################################################################
    # PER priority update
    ###################################################################
    def update_priority(self, idx, td_err):
        """
        idx : list or int
        td_err : corresponding TD-error(s)
        """
        if np.isscalar(idx):
            self.prior_buf[idx] = abs(float(td_err)) + 1e-6
            return

        td_err = np.asarray(td_err)
        for i, e in zip(idx, td_err):
            self.prior_buf[i] = abs(float(e)) + 1e-6


    ###################################################################
    def __len__(self):
        return self.size
