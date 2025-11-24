###############################################################
# utils/replay_buffer.py — Structure-Level SAC Replay Buffer
# -------------------------------------------------------------
# - obs_global_dim = N * obs_dim_atom (flatten)
# - act_global_dim = N * 3            (flatten)
#
# Features:
#   ✓ PER (Prioritized Experience Replay)
#   ✓ n-step returns (structure-level)
#   ✓ episode-aware sampling
#   ✓ handles >10M transitions
#   ✓ TD-error 기반 priority update
#
# Compatible with:
#   main_train.py
#   sac/actor.py
#   sac/critic.py
#   sac/agent.py
###############################################################

import numpy as np
import random
from collections import deque



class ReplayBuffer:
    """
    Structure-level Replay Buffer for MACS SAC.

    Stores:
        s_global   : (obs_global_dim,)
        a_global   : (act_global_dim,)
        r          : float
        ns_global  : (obs_global_dim,)
        done       : bool

    Key Features:
        ✓ Prioritized Experience Replay
        ✓ n-step return merging
        ✓ episode-aware sampling (balanced training)
        ✓ 10M-scale circular buffer
    """

    def __init__(
        self,
        obs_global_dim: int,
        act_global_dim: int,        # = 3 * N_atoms
        max_size=5_000_000,
        alpha=0.6,                  # PER exponent
        beta=0.4,                   # PER bias correction
        n_step=1,
        gamma=0.995
    ):
        ##########################
        # DIMENSIONS
        ##########################
        self.obs_dim = obs_global_dim
        self.act_dim = act_global_dim

        ##########################
        # PER settings
        ##########################
        self.alpha = alpha
        self.beta = beta

        ##########################
        # n-step settings
        ##########################
        self.n_step = n_step
        self.gamma = gamma
        self.n_queue = deque(maxlen=n_step)

        ##########################
        # Memory allocation
        ##########################
        self.max_size = max_size

        self.obs_buf  = np.zeros((max_size, self.obs_dim), np.float32)
        self.act_buf  = np.zeros((max_size, self.act_dim), np.float32)
        self.rew_buf  = np.zeros((max_size,), np.float32)
        self.nobs_buf = np.zeros((max_size, self.obs_dim), np.float32)
        self.done_buf = np.zeros((max_size,), np.bool_)

        # PER priority buffer
        self.prior_buf = np.zeros((max_size,), np.float32) + 1e-6

        ##########################
        # Pointer tracking
        ##########################
        self.ptr = 0
        self.size = 0

        ##########################
        # Episode management
        ##########################
        self.current_ep_indices = []
        self.episodes = []     # list of lists (episode index lists)



    ###############################################################
    # Episode lifecycle
    ###############################################################
    def new_episode(self):
        """Call at the beginning of each episode."""
        self.current_ep_indices = []
        self.n_queue.clear()


    def end_episode(self, keep=True):
        """
        If keep=False:
            Reset priorities for this episode (failed episode)
        """
        if not keep:
            for idx in self.current_ep_indices:
                self.prior_buf[idx] = 0.0
            self.current_ep_indices.clear()
            self.n_queue.clear()
            return

        # store successful episode
        if len(self.current_ep_indices) > 0:
            self.episodes.append(list(self.current_ep_indices))

        self.current_ep_indices.clear()
        self.n_queue.clear()



    ###############################################################
    # n-step return merging
    ###############################################################
    def _merge_n_step(self, s, a, r, ns, done):
        """
        s, ns  : flattened obs_global_dim vector
        a      : flattened act_global_dim vector
        r      : float
        done   : bool

        Returns:
            (s0, a0, Rn, ns_n, done_n)
        OR:
            None (if insufficient queue yet)
        """

        self.n_queue.append((s, a, r, ns, done))

        # insufficient history
        if len(self.n_queue) < self.n_step:
            return None

        # compute n-step return
        R = 0.0
        g = 1.0
        for (_, _, r_i, _, _) in self.n_queue:
            R += r_i * g
            g *= self.gamma

        s0, a0, _, _, _     = self.n_queue[0]
        _, _, _, ns_n, d_n  = self.n_queue[-1]

        return s0, a0, R, ns_n, d_n



    ###############################################################
    # Store transition (structure-level)
    ###############################################################
    def store(self, s, a, r, ns, done):
        """
        s   : (obs_global_dim,)
        a   : (act_global_dim,)
        r   : float
        ns  : (obs_global_dim,)
        done: bool
        """

        merged = self._merge_n_step(s, a, r, ns, done)
        if merged is None:
            return

        s0, a0, Rn, ns_n, d_n = merged

        idx = self.ptr

        # write into circular buffer
        self.obs_buf[idx]  = s0
        self.act_buf[idx]  = a0
        self.rew_buf[idx]  = Rn
        self.nobs_buf[idx] = ns_n
        self.done_buf[idx] = d_n

        # initial priority = |reward|
        self.prior_buf[idx] = abs(Rn) + 1e-6

        # episode index tracking
        self.current_ep_indices.append(idx)

        # ptr update
        self.ptr = (idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)



    ###############################################################
    # Episode-aware PER sampling
    ###############################################################
    def sample(self, batch_size):

        if self.size == 0:
            raise RuntimeError("ReplayBuffer is empty.")

        ###############################
        # If no episodes → uniform sampling
        ###############################
        if len(self.episodes) == 0:
            idxs = np.random.randint(0, self.size, batch_size)
            w = np.ones(batch_size, np.float32)
            return self._package(idxs, w)


        ###############################
        # 1) choose episodes uniformly
        ###############################
        chosen_eps = random.sample(
            self.episodes,
            k=min(batch_size, len(self.episodes))
        )

        idxs = []

        ###############################
        # 2) PER sampling inside each episode
        ###############################
        for ep_idxs in chosen_eps:
            pri = self.prior_buf[ep_idxs]

            if np.sum(pri) < 1e-12:
                continue

            prob = pri ** self.alpha
            prob /= prob.sum()

            chosen_idx = np.random.choice(ep_idxs, p=prob)
            idxs.append(chosen_idx)

        ###############################
        # 3) fill remainder (uniform)
        ###############################
        if len(idxs) < batch_size:
            need = batch_size - len(idxs)
            extra = np.random.randint(0, self.size, need)
            idxs.extend(extra.tolist())

        idxs = np.array(idxs[:batch_size])

        ###############################
        # 4) PER importance sampling
        ###############################
        pr = self.prior_buf[idxs] + 1e-12
        prob = pr ** self.alpha
        prob /= prob.sum()

        w = (self.size * prob) ** (-self.beta)
        w /= w.max()

        return self._package(idxs, w.astype(np.float32))



    ###############################################################
    # Packaging for SACAgent.update()
    ###############################################################
    def _package(self, idxs, w):
        return dict(
            obs   = self.obs_buf[idxs],       # (B, obs_dim)
            act   = self.act_buf[idxs],       # (B, act_dim)
            rew   = self.rew_buf[idxs],       # (B,)
            nobs  = self.nobs_buf[idxs],      # (B, obs_dim)
            done  = self.done_buf[idxs],      # (B,)
            weights = w,                      # (B,)
            idx    = idxs                     # (B,)
        )



    ###############################################################
    # PER priority update (TD error 기반)
    ###############################################################
    def update_priority(self, idxs, td_errors):
        """
        idxs      : array-like index list
        td_errors : array-like TD error list
        """
        td_errors = np.abs(np.asarray(td_errors)) + 1e-6

        for i, e in zip(idxs, td_errors):
            self.prior_buf[i] = e



    ###############################################################
    def __len__(self):
        return self.size
