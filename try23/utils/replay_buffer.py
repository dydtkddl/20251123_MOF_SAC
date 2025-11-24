###############################################################
# utils/replay_buffer.py — Structure-Level SAC Replay Buffer
# -------------------------------------------------------------
# 옵션 1(강력 추천):
#   - obs_global (flatten) 저장 X
#   - obs_atom (N, FEAT) 저장 O
#   - update()에서 flatten (Agent에서 수행)
#
# Features:
#   ✓ PER (Prioritized Experience Replay)
#   ✓ n-step returns
#   ✓ episode-aware sampling
#   ✓ >10M transitions 지원
#   ✓ TD-error 기반 priority update
#
# 저장되는 항목:
#   s_atom   : (N, FEAT)
#   a_global : (3N,)
#   r        : float
#   ns_atom  : (N, FEAT)
#   done     : bool
###############################################################

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Structure-level Replay Buffer (per-atom obs 저장 방식)

    저장 항목:
        s_atom    : (N, FEAT)
        a_global  : (3N,)
        r         : float
        ns_atom   : (N, FEAT)
        done      : bool

    PER 및 n-step return을 지원한다.
    """

    ###############################################################
    # Constructor
    ###############################################################
    def __init__(
        self,
        n_atoms: int,
        feat_dim: int,
        act_global_dim: int,
        max_size=4_000_000,
        alpha=0.6,
        beta=0.4,
        n_step=1,
        gamma=0.995
    ):
        """
        n_atoms      : MOF atom count
        feat_dim     : per-atom feature dimension
        act_global_dim = 3 * n_atoms
        """

        # ---------------------------
        # Store dims
        # ---------------------------
        self.N = n_atoms
        self.F = feat_dim
        self.obs_atom_dim = (self.N, self.F)
        self.act_dim = act_global_dim

        # PER settings
        self.alpha = alpha
        self.beta = beta

        # n-step settings
        self.n_step = n_step
        self.gamma = gamma
        self.n_queue = deque(maxlen=n_step)

        # Memory allocation
        self.max_size = max_size

        # (max_size, N, F)
        self.obs_buf = np.zeros((max_size, self.N, self.F), dtype=np.float32)
        self.nobs_buf = np.zeros((max_size, self.N, self.F), dtype=np.float32)

        # (max_size, 3N)
        self.act_buf = np.zeros((max_size, self.act_dim), dtype=np.float32)

        # reward, done
        self.rew_buf = np.zeros((max_size,), dtype=np.float32)
        self.done_buf = np.zeros((max_size,), dtype=np.bool_)

        # priority
        self.prior_buf = np.zeros((max_size,), dtype=np.float32) + 1e-6

        # ptr tracking
        self.ptr = 0
        self.size = 0

        # Episode tracking
        self.current_ep_indices = []
        self.episodes = []


    ###############################################################
    # Episode lifecycle
    ###############################################################
    def new_episode(self):
        self.current_ep_indices = []
        self.n_queue.clear()

    def end_episode(self, keep=True):

        # 실패 episode (COM blowup, bond break)
        if not keep:
            for idx in self.current_ep_indices:
                self.prior_buf[idx] = 0.0
            self.current_ep_indices.clear()
            self.n_queue.clear()
            return

        # 성공 episode 저장
        if len(self.current_ep_indices) > 0:
            self.episodes.append(list(self.current_ep_indices))

        self.current_ep_indices.clear()
        self.n_queue.clear()



    ###############################################################
    # n-step return merging
    ###############################################################
    def _merge_n_step(self, s_atom, a_global, r, ns_atom, done):
        """
        n-step 반환:
            (s0_atom, a0_global, Rn, ns_n_atom, done_n)
        """

        self.n_queue.append((s_atom, a_global, r, ns_atom, done))

        if len(self.n_queue) < self.n_step:
            return None

        # n-step 누적 reward
        R = 0.0
        g = 1.0
        for (_, _, r_i, _, _) in self.n_queue:
            R += r_i * g
            g *= self.gamma

        s0, a0, _, _, _ = self.n_queue[0]
        _, _, _, ns_n, d_n = self.n_queue[-1]

        return s0, a0, R, ns_n, d_n



    ###############################################################
    # Store transition
    ###############################################################
    def store(self, s_atom, a_global, r, ns_atom, done):
        """
        s_atom  : (N, FEAT)
        a_global: (3N,)
        ns_atom : (N, FEAT)
        """

        merged = self._merge_n_step(s_atom, a_global, r, ns_atom, done)
        if merged is None:
            return

        s0_atom, a0, Rn, ns_n_atom, d_n = merged

        idx = self.ptr

        # WRITE
        self.obs_buf[idx] = s0_atom
        self.act_buf[idx] = a0
        self.rew_buf[idx] = Rn
        self.nobs_buf[idx] = ns_n_atom
        self.done_buf[idx] = d_n

        # 초기 priority = |reward|
        self.prior_buf[idx] = abs(Rn) + 1e-6

        # episode tracking
        self.current_ep_indices.append(idx)

        # pointer
        self.ptr = (idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)



    ###############################################################
    # Sample (episode-aware PER)
    ###############################################################
    def sample(self, batch_size):

        if self.size == 0:
            raise RuntimeError("ReplayBuffer empty")

        # episode 없는 경우 → uniform sampling
        if len(self.episodes) == 0:
            idxs = np.random.randint(0, self.size, batch_size)
            w = np.ones(batch_size, np.float32)
            return self._package(idxs, w)

        # 1) episode 균등 선택
        chosen_eps = random.sample(
            self.episodes,
            k=min(batch_size, len(self.episodes))
        )

        idxs = []

        # 2) 각 episode 안에서 priority sampling
        for ep_idxs in chosen_eps:
            pri = self.prior_buf[ep_idxs]

            if np.sum(pri) < 1e-12:
                continue

            prob = pri ** self.alpha
            prob /= prob.sum()

            chosen_idx = np.random.choice(ep_idxs, p=prob)
            idxs.append(chosen_idx)

        # 부족분은 uniform random으로 채움
        if len(idxs) < batch_size:
            need = batch_size - len(idxs)
            extra = np.random.randint(0, self.size, need)
            idxs.extend(extra.tolist())

        idxs = np.array(idxs[:batch_size])

        # 3) PER importance sampling weight
        pr = self.prior_buf[idxs] + 1e-12
        prob = pr ** self.alpha
        prob /= prob.sum()

        w = (self.size * prob) ** (-self.beta)
        w /= w.max()

        return self._package(idxs, w.astype(np.float32))



    ###############################################################
    # Package batch for SACAgent.update()
    ###############################################################
    def _package(self, idxs, w):
        return dict(
            obs_atom = self.obs_buf[idxs],       # (B, N, FEAT)
            act      = self.act_buf[idxs],       # (B, 3N)
            rew      = self.rew_buf[idxs],       # (B,)
            nobs_atom= self.nobs_buf[idxs],      # (B, N, FEAT)
            done     = self.done_buf[idxs],      # (B,)
            weights  = w,                        # (B,)
            idx      = idxs
        )



    ###############################################################
    # Update TD-error priorities (from SACAgent)
    ###############################################################
    def update_priority(self, idxs, td_errors):
        td = np.abs(td_errors) + 1e-6
        for i, e in zip(idxs, td):
            self.prior_buf[i] = e



    ###############################################################
    def __len__(self):
        return self.size
