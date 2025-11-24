import numpy as np
from collections import deque
import random


class PrioritizedReplayBuffer:
    """
    Upgraded MACS-style ReplayBuffer
    ---------------------------------------------------------
    - PER (Prioritized Experience Replay)
    - Sliding window (keep recent MAX only)
    - Episode-balanced sampling
    - N-step return support
    ---------------------------------------------------------
    Stores *per-atom transitions*:
        obs_i       : (obs_dim,)
        act_i       : (3,)
        reward_i    : float
        next_obs_i  : (obs_dim,)
        done        : bool
        episode_id  : int
        priority    : float
    """

    def __init__(
        self,
        obs_dim: int,
        max_size: int = 2_000_000,     # Sliding window target
        alpha: float = 0.6,            # PER exponent
        beta: float = 0.4,             # importance-sampling
        n_step: int = 3,
        gamma: float = 0.995,
    ):
        self.obs_dim = obs_dim
        self.act_dim = 3

        # Sliding window max
        self.max_size = max_size

        # PER parameters
        self.alpha = alpha
        self.beta = beta

        # Episode tracking
        self.episode_id_counter = 0
        self.episode_track = []      # stores indices belonging to each episode

        # Internal storage
        self.ptr = 0
        self.size = 0

        # Buffers
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, 3), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.nobs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.bool_)

        # PER priority
        self.prior_buf = np.zeros(max_size, dtype=np.float32) + 1e-6

        # N-step
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_queue = deque(maxlen=n_step)


    ##########################################################
    # Start new episode
    ##########################################################
    def new_episode(self):
        """
        Must be called at the start of each episode.
        """
        self.episode_id_counter += 1
        self.current_episode_indices = []


    ##########################################################
    # N-step helper
    ##########################################################
    def _get_n_step_transition(self, s, a, r, ns, d):
        """
        Compute n-step reward and next state.
        """

        # Add current transition to queue
        self.n_step_queue.append((s, a, r, ns, d))

        # Not enough transitions yet
        if len(self.n_step_queue) < self.n_step:
            return None

        # Compute n-step reward
        R = 0
        discount = 1
        for (_, _, r_i, _, _) in self.n_step_queue:
            R += r_i * discount
            discount *= self.gamma

        # n-step next_state and done
        _, _, _, next_obs_n, done_n = self.n_step_queue[-1]

        (s0, a0, _, _, d0) = self.n_step_queue[0]
        return s0, a0, R, next_obs_n, done_n


    ##########################################################
    # Store transition
    ##########################################################
    def store(self, obs_i, act_i, rew_i, next_obs_i, done_i):
        """
        Store N-step transition + PER priority update.
        """

        nstep = self._get_n_step_transition(obs_i, act_i, rew_i, next_obs_i, done_i)

        # Not enough steps → skip for now
        if nstep is None:
            return

        obs0, act0, Rn, next_obs_n, done_n = nstep

        # Save to buffer
        idx = self.ptr

        self.obs_buf[idx] = obs0
        self.act_buf[idx] = act0
        self.rew_buf[idx] = Rn
        self.nobs_buf[idx] = next_obs_n
        self.done_buf[idx] = done_n

        # PER priority = |reward| + eps
        self.prior_buf[idx] = abs(float(Rn)) + 1e-6

        # Episode tracking
        self.current_episode_indices.append(idx)

        # Pointer update
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        

    ##########################################################
    # Finalize episode
    ##########################################################
    def end_episode(self, keep: bool):
        """
        keep=False → remove episode transitions (COM/bond episode)
        keep=True  → commit episode transitions
        """
        if not keep:
            # Erase transitions by zeroing priority
            for idx in self.current_episode_indices:
                self.prior_buf[idx] = 0
            self.current_episode_indices = []
        else:
            # Add episode ID in global track
            self.episode_track.append(self.current_episode_indices)
            self.current_episode_indices = []


    ##########################################################
    # Sliding Window Maintenance
    ##########################################################
    def _apply_sliding_window(self):
        """
        Zero out old transitions (priority=0) when buffer is full.
        """
        if self.size == self.max_size:
            # Lower priority → more likely to get overwritten
            small_indices = np.argsort(self.prior_buf)[:100000]  # remove 100k
            for idx in small_indices:
                self.prior_buf[idx] = 0.0


    ##########################################################
    # PER + Episode-balanced Sampling
    ##########################################################
    def sample(self, batch_size):
        """
        PER sampling + Episode-balanced sampling
        """

        # Sample episodes uniformly
        episodes = random.sample(self.episode_track, k=min(len(self.episode_track), batch_size))

        indices = []
        for ep in episodes:
            # PER sampling inside the episode
            prios = self.prior_buf[ep]
            if prios.sum() < 1e-9:
                continue
            probs = prios ** self.alpha
            probs /= probs.sum()
            idx = np.random.choice(ep, p=probs)
            indices.append(idx)

        # Fallback: if insufficient, random sample
        if len(indices) < batch_size:
            extra = batch_size - len(indices)
            random_ids = np.random.randint(0, self.size, size=extra)
            indices.extend(random_ids)

        indices = np.array(indices[:batch_size])

        # PER importance sampling weights
        priorities = self.prior_buf[indices]
        probs = priorities / priorities.sum()
        weights = (len(self.prior_buf) * probs) ** (-self.beta)
        weights /= weights.max()  # normalize

        return dict(
            obs=self.obs_buf[indices],
            act=self.act_buf[indices],
            rew=self.rew_buf[indices],
            nobs=self.nobs_buf[indices],
            done=self.done_buf[indices],
            weights=weights.astype(np.float32),
            idx=indices,
        )


    ##########################################################
    def update_priority(self, idx, new_priority):
        """
        Optional: agent can update priority after TD error.
        """
        self.prior_buf[idx] = abs(new_priority) + 1e-6


    ##########################################################
    def __len__(self):
        return self.size
