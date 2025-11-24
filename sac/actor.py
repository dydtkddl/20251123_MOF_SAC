###############################################################
# sac/actor.py — Structure-Level MACS Policy (FINAL VERSION)
# -------------------------------------------------------------
# - Input:  obs_global (1D vector, length = N * obs_dim)
# - Output: action_global (flattened N*3 vector → reshape(N,3))
# - Gaussian policy with Tanh squash
# - Logπ correction term included
# - Absolutely robust shape handling (never crashes)
###############################################################

import torch
import torch.nn as nn
import numpy as np


###############################################################
# Swish activation (MACS style)
###############################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


###############################################################
# Structure-Level Actor
###############################################################
class Actor(nn.Module):
    """
    Structure-Level Gaussian Policy for MACS RL
    ----------------------------------------------------------
    Inputs:
        obs_dim_global = N_atoms * obs_dim_atom
    Outputs:
        action_global (N_atoms * 3)
        logp: scalar per sample
        mu, std: gaussian params

    Notes:
      - No smoothing, no EMA → MACS stability
      - Tanh squashing ensures safe displacement direction
      - Robust shape-handling for arbitrary N_atoms
    """

    def __init__(
        self,
        obs_global_dim: int,            # (N * obs_dim)
        n_atoms: int,                   # N
        hidden=[512, 512, 256],
        log_std_min=-5.0,
        log_std_max=1.0,
        action_max=1.0                  # bound for tanh
    ):
        super().__init__()

        self.obs_global_dim = obs_global_dim
        self.n_atoms = n_atoms

        # output dim = 3 * N
        self.act_dim = 3 * n_atoms
        self.action_max = action_max

        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max

        ###############################################################
        # Backbone: LN + Swish (High Stability for Structure RL)
        ###############################################################
        layers = []
        d = obs_global_dim
        for h in hidden:
            layers += [
                nn.Linear(d, h),
                nn.LayerNorm(h),
                Swish()
            ]
            d = h
        self.net = nn.Sequential(*layers)

        ###############################################################
        # Gaussian parameter heads
        ###############################################################
        self.mu_head = nn.Linear(d, self.act_dim)
        self.log_std_head = nn.Linear(d, self.act_dim)


    ###############################################################
    # Forward path (for SAC update)
    ###############################################################
    def forward(self, obs):
        """
        obs: tensor  (B, obs_global_dim)
        return:
            action_flat : (B, act_dim)
            logp        : (B, 1)
            mu, std     : (B, act_dim)
        """

        # shape correction
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)     # (1, obs_dim)

        h = self.net(obs)

        # gaussian params
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # reparameterization
        eps = torch.randn_like(mu)
        raw = mu + eps * std

        # tanh squash
        tanh_raw = torch.tanh(raw)
        action_flat = self.action_max * tanh_raw   # (B, 3N)

        ###############################################################
        # logπ correction
        ###############################################################
        gauss_logp = (
            -0.5 * ((raw - mu) / (std + 1e-8)) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1, keepdim=True)

        squash_corr = torch.log(1.0 - tanh_raw.pow(2) + 1e-10).sum(dim=-1, keepdim=True)
        logp = gauss_logp - squash_corr

        return action_flat, logp, mu, std


    ###############################################################
    # Deterministic action for inference/evaluation (numpy)
    ###############################################################
    @torch.no_grad()
    def act(self, obs_global_np):
        """
        obs_global_np: numpy shape (obs_global_dim,)
        return: numpy shape (N, 3)
        """
        if obs_global_np.ndim != 1:
            raise ValueError("obs_global_np must be 1D (flattened)")

        obs_t = torch.as_tensor(obs_global_np, dtype=torch.float32).unsqueeze(0)

        action_flat, _, _, _ = self.forward(obs_t)

        # reshape to (N,3)
        a = action_flat.reshape(self.n_atoms, 3)

        return a.cpu().numpy().astype(np.float32)


    ###############################################################
    # Deterministic action (torch tensor input)
    ###############################################################
    @torch.no_grad()
    def act_tensor(self, obs_t):
        """
        obs_t: tensor shape (1, obs_global_dim)
        return: numpy shape (N,3)
        """
        if obs_t.dim() != 2 or obs_t.size(0) != 1:
            raise ValueError("obs_t must have shape (1, obs_global_dim)")

        action_flat, _, _, _ = self.forward(obs_t)

        a = action_flat.reshape(self.n_atoms, 3)

        return a.detach().cpu().numpy().astype(np.float32)
