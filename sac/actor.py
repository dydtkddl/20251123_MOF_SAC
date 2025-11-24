###############################################################
# sac/actor.py — Structure-Level MACS Policy (FINAL VERSION)
# -------------------------------------------------------------
# Input:
#   obs_global: (flattened) shape = N_atoms * obs_dim_atom
# Output:
#   action_global_flat: (3*N)  → reshape(N,3) 가능
#
# 특징:
#   - Gaussian policy with tanh squash
#   - Logπ correction
#   - LayerNorm + Swish backbone (MACS 안정성)
###############################################################

import torch
import torch.nn as nn
import numpy as np


###############################################################
# Swish
###############################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


###############################################################
# Structure-Level Actor
###############################################################
class Actor(nn.Module):
    def __init__(
        self,
        obs_global_dim: int,          # N * obs_dim_atom (flatten)
        n_atoms: int,                 # N
        hidden=[512, 512, 256],
        log_std_min=-5.0,
        log_std_max=1.0,
        action_max=1.0
    ):
        super().__init__()

        self.obs_global_dim = obs_global_dim
        self.n_atoms = n_atoms

        # output_dim = 3*N
        self.act_dim = 3 * n_atoms
        self.action_max = action_max

        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max

        ###############################################################
        # Backbone MLP
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

        # Gaussian heads
        self.mu_head = nn.Linear(d, self.act_dim)
        self.log_std_head = nn.Linear(d, self.act_dim)


    ###############################################################
    # SAC training forward
    ###############################################################
    def forward(self, obs):
        """
        obs: (B, obs_dim_global) or (obs_dim_global,)
        return:
           action_flat: (B, act_dim)
           logp: (B,1)
           mu, std
        """

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        h = self.net(obs)

        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # reparameterization
        eps = torch.randn_like(mu)
        raw = mu + eps * std
        tanh_raw = torch.tanh(raw)

        action = self.action_max * tanh_raw   # (B, 3N)

        ###############################################################
        # log π(a|s) with change-of-variables correction
        ###############################################################
        gauss_logp = (
            -0.5 * ((raw - mu) / (std + 1e-8)) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1, keepdim=True)

        squash_corr = torch.log(1.0 - tanh_raw.pow(2) + 1e-10).sum(dim=-1, keepdim=True)
        logp = gauss_logp - squash_corr

        return action, logp, mu, std


    ###############################################################
    # Deterministic action (numpy, env rollout)
    ###############################################################
    @torch.no_grad()
    def act(self, obs_global_np):
        """
        obs_global_np: numpy 1D flatten (obs_global_dim,)
        return: (N,3) numpy
        """
        assert obs_global_np.ndim == 1, "obs_global_np must be a 1D flattened vector."

        obs_t = torch.as_tensor(obs_global_np, dtype=torch.float32).unsqueeze(0)

        a_flat, _, _, _ = self.forward(obs_t)
        a = a_flat.reshape(self.n_atoms, 3)

        return a.cpu().numpy().astype(np.float32)


    ###############################################################
    # Deterministic action for tensor input
    ###############################################################
    @torch.no_grad()
    def act_tensor(self, obs_t):
        """
        obs_t: tensor shape (1, obs_global_dim)
        return: (N,3) numpy
        """
        if obs_t.dim() != 2 or obs_t.size(0) != 1:
            raise ValueError("obs_t must be (1, obs_global_dim)")

        a_flat, _, _, _ = self.forward(obs_t)
        a = a_flat.reshape(self.n_atoms, 3)

        return a.cpu().numpy().astype(np.float32)
