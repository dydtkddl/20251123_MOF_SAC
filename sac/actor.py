###############################################################
# sac/actor.py — MACS-style 3D Gaussian Policy (FINAL + FIXED)
###############################################################

import torch
import torch.nn as nn
import numpy as np


###############################################################
# Swish: MACS-like smooth activation
###############################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


###############################################################
# Actor Network (3D Gaussian Policy + Tanh squash)
###############################################################
class Actor(nn.Module):
    """
    MACS-compatible 3D continuous control policy:
    - Gaussian reparameterization (μ, σ)
    - tanh squashing to [-1, 1]^3
    - NO smoothing (global drift 방지)
    - Robust action shape handling (절대 crash 안남)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 3,                  # ★ 3D vector action
        hidden=[256, 256, 128],
        log_std_min=-5.0,
        log_std_max=1.0,
        action_max=1.0                     # [-1,1]^3 range
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_max = action_max

        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max

        ###############################################################
        # Backbone: LN + Swish (MACS style)
        ###############################################################
        layers = []
        d = obs_dim
        for h in hidden:
            layers += [
                nn.Linear(d, h),
                nn.LayerNorm(h),
                Swish()
            ]
            d = h
        self.net = nn.Sequential(*layers)

        ###############################################################
        # Gaussian heads
        ###############################################################
        self.mu_head = nn.Linear(d, act_dim)
        self.log_std_head = nn.Linear(d, act_dim)

        # smoothing 제거
        self.prev_action = None


    ###############################################################
    # Forward: stochastic action & logπ
    ###############################################################
    def forward(self, obs: torch.Tensor):
        """
        obs: Tensor (B, obs_dim)
        returns:
            action: (B, act_dim)
            logp:   (B, 1)
            mu:     (B, act_dim)
            std:    (B, act_dim)
        """

        # ensure (B, obs_dim)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        h = self.net(obs)

        # Gaussian params
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # reparameterization
        eps = torch.randn_like(mu)
        raw = mu + eps * std

        # tanh squash
        tanh_raw = torch.tanh(raw)
        action = self.action_max * tanh_raw

        # logπ with correction term
        gauss_logp = (
            -0.5 * ((raw - mu) / (std + 1e-8)) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1, keepdim=True)

        squash_corr = torch.log(1.0 - tanh_raw.pow(2) + 1e-10).sum(dim=-1, keepdim=True)

        logp = gauss_logp - squash_corr

        return action, logp, mu, std


    ###############################################################
    # Deterministic: obs (np array) → (3,) np vector
    ###############################################################
    @torch.no_grad()
    def act(self, obs_np: np.ndarray):
        """
        obs_np: numpy array of shape (obs_dim,)
        return: (3,) numpy float32
        """
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        action, _, _, _ = self.forward(obs)

        # robust shape normalizer (절대 crash 안남)
        a = action.reshape(-1)[:self.act_dim]     # (3,)
        return a.cpu().numpy().astype(np.float32)


    ###############################################################
    # Deterministic for GPU tensor input
    ###############################################################
    @torch.no_grad()
    def act_tensor(self, obs_t: torch.Tensor):
        """
        obs_t: (1, obs_dim) GPU tensor
        return: (3,) numpy
        """
        action, _, _, _ = self.forward(obs_t)

        # robust flattening
        a = action.reshape(-1)[:self.act_dim]

        return a.detach().cpu().numpy().astype(np.float32)
