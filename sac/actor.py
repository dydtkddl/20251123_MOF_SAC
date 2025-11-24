###############################################################
# sac/actor.py — MACS-style 3D Gaussian Policy (FINAL VERSION)
###############################################################

import torch
import torch.nn as nn
import numpy as np


###############################################################
# Swish Activation
###############################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


###############################################################
# Actor: 3D Gaussian → Tanh Squash → [-1,1]^3
###############################################################
class Actor(nn.Module):
    """
    MACS-compatible actor network:
    - 3D vector action
    - Gaussian reparameterization
    - Tanh squashing
    - NO smoothing (smoothing = major reason for global drift)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 3,                 # ★ MACS-vector
        hidden=[256, 256, 128],
        log_std_min=-5.0,
        log_std_max=1.0,
        action_max=1.0                    # ★ MACS: [-1,1]^3
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_max = action_max
        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max

        ###############################################################
        # Backbone: LayerNorm + Swish (MACS-like smooth features)
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

        # Gaussian heads
        self.mu_head = nn.Linear(d, act_dim)
        self.log_std_head = nn.Linear(d, act_dim)

        # smoothing 제거 (MACS stability)
        self.prev_action = None


    ###############################################################
    # Forward: returns stochastic action + logπ(a|s)
    ###############################################################
    def forward(self, obs: torch.Tensor):
        """
        obs: (B, obs_dim)
        returns:
            action: (B, act_dim)
            logp:   (B, 1)
            mu:     (B, act_dim)
            std:    (B, act_dim)
        """

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        h = self.net(obs)

        # Gaussian parameters
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Reparameterization trick
        eps = torch.randn_like(mu)
        raw = mu + eps * std

        # Tanh squash for directional unit vector
        tanh_raw = torch.tanh(raw)
        action = self.action_max * tanh_raw

        # Logπ with squash correction
        # logπ = logN(raw|mu,std) - Σ log(1 - tanh(raw)^2)
        gauss_logp = (
            -0.5 * ((raw - mu) / (std + 1e-8)) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1, keepdim=True)

        squash_corr = torch.log(
            1.0 - tanh_raw.pow(2) + 1e-10
        ).sum(dim=-1, keepdim=True)

        logp = gauss_logp - squash_corr

        return action, logp, mu, std


    ###############################################################
    # Deterministic inference for evaluation
    ###############################################################
    @torch.no_grad()
    def act(self, obs_np: np.ndarray):
        """
        obs_np: numpy array (obs_dim,)
        return: numpy array (3,)
        """
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        action, _, _, _ = self.forward(obs)
        a = action.squeeze().cpu().numpy()

        # smoothing 제거 (넘어오던 문제)
        return a


    ###############################################################
    # Deterministic action for batched GPU tensor
    ###############################################################
    @torch.no_grad()
    def act_tensor(self, obs_t: torch.Tensor):
        """
        obs_t: (1, obs_dim) GPU tensor
        return: numpy array (3,)
        """
        action, _, _, _ = self.forward(obs_t)
        a = action.squeeze()
        return a.detach().cpu().numpy()
