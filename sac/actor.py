###############################################################
# sac/actor.py — FINAL FULL-STABLE VERSION
###############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


######################################################################
# Smooth Activation (Swish)
######################################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


######################################################################
# Gaussian Policy → Squashed to [0, action_max]
# + Smoothing + act_tensor() support
######################################################################
class Actor(nn.Module):

    def __init__(
        self,
        obs_dim,
        act_dim=1,
        hidden_sizes=[256, 256, 128, 64],
        log_std_min=-4.0,
        log_std_max=1.0,
        action_max=0.12
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_max = action_max

        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max

        # ------------------------------------------------------------
        # Deep network backbone (LayerNorm + Swish)
        # ------------------------------------------------------------
        layers = []
        d = obs_dim
        for h in hidden_sizes:
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

        # Action smoothing memory
        self.prev_action = None


    ##################################################################
    # Forward pass: returns (action, logp, mu, std)
    ##################################################################
    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        h = self.net(obs)

        mu = self.mu_head(h)
        log_std = self.log_std_head(h)

        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Reparameterization trick
        eps = torch.randn_like(mu)
        raw = mu + eps * std

        # Softplus-based squash into [0,1]
        squashed = F.softplus(raw) / (1.0 + F.softplus(raw))

        # Convert to [0, action_max]
        scale = squashed * self.action_max

        # ------------------------------------------------------------
        # Log prob of Gaussian BEFORE squashing
        # ------------------------------------------------------------
        gauss_logp = (
            -0.5 * ((raw - mu) / (std + 1e-8)) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1, keepdim=True)

        # Jacobian correction term for softplus squash
        jac = torch.log(
            (1.0 / (1.0 + F.softplus(raw))) *
            (F.softplus(raw) / (1.0 + F.softplus(raw))) + 1e-10
        ).sum(dim=-1, keepdim=True)

        logp = gauss_logp - jac

        return scale, logp, mu, std


    ##################################################################
    # Deterministic action for evaluation (CPU)
    ##################################################################
    @torch.no_grad()
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        scale, _, _, _ = self.forward(obs_t)
        scale = scale.squeeze().cpu().numpy()

        # Smoothing (critical!)
        if self.prev_action is None:
            self.prev_action = scale
        else:
            scale = 0.7 * self.prev_action + 0.3 * scale
            self.prev_action = scale

        return scale


    ##################################################################
    # Deterministic action for GPU tensor input (main_train.py → agent)
    ##################################################################
    @torch.no_grad()
    def act_tensor(self, obs_t):
        """
        obs_t: already a GPU tensor of shape (1, obs_dim)
        """
        scale, _, _, _ = self.forward(obs_t)
        scale = scale.squeeze()

        # smoothing
        if self.prev_action is None:
            self.prev_action = scale
        else:
            scale = 0.7 * self.prev_action + 0.3 * scale
            self.prev_action = scale

        return scale.cpu().numpy()
