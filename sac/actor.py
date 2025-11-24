###############################################################
# sac/actor.py — MACS-style 3D Gaussian Policy (FINAL VERSION)
###############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


###############################################################
# Swish activation (smooth MLP)
###############################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


###############################################################
# Actor: 3D Gaussian → Tanh squash → action vector in [-a_max, a_max]^3
###############################################################
class Actor(nn.Module):

    def __init__(
        self,
        obs_dim,
        act_dim=3,                      # ★ MACS: 3D action
        hidden_sizes=[256, 256, 128],
        log_std_min=-5.0,
        log_std_max=1.0,
        action_max=1.0                  # ★ RL world: MACS uses u ∈ [-1,1]^3
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_max = action_max

        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max

        ###########################################################
        # Backbone MLP + LayerNorm + Swish
        ###########################################################
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

        # temporal smoothing memory
        self.prev_action = None


    ###############################################################
    # Forward (training mode): returns
    #   π(a|s) sample, logp(a|s), μ, std
    ###############################################################
    def forward(self, obs):
        """
        obs: Tensor (B, obs_dim)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        h = self.net(obs)

        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # ---------------------------------------------------------
        # Reparameterization Trick: raw action
        # ---------------------------------------------------------
        eps = torch.randn_like(mu)
        raw = mu + eps * std

        # ---------------------------------------------------------
        # Squash with tanh (MACS style: u ∈ [-1,1]^3)
        # ---------------------------------------------------------
        a_tanh = torch.tanh(raw)
        action = self.action_max * a_tanh

        # ---------------------------------------------------------
        # Compute log probability with Jacobian correction
        # logπ(a|s) = logπ_raw(raw|s) - Σ log(1 - tanh(raw)^2)
        # ---------------------------------------------------------
        gauss_logp = (
            -0.5 * ((raw - mu) / (std + 1e-8)) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1, keepdim=True)

        # tanh squash correction term
        squash_corr = torch.log(
            1.0 - a_tanh.pow(2) + 1e-10
        ).sum(dim=-1, keepdim=True)

        logp = gauss_logp - squash_corr

        return action, logp, mu, std


    ###############################################################
    # Deterministic action for evaluation (CPU)
    ###############################################################
    @torch.no_grad()
    def act(self, obs):
        """
        obs: numpy array (obs_dim,)
        returns: numpy array (3,)  <- 3D action vector
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _, _, _ = self.forward(obs_t)
        a = action.squeeze().cpu().numpy()

        # smoothing
        if self.prev_action is None:
            self.prev_action = a
        else:
            a = 0.7 * self.prev_action + 0.3 * a
            self.prev_action = a

        return a


    ###############################################################
    # Deterministic action for GPU tensor input
    ###############################################################
    @torch.no_grad()
    def act_tensor(self, obs_t):
        """
        obs_t: torch tensor (1, obs_dim) on GPU
        return numpy action (3,)
        """
        action, _, _, _ = self.forward(obs_t)
        a = action.squeeze()

        # smoothing
        if self.prev_action is None:
            self.prev_action = a
        else:
            a = 0.7 * self.prev_action + 0.3 * a
            self.prev_action = a

        return a.detach().cpu().numpy()
