import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


######################################################################
# Activation: MISH (smoother than ReLU)
######################################################################
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


######################################################################
# Deep Gaussian Policy → Squashed to [0, 1]
######################################################################
class Actor(nn.Module):
    """
    Per-atom scalar policy for MOF structure optimization:
    - Input: obs_dim (~20~30)
    - Output: scale factor ∈ [0,1]
    - Gaussian sampling with tanh squash
    """

    def __init__(
        self,
        obs_dim,
        act_dim=1,
        hidden_sizes=[256, 256, 128, 64],
        log_std_min=-5.0,
        log_std_max=2.0
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim  # always 1 for scale-factor SAC

        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max

        # ------------------------------------------------------------
        # Deep MLP Backbone (LayerNorm + Mish)
        # ------------------------------------------------------------
        layers = []
        in_dim = obs_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(Mish())
            in_dim = h

        self.net = nn.Sequential(*layers)

        # Gaussian heads
        self.mu_head = nn.Linear(in_dim, act_dim)
        self.log_std_head = nn.Linear(in_dim, act_dim)

    ######################################################################
    def forward(self, obs):
        """
        SAC policy forward pass.

        Args:
            obs: (batch, obs_dim)

        Returns:
            scale: (batch, 1) ∈ [0,1]
            logp:  (batch, 1)
            mu:    (batch, 1)
            std:   (batch, 1)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        h = self.net(obs)

        # Gaussian parameters
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # ------------------------------------------------------------
        # Reparameterization trick
        # ------------------------------------------------------------
        eps = torch.randn_like(mu)
        raw_action = mu + eps * std

        # ------------------------------------------------------------
        # Squashing to [-1,1] → map to [0,1]
        # ------------------------------------------------------------
        squashed = torch.tanh(raw_action)
        scale = 0.5 * (squashed + 1.0)  # final action in [0,1]

        # ------------------------------------------------------------
        # Log probability with squash correction
        # ------------------------------------------------------------
        gaussian_logp = (
            -0.5 * ((raw_action - mu) / std) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1, keepdim=True)

        # Jacobian correction
        squash_correction = torch.log(1 - squashed.pow(2) + 1e-10)
        squash_correction = squash_correction.sum(dim=-1, keepdim=True)

        logp = gaussian_logp - squash_correction

        return scale, logp, mu, std

    ######################################################################
    @torch.no_grad()
    def act(self, obs):
        """
        Deterministic policy used during evaluation/inference.
        Args:
            obs: (obs_dim,)
        Returns:
            scale: float ∈ [0,1]
        """
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        h = self.net(obs)

        mu = self.mu_head(h)
        squashed = torch.tanh(mu)
        scale = 0.5 * (squashed + 1.0)

        return float(scale.cpu().numpy()[0])
