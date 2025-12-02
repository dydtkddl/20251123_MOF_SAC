# sac/actor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import logging

logger = logging.getLogger(__name__)


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3):
    """Small uniform init for output layers (SAC style)."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


class Actor(nn.Module):
    """
    Per-Atom Actor Network (for MACS-MOF SAC)

    - Input : obs (batch, obs_dim)
    - Output: action (batch, act_dim) in [-1, 1]
              + log_prob (batch, 1)
              + mu, std (for diagnostics)

    For 4D action:
        action[..., 0] : gate (to be mapped [-1,1] -> [0,1] in env)
        action[..., 1:] : (dx, dy, dz)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 4,          # default 4: (gate, dx, dy, dz)
        hidden_dim: int = 256,
        log_std_min: float = -10.0,
        log_std_max: float = 1.5,
    ):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_dim = act_dim

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)

        init_layer_uniform(self.mu_layer)
        init_layer_uniform(self.log_std_layer)

        logger.info(
            f"[Actor] Initialized: obs_dim={obs_dim}, act_dim={act_dim}, "
            f"hidden_dim={hidden_dim}"
        )

    def forward(self, obs: torch.Tensor):
        """
        Parameters
        ----------
        obs : torch.Tensor
            Shape (batch, obs_dim)

        Returns
        -------
        action : torch.Tensor
            Shape (batch, act_dim), in [-1, 1]
        log_prob : torch.Tensor
            Shape (batch, 1)
        mu : torch.Tensor
            Shape (batch, act_dim)
        std : torch.Tensor
            Shape (batch, act_dim)
        """
        obs = obs.float()

        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        mu = self.mu_layer(x)

        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()  # reparameterization

        # squash to [-1, 1]
        action = torch.tanh(z)

        # Tanh correction term
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mu, std
