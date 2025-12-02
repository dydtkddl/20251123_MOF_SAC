# sac/critic.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def init_layer_uniform(layer: nn.Linear, w: float = 3e-3):
    layer.weight.data.uniform_(-w, w)
    layer.bias.data.uniform_(-w, w)
    return layer


class CriticQ(nn.Module):
    """
    Per-Atom Q Network: Q(obs_i, act_i)

    - Input:
        obs : (batch, obs_dim)
        act : (batch, act_dim)
    - Output:
        Q-value : (batch, 1)
    """

    def __init__(self, obs_dim: int, act_dim: int = 4, hidden_dim: int = 256):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

        init_layer_uniform(self.out)

        logger.info(
            f"[CriticQ] Initialized: obs_dim={obs_dim}, act_dim={act_dim}, "
            f"hidden_dim={hidden_dim}"
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : torch.Tensor
            Shape (batch, obs_dim)
        act : torch.Tensor
            Shape (batch, act_dim)

        Returns
        -------
        q : torch.Tensor
            Shape (batch, 1)
        """
        x = torch.cat([obs.float(), act.float()], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class CriticV(nn.Module):
    """
    Per-Atom V Network: V(obs_i)

    - Input:
        obs : (batch, obs_dim)
    - Output:
        V-value : (batch, 1)
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

        init_layer_uniform(self.out)

        logger.info(
            f"[CriticV] Initialized: obs_dim={obs_dim}, hidden_dim={hidden_dim}"
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
