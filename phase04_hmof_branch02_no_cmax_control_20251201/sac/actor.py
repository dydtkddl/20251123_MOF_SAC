# sac/actor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3):
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


class Actor(nn.Module):
    """
    Per-Atom Actor Network
    obs_dim -> (mu, std) -> action(3,)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 3,     # ALWAYS 3 (dx,dy,dz)
        hidden_dim: int = 256,
        log_std_min: float = -10,
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

    def forward(self, obs):
        """
        obs: (batch, obs_dim)
        return: action(3,), log_prob, mu, std
        """

        # FP32 강화
        obs = obs.float()

        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        mu = self.mu_layer(x)

        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # Gaussian reparameterization
        dist = Normal(mu, std)
        z = dist.rsample()

        # Output action in [-1,1]
        action = torch.tanh(z)

        # Tanh correction
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, mu, std
