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

    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_dim=256,
                 log_std_min=-20,
                 log_std_max=2):

        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)

        init_layer_uniform(self.mu_layer)
        init_layer_uniform(self.log_std_layer)


    def forward(self, obs):

        # **ADD THIS BLOCK**
        if not torch.is_floating_point(obs):
            obs = obs.float()
        elif obs.dtype != torch.float32:
            obs = obs.float()
        # -------------------

        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        mu = self.mu_layer(x)

        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        std = torch.exp(log_std)

        # gaussian
        dist = Normal(mu, std)

        # reparameterization
        z = dist.rsample()

        # tanh policy
        action = torch.tanh(z)

        # log prob correction term
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)

        return action, log_prob, mu, std
