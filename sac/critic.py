# sac/critic.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer_uniform(layer, w=3e-3):
    layer.weight.data.uniform_(-w, w)
    layer.bias.data.uniform_(-w, w)
    return layer


class CriticQ(nn.Module):
    """
    Per-Atom Q Network
    Q(obs_i, act_i)
    """

    def __init__(self, obs_dim: int, act_dim: int = 3, hidden_dim: int = 256):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

        init_layer_uniform(self.out)

    def forward(self, obs, act):
        # obs  (batch, obs_dim)
        # act  (batch, 3)
        x = torch.cat([obs.float(), act.float()], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class CriticV(nn.Module):
    """
    Per-Atom V Network
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

        init_layer_uniform(self.out)

    def forward(self, obs):
        x = obs.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
