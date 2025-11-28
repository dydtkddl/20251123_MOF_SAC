# sac/critic.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer_uniform(layer, w=3e-3):
    layer.weight.data.uniform_(-w, w)
    layer.bias.data.uniform_(-w, w)
    return layer


class CriticQ(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dim=256):

        super().__init__()

        self.fc1 = nn.Linear(obs_dim+act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

        init_layer_uniform(self.out)


    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)



class CriticV(nn.Module):

    def __init__(self, obs_dim, hidden_dim=256):

        super().__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

        init_layer_uniform(self.out)


    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.out(x)
