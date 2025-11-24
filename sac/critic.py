import torch
import torch.nn as nn
import torch.nn.functional as F


######################################################################
# Swish Activation
######################################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


######################################################################
# V(s)
######################################################################
class CriticV(nn.Module):

    def __init__(self, obs_dim, hidden=[256, 256, 128, 64, 32]):
        super().__init__()

        layers = []
        d = obs_dim

        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), Swish()]
            d = h

        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs)


######################################################################
# Q(s,a)
######################################################################
class CriticQ(nn.Module):

    def __init__(self, obs_dim, act_dim=1, hidden=[256, 256, 128, 64, 32]):
        super().__init__()

        layers = []
        d = obs_dim + act_dim

        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), Swish()]
            d = h

        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs, act):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            act = act.unsqueeze(0)

        x = torch.cat([obs, act], dim=-1)
        return self.net(x)
