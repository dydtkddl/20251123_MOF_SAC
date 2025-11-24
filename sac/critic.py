import torch
import torch.nn as nn
import torch.nn.functional as F


###########################################################################
# Activation: MISH (Actor와 동일하게 사용)
###########################################################################
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


###########################################################################
# CriticV(s): Value Function (Deep & Stable)
###########################################################################
class CriticV(nn.Module):
    """
    V(s) network:
        Input : obs_dim (~30~40)
        Output: scalar state value
    """

    def __init__(self, obs_dim, hidden_sizes=[256, 256, 128, 64, 32]):
        super().__init__()

        layers = []
        in_dim = obs_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(Mish())
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))  # final output

        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        """
        obs: (batch, obs_dim)
        return: (batch, 1)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs)


###########################################################################
# CriticQ(s, a): Q-function for scalar action (scale ∈ [0,1])
###########################################################################
class CriticQ(nn.Module):
    """
    Q(s, a):

    Input:
        obs → (batch, obs_dim)
        act → (batch, 1)

    Output:
        Q-value → (batch, 1)
    """

    def __init__(self, obs_dim, act_dim=1, hidden_sizes=[256, 256, 128, 64, 32]):
        super().__init__()

        layers = []
        in_dim = obs_dim + act_dim  # concat(s, a)

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(Mish())
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, obs, act):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            act = act.unsqueeze(0)

        x = torch.cat([obs, act], dim=-1)
        return self.net(x)
