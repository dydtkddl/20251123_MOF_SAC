###############################################################
# sac/critic.py — MACS 3D-Action Fully Compatible FINAL VERSION
# - CriticV: V(s)
# - CriticQ: Q(s,a)  (3D action version)
# - TwinCriticQ: SAC Double-Q (Q1, Q2)
###############################################################

import torch
import torch.nn as nn
import numpy as np

###############################################################
# Swish activation (same as actor)
###############################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


###############################################################
# Utility: build a MACS-style MLP block (Swish + LayerNorm)
###############################################################
def build_mlp(in_dim, hidden_sizes, out_dim, final_act=False):
    layers = []
    d = in_dim

    for h in hidden_sizes:
        layers += [
            nn.Linear(d, h),
            nn.LayerNorm(h),
            Swish()
        ]
        d = h

    layers.append(nn.Linear(d, out_dim))

    if final_act:
        layers.append(Swish())

    return nn.Sequential(*layers)


###############################################################
# Value Critic: V(s)
###############################################################
class CriticV(nn.Module):
    """
    V(s) network.
    Input: obs_dim
    Output: scalar value V(s)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden=[256, 256, 128, 64]
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.net = build_mlp(
            in_dim=obs_dim,
            hidden_sizes=hidden,
            out_dim=1,
            final_act=False
        )

    def forward(self, obs: torch.Tensor):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs)


###############################################################
# Single Q-network: Q(s,a)
###############################################################
class CriticQ(nn.Module):
    """
    Q(s,a) network.
    Input:
        obs: (B, obs_dim)
        act: (B, 3)   # ★ 3D action
    Output:
        Q-value scalar
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 3,
        hidden=[256, 256, 128, 64]
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.net = build_mlp(
            in_dim=obs_dim + act_dim,
            hidden_sizes=hidden,
            out_dim=1,
            final_act=False
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        """
        obs: (B, obs_dim)
        act: (B, 3)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if act.dim() == 1:
            act = act.unsqueeze(0)

        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


###############################################################
# Twin Critic for SAC: Q1(s,a), Q2(s,a)
###############################################################
class TwinCriticQ(nn.Module):
    """
    Two independent Q networks for SAC
    (Q1, Q2) to mitigate positive bias.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 3,
        hidden=[256, 256, 128, 64]
    ):
        super().__init__()

        self.Q1 = CriticQ(obs_dim, act_dim, hidden)
        self.Q2 = CriticQ(obs_dim, act_dim, hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        """
        returns:
            Q1(s,a), Q2(s,a)
        """
        return self.Q1(obs, act), self.Q2(obs, act)
