###############################################################
# sac/critic.py — MACS 3D-Action Fully Compatible FINAL VERSION
###############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################
# Swish activation (same as actor)
###############################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


###############################################################
# Value Critic V(s)
###############################################################
class CriticV(nn.Module):

    def __init__(
        self,
        obs_dim,
        hidden=[256, 256, 128, 64]
    ):
        """
        V(s) network.
        Input: obs_dim
        Output: scalar V(s)
        """

        super().__init__()

        layers = []
        d = obs_dim

        # Deep backbone: LN + Swish (MACS stability)
        for h in hidden:
            layers += [
                nn.Linear(d, h),
                nn.LayerNorm(h),
                Swish()
            ]
            d = h

        # Output layer
        layers += [nn.Linear(d, 1)]

        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        """
        obs: (B, obs_dim)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        return self.net(obs)


###############################################################
# Action-Value Critic Q(s,a)
###############################################################
class CriticQ(nn.Module):

    def __init__(
        self,
        obs_dim,
        act_dim=3,                 # ★ 3D action for MACS
        hidden=[256, 256, 128, 64]
    ):
        """
        Q(s,a) network.
        Input: concat(obs, action)
        Output: scalar Q(s,a)
        """

        super().__init__()

        layers = []
        d = obs_dim + act_dim      # ★ MUST match actor's 3D action

        # Deep backbone
        for h in hidden:
            layers += [
                nn.Linear(d, h),
                nn.LayerNorm(h),
                Swish()
            ]
            d = h

        # Output
        layers += [nn.Linear(d, 1)]

        self.net = nn.Sequential(*layers)

    def forward(self, obs, act):
        """
        obs: (B, obs_dim)
        act: (B, 3)
        """

        # single sample → batchify
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if act.dim() == 1:
            act = act.unsqueeze(0)

        # Concatenate in correct order
        x = torch.cat([obs, act], dim=-1)

        return self.net(x)
