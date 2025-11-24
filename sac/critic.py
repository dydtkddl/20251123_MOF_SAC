###############################################################
# sac/critic.py — Structure-Level Critic (MACS Global Policy)
# -------------------------------------------------------------
# - CriticV: V(s_global)
# - CriticQ: Q(s_global, a_global)
# - TwinCriticQ: SAC Double-Q
#
# Input:
#   obs_global_dim = N_atoms * obs_dim_atom
#   act_global_dim = N_atoms * 3
#
# NOTES:
#   * No per-atom critic
#   * Pure structure-level critic for stable MACS RL
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
# Utility: MACS MLP (LN + Swish)
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
# Value Critic: V(s_global)
###############################################################
class CriticV(nn.Module):
    """
    Structure-level V(s):
        Input  : obs_global (flattened)
        Output : scalar V(s)
    """

    def __init__(self,
                 obs_global_dim: int,
                 hidden=[512, 512, 256]):
        super().__init__()

        self.obs_global_dim = obs_global_dim

        self.net = build_mlp(
            in_dim=obs_global_dim,
            hidden_sizes=hidden,
            out_dim=1,
            final_act=False
        )


    def forward(self, obs_global: torch.Tensor):
        """
        obs_global: (B, obs_global_dim) or (obs_global_dim,)
        """
        if obs_global.dim() == 1:
            obs_global = obs_global.unsqueeze(0)
        return self.net(obs_global)  # (B,1)


###############################################################
# Q Critic: Q(s_global, a_global)
###############################################################
class CriticQ(nn.Module):
    """
    Structure-level Q(s,a)
    ----------------------
    Inputs:
        obs_global: (B, obs_global_dim)
        act_global: (B, act_global_dim)

    Output:
        Q-value scalar
    """

    def __init__(self,
                 obs_global_dim: int,
                 act_global_dim: int,
                 hidden=[512, 512, 256]):
        super().__init__()

        self.obs_global_dim = obs_global_dim
        self.act_global_dim = act_global_dim

        input_dim = obs_global_dim + act_global_dim

        self.net = build_mlp(
            in_dim=input_dim,
            hidden_sizes=hidden,
            out_dim=1,
            final_act=False
        )


    def forward(self, obs_global: torch.Tensor, act_global: torch.Tensor):
        """
        obs_global: (B, obs_global_dim)  OR  (obs_global_dim,)
        act_global: (B, act_global_dim)  OR  (act_global_dim,)
        """

        # shape-normalization → never crash
        if obs_global.dim() == 1:
            obs_global = obs_global.unsqueeze(0)
        if act_global.dim() == 1:
            act_global = act_global.unsqueeze(0)

        x = torch.cat([obs_global, act_global], dim=-1)
        return self.net(x)  # (B,1)


###############################################################
# Twin Critic (SAC Double-Q)
###############################################################
class TwinCriticQ(nn.Module):
    """
    Independent Q1, Q2 networks
    """
    def __init__(self,
                 obs_global_dim: int,
                 act_global_dim: int,
                 hidden=[512, 512, 256]):
        super().__init__()

        self.Q1 = CriticQ(obs_global_dim, act_global_dim, hidden)
        self.Q2 = CriticQ(obs_global_dim, act_global_dim, hidden)


    def forward(self, obs_global: torch.Tensor, act_global: torch.Tensor):
        """
        return: Q1(s,a), Q2(s,a)
        """
        return self.Q1(obs_global, act_global), self.Q2(obs_global, act_global)
