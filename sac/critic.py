###############################################################
# sac/critic.py — Structure-Level Critic (MACS Global Policy)
# -------------------------------------------------------------
#  - CriticV : V(s_global)
#  - CriticQ : Q(s_global, a_global)
#  - TwinCriticQ : Double Q (SAC)
#
# 입력
#   obs_global_dim = N_atoms * obs_dim_atom  (flatten)
#   act_global_dim = N_atoms * 3
#
# 구조-level only (per-atom critic 없음)
###############################################################

import torch
import torch.nn as nn
import numpy as np


###############################################################
# Swish (MACS 안정 activation)
###############################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


###############################################################
# 공통 MLP (LN + Swish)
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
# Critic V(s)
###############################################################
class CriticV(nn.Module):
    """
    Structure-level V(s)
    ---------------------
    obs_global: (B, obs_global_dim) or (obs_global_dim,)
    output    : (B,1)
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
        if obs_global.dim() == 1:
            obs_global = obs_global.unsqueeze(0)
        return self.net(obs_global)  # (B,1)


###############################################################
# Critic Q(s,a)
###############################################################
class CriticQ(nn.Module):
    """
    Structure-level Q(s,a)
    ----------------------
    obs_global: (B, obs_global_dim)
    act_global: (B, act_global_dim)
    output    : (B,1)
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
        if obs_global.dim() == 1:
            obs_global = obs_global.unsqueeze(0)
        if act_global.dim() == 1:
            act_global = act_global.unsqueeze(0)

        x = torch.cat([obs_global, act_global], dim=-1)
        return self.net(x)   # (B,1)


###############################################################
# Twin Critic for SAC
###############################################################
class TwinCriticQ(nn.Module):
    def __init__(self,
                 obs_global_dim: int,
                 act_global_dim: int,
                 hidden=[512, 512, 256]):
        super().__init__()

        self.Q1 = CriticQ(obs_global_dim, act_global_dim, hidden)
        self.Q2 = CriticQ(obs_global_dim, act_global_dim, hidden)

    def forward(self, obs, act):
        return self.Q1(obs, act), self.Q2(obs, act)
