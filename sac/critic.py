import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ):
        super().__init__()
        in_dim = obs_dim + act_dim
        layers = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if act.dim() == 1:
            act = act.unsqueeze(0)
        x = torch.cat([obs, act], dim=-1)
        x = self.backbone(x)
        q = self.q_head(x)
        return q


class TwinQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ):
        super().__init__()
        self.q1 = QNetwork(obs_dim, act_dim, hidden_dim, hidden_layers)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_dim, hidden_layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        return q1, q2
