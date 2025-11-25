import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        in_dim = obs_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.mean_linear = nn.Linear(hidden_dim, act_dim)
        self.log_std_linear = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = self.backbone(obs)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self, obs: torch.Tensor, deterministic: bool = False
    ):
        mean, log_std = self.forward(obs)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
            return action, log_prob, action

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        # Tanh-squashed log_prob
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action
