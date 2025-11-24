import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


######################################################################
# Smooth Activation (Swish)
######################################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


######################################################################
# Stable Gaussian Policy → Squashed to [0, 1] (Softplus-based)
######################################################################
class Actor(nn.Module):

    def __init__(
        self,
        obs_dim,
        act_dim=1,
        hidden_sizes=[256, 256, 128, 64],
        log_std_min=-4.0,
        log_std_max=1.0,
        action_max=0.12               # ★ max action scale
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_max = action_max

        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max

        # ------------------------------------------------------------
        # Deep MLP Backbone
        # ------------------------------------------------------------
        layers = []
        in_dim = obs_dim

        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                Swish()
            ]
            in_dim = h

        self.net = nn.Sequential(*layers)

        # Gaussian heads
        self.mu_head = nn.Linear(in_dim, act_dim)
        self.log_std_head = nn.Linear(in_dim, act_dim)

        # action smoothing memory
        self.prev_action = None


    ##################################################################
    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        h = self.net(obs)

        mu = self.mu_head(h)
        log_std = self.log_std_head(h)

        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Reparameterization
        eps = torch.randn_like(mu)
        raw = mu + eps * std

        # ★ key change: softplus → safe bounded action
        squashed = F.softplus(raw) / (1.0 + F.softplus(raw))
        scale = squashed * self.action_max

        # ------------------------------------------------------------
        # log-probability of unsquashed Gaussian
        # ------------------------------------------------------------
        gauss_logp = (
            -0.5 * ((raw - mu) / (std + 1e-8)) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1, keepdim=True)

        # J' correction for softplus squash
        jacobian = torch.log(
            (1.0 / (1.0 + F.softplus(raw))) *
            (F.softplus(raw) / (1.0 + F.softplus(raw))) + 1e-10
        ).sum(dim=-1, keepdim=True)

        logp = gauss_logp - jacobian

        return scale, logp, mu, std

    ##################################################################
    @torch.no_grad()
    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        scale, _, _, _ = self.forward(obs)

        scale = scale.squeeze().cpu().numpy()

        # ------------------------------------------------------------
        # ★ smoothing (critical for stable MOF optimization)
        # ------------------------------------------------------------
        if self.prev_action is None:
            self.prev_action = scale
        else:
            scale = 0.7 * self.prev_action + 0.3 * scale
            self.prev_action = scale

        return scale
