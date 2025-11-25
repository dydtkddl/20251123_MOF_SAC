# sac/actor.py
###############################################################
# AtomActor — Per-Atom Gaussian Policy (Shared Parameters)
# -------------------------------------------------------------
# - Multi-Agent 설정에서 "원자 = 에이전트" 일 때 쓰는 정책 네트워크
# - 모든 원자가 동일한 Actor 파라미터를 공유 (Parameter Sharing)
# - 입력:
#     obs_i        : (B, obs_dim)      per-atom feature
#     atom_type_id : (B,)              optional, int64
#     global_feat  : (B, global_dim)   optional, 구조 전체 embedding
# - 출력:
#     action       : (B, act_dim)      tanh-squashed action (예: 3D disp)
#     logp         : (B, 1)            log π(a|s) (SAC용)
#     mu, std      : (B, act_dim)      Gaussian 파라미터
###############################################################

import logging
from typing import Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("sac.actor")


###############################################################
# Swish / SiLU activation
###############################################################
class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


###############################################################
# Small helper: MLP with LayerNorm + Swish
###############################################################
def build_mlp(
    in_dim: int,
    hidden_sizes: Sequence[int],
    out_dim: int,
    final_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(d, h))
        layers.append(nn.LayerNorm(h))
        layers.append(Swish())
        d = h

    layers.append(nn.Linear(d, out_dim))
    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)


###############################################################
# AtomActor
###############################################################
class AtomActor(nn.Module):
    """
    Per-atom Gaussian policy with tanh squash (SAC-style).

    Parameters
    ----------
    obs_dim : int
        Base per-atom observation dimension (e.g. MOFEnv에서 나온 obs_atom[i].shape[-1]).
    act_dim : int
        Per-atom action dimension (기본: 3, Δx, Δy, Δz).
    hidden : Sequence[int]
        MLP hidden layer sizes.
    n_atom_types : Optional[int]
        서로 다른 atom type 개수 (e.g. 0~K-1). 주어지면 nn.Embedding을 사용.
    type_embed_dim : int
        atom_type embedding dimension.
    global_dim : int
        구조 전체 embedding (global_feat) dimension. (옵션)
    log_std_min, log_std_max : float
        log σ clamp 범위.
    action_max : float
        tanh 후 scaling (기본 1.0 → env에서 바로 [-1,1]로 사용).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 3,
        hidden: Sequence[int] = (256, 256),
        n_atom_types: Optional[int] = None,
        type_embed_dim: int = 8,
        global_dim: int = 0,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
        action_max: float = 1.0,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden = tuple(hidden)
        self.n_atom_types = n_atom_types
        self.type_embed_dim = int(type_embed_dim) if n_atom_types is not None else 0
        self.global_dim = int(global_dim)

        self.LOG_STD_MIN = float(log_std_min)
        self.LOG_STD_MAX = float(log_std_max)
        self.action_max = float(action_max)

        # Optional atom-type embedding
        if self.n_atom_types is not None:
            self.type_embedding = nn.Embedding(self.n_atom_types, self.type_embed_dim)
            nn.init.uniform_(self.type_embedding.weight, -0.1, 0.1)
        else:
            self.type_embedding = None

        # 최종 입력 차원 = obs + (type_embed) + (global_feat)
        input_dim = self.obs_dim + self.type_embed_dim + self.global_dim

        # Backbone MLP
        self.backbone = build_mlp(
            in_dim=input_dim,
            hidden_sizes=self.hidden,
            out_dim=self.hidden[-1],
        )

        # Gaussian head
        self.mu_head = nn.Linear(self.hidden[-1], self.act_dim)
        self.log_std_head = nn.Linear(self.hidden[-1], self.act_dim)

        self._init_weights()

        logger.info(
            "[AtomActor.__init__] obs_dim=%d, act_dim=%d, hidden=%s, "
            "n_atom_types=%s, type_embed_dim=%d, global_dim=%d, "
            "log_std=[%.1f, %.1f], action_max=%.3f",
            self.obs_dim,
            self.act_dim,
            str(self.hidden),
            str(self.n_atom_types),
            self.type_embed_dim,
            self.global_dim,
            self.LOG_STD_MIN,
            self.LOG_STD_MAX,
            self.action_max,
        )

    # ----------------------------------------------------------
    # Weight init (optional Xavier)
    # ----------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ----------------------------------------------------------
    # Input builder: obs + (type_emb) + (global_feat)
    # ----------------------------------------------------------
    def _build_input_tensor(
        self,
        obs: torch.Tensor,
        atom_type_id: Optional[torch.Tensor] = None,
        global_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        obs        : (B, obs_dim)
        atom_type_id : (B,) or None
        global_feat: (B, global_dim) or None
        """
        B = obs.size(0)
        x = [obs]

        if self.type_embedding is not None:
            if atom_type_id is None:
                raise ValueError(
                    "AtomActor was created with n_atom_types, "
                    "but atom_type_id is None in forward()."
                )
            if atom_type_id.dim() == 1:
                atom_type_id = atom_type_id.view(B)
            else:
                atom_type_id = atom_type_id.view(B)

            type_emb = self.type_embedding(atom_type_id)  # (B, type_embed_dim)
            x.append(type_emb)

        if self.global_dim > 0:
            if global_feat is None:
                raise ValueError(
                    "AtomActor was created with global_dim>0, "
                    "but global_feat is None in forward()."
                )
            if global_feat.dim() == 1:
                global_feat = global_feat.view(1, -1).expand(B, -1)
            x.append(global_feat)

        return torch.cat(x, dim=-1)

    # ----------------------------------------------------------
    # Forward (SAC training)
    # ----------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
        atom_type_id: Optional[torch.Tensor] = None,
        global_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs        : (B, obs_dim)
        atom_type_id : (B,) int64 or None
        global_feat: (B, global_dim) or (global_dim,) or None

        Returns
        -------
        action : (B, act_dim)         tanh-squashed action
        logp   : (B, 1)               log π(a|s)
        mu     : (B, act_dim)
        std    : (B, act_dim)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        h_in = self._build_input_tensor(obs, atom_type_id, global_feat)
        h = self.backbone(h_in)

        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Reparameterization trick: z = mu + std * eps
        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps
        tanh_out = torch.tanh(pre_tanh)

        action = self.action_max * tanh_out  # (B, act_dim)

        # Gaussian log prob
        # log N(pre_tanh | mu, std^2)
        # = -0.5 * [ ((pre_tanh - mu)/std)^2 + 2 log_std + log(2π) ]
        gauss_logp = (
            -0.5 * (((pre_tanh - mu) / (std + 1e-8)) ** 2)
            - log_std
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1, keepdim=True)

        # Tanh correction: sum log(1 - tanh(z)^2)
        log_det = torch.log(1.0 - tanh_out.pow(2) + 1e-10).sum(dim=-1, keepdim=True)

        logp = gauss_logp - log_det

        return action, logp, mu, std

    # ----------------------------------------------------------
    # Deterministic action (mean) for rollout (torch input)
    # ----------------------------------------------------------
    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        atom_type_id: Optional[torch.Tensor] = None,
        global_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Deterministic action (policy mean) for torch input.

        Parameters
        ----------
        obs : (B, obs_dim) or (obs_dim,)
        atom_type_id : (B,) or None
        global_feat : (B, global_dim) or None

        Returns
        -------
        action : (B, act_dim)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        h_in = self._build_input_tensor(obs, atom_type_id, global_feat)
        h = self.backbone(h_in)

        mu = self.mu_head(h)
        tanh_out = torch.tanh(mu)
        action = self.action_max * tanh_out
        return action

    # ----------------------------------------------------------
    # Deterministic action (numpy) for env rollout
    # ----------------------------------------------------------
    @torch.no_grad()
    def act_numpy(
        self,
        obs_np: np.ndarray,
        atom_type_id_np: Optional[np.ndarray] = None,
        global_feat_np: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Numpy 입력을 받아 deterministic action을 numpy로 반환.

        Parameters
        ----------
        obs_np : (N, obs_dim) or (obs_dim,)
        atom_type_id_np : (N,) or None
        global_feat_np : (N, global_dim) or (global_dim,) or None
        device : torch.device or None

        Returns
        -------
        action_np : (N, act_dim)
        """
        if device is None:
            device = torch.device("cpu")

        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        atom_type_t = None
        if atom_type_id_np is not None and self.type_embedding is not None:
            atom_type_t = torch.as_tensor(
                atom_type_id_np, dtype=torch.long, device=device
            )
            if atom_type_t.dim() == 0:
                atom_type_t = atom_type_t.view(1)

        global_t = None
        if global_feat_np is not None and self.global_dim > 0:
            global_t = torch.as_tensor(
                global_feat_np, dtype=torch.float32, device=device
            )
            if global_t.dim() == 1:
                global_t = global_t.unsqueeze(0)

        action_t = self.act(obs_t, atom_type_t, global_t)
        action_np = action_t.cpu().numpy().astype(np.float32)

        logger.debug(
            "[AtomActor.act_numpy] obs_shape=%s, action_shape=%s",
            tuple(obs_np.shape),
            tuple(action_np.shape),
        )
        return action_np
