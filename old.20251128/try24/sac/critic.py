# sac/critic.py
###############################################################
# CentralCritic / TwinQCentral for CTDE-style Multi-Agent SAC
# -------------------------------------------------------------
# - Per-atom Q-network with optional:
#     * atom-type embedding (metal / linker / etc.)
#     * global embedding g (구조 전체 통계 등)
# - Twin Q (Q1, Q2) for SAC (TD target에서 min(Q1, Q2) 사용)
# - 입력은 "per-atom 튜플" 단위로 설계:
#     obs_i, act_i, (optional) atom_type_id_i, (optional) global_feat
# - 출력은 per-atom Q_i (B, 1). 필요하면 외부에서 평균/합산해서
#   structure-wise Q_total로 쓸 수 있음.
###############################################################

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .actor import Swish, build_mlp  # 같은 스타일 MLP/activation 재사용

logger = logging.getLogger("sac.critic")


###############################################################
# CentralQNet — per-atom Q(s_i, a_i, g)
###############################################################
class CentralQNet(nn.Module):
    """
    Per-atom Q-network for CTDE-lite SAC.

    Q_i = Q(obs_i, act_i, global_feat, atom_type_emb)

    Parameters
    ----------
    obs_dim : int
        Per-atom observation dimension.
    act_dim : int
        Per-atom action dimension.
    hidden : Sequence[int]
        Hidden layer sizes.
    n_atom_types : Optional[int]
        서로 다른 atom type 개수 (0 ~ n_atom_types-1). 지정하면 embedding 사용.
    type_embed_dim : int
        Atom-type embedding dimension.
    global_dim : int
        Global embedding dimension (예: mean|F|, Fmax, N_atoms, cell lengths 등).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (256, 256),
        n_atom_types: Optional[int] = None,
        type_embed_dim: int = 8,
        global_dim: int = 0,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden = tuple(hidden)
        self.n_atom_types = n_atom_types
        self.type_embed_dim = int(type_embed_dim) if n_atom_types is not None else 0
        self.global_dim = int(global_dim)

        # Optional atom-type embedding
        if self.n_atom_types is not None:
            self.type_embedding = nn.Embedding(self.n_atom_types, self.type_embed_dim)
            nn.init.uniform_(self.type_embedding.weight, -0.1, 0.1)
        else:
            self.type_embedding = None

        # 최종 입력 차원 = obs + act + (type_emb) + (global_feat)
        input_dim = self.obs_dim + self.act_dim + self.type_embed_dim + self.global_dim

        self.backbone = build_mlp(
            in_dim=input_dim,
            hidden_sizes=self.hidden,
            out_dim=1,  # scalar Q_i
            final_activation=None,
        )

        self._init_weights()

        logger.info(
            "[CentralQNet.__init__] obs_dim=%d, act_dim=%d, hidden=%s, "
            "n_atom_types=%s, type_embed_dim=%d, global_dim=%d",
            self.obs_dim,
            self.act_dim,
            str(self.hidden),
            str(self.n_atom_types),
            self.type_embed_dim,
            self.global_dim,
        )

    # ----------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ----------------------------------------------------------
    def _build_input_tensor(
        self,
        obs: torch.Tensor,
        acts: torch.Tensor,
        atom_type_id: Optional[torch.Tensor] = None,
        global_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        obs         : (B, obs_dim)
        acts        : (B, act_dim)
        atom_type_id: (B,) or None
        global_feat : (B, global_dim) or (global_dim,) or None
        """
        B = obs.size(0)
        x = [obs, acts]

        if self.type_embedding is not None:
            if atom_type_id is None:
                raise ValueError(
                    "CentralQNet created with n_atom_types, "
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
                    "CentralQNet created with global_dim>0, "
                    "but global_feat is None in forward()."
                )
            if global_feat.dim() == 1:
                global_feat = global_feat.view(1, -1).expand(B, -1)
            x.append(global_feat)

        return torch.cat(x, dim=-1)

    # ----------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
        acts: torch.Tensor,
        atom_type_id: Optional[torch.Tensor] = None,
        global_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        obs : (B, obs_dim) or (obs_dim,)
        acts : (B, act_dim) or (act_dim,)
        atom_type_id : (B,) or None
        global_feat : (B, global_dim) or (global_dim,) or None

        Returns
        -------
        q : (B, 1)
            Per-atom Q_i(s_i, a_i, g).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if acts.dim() == 1:
            acts = acts.unsqueeze(0)

        h_in = self._build_input_tensor(obs, acts, atom_type_id, global_feat)
        q = self.backbone(h_in)

        logger.debug(
            "[CentralQNet.forward] obs_shape=%s, acts_shape=%s, q_shape=%s",
            tuple(obs.shape),
            tuple(acts.shape),
            tuple(q.shape),
        )
        return q


###############################################################
# CentralVNet — optional V(s) for SAC (state-value)
###############################################################
class CentralVNet(nn.Module):
    """
    Optional V-network: per-atom V(s_i, g).

    보통 최신 SAC 구현은 V 없이 TwinQ + targetQ만 쓰지만,
    필요하면 이 네트워크로 V(s)를 따로 근사할 수 있음.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden: Sequence[int] = (256, 256),
        n_atom_types: Optional[int] = None,
        type_embed_dim: int = 8,
        global_dim: int = 0,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.hidden = tuple(hidden)
        self.n_atom_types = n_atom_types
        self.type_embed_dim = int(type_embed_dim) if n_atom_types is not None else 0
        self.global_dim = int(global_dim)

        if self.n_atom_types is not None:
            self.type_embedding = nn.Embedding(self.n_atom_types, self.type_embed_dim)
            nn.init.uniform_(self.type_embedding.weight, -0.1, 0.1)
        else:
            self.type_embedding = None

        input_dim = self.obs_dim + self.type_embed_dim + self.global_dim

        self.backbone = build_mlp(
            in_dim=input_dim,
            hidden_sizes=self.hidden,
            out_dim=1,
            final_activation=None,
        )

        self._init_weights()

        logger.info(
            "[CentralVNet.__init__] obs_dim=%d, hidden=%s, "
            "n_atom_types=%s, type_embed_dim=%d, global_dim=%d",
            self.obs_dim,
            str(self.hidden),
            str(self.n_atom_types),
            self.type_embed_dim,
            self.global_dim,
        )

    # ----------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ----------------------------------------------------------
    def _build_input_tensor(
        self,
        obs: torch.Tensor,
        atom_type_id: Optional[torch.Tensor] = None,
        global_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = obs.size(0)
        x = [obs]

        if self.type_embedding is not None:
            if atom_type_id is None:
                raise ValueError(
                    "CentralVNet created with n_atom_types, "
                    "but atom_type_id is None in forward()."
                )
            if atom_type_id.dim() == 1:
                atom_type_id = atom_type_id.view(B)
            else:
                atom_type_id = atom_type_id.view(B)
            type_emb = self.type_embedding(atom_type_id)
            x.append(type_emb)

        if self.global_dim > 0:
            if global_feat is None:
                raise ValueError(
                    "CentralVNet created with global_dim>0, "
                    "but global_feat is None in forward()."
                )
            if global_feat.dim() == 1:
                global_feat = global_feat.view(1, -1).expand(B, -1)
            x.append(global_feat)

        return torch.cat(x, dim=-1)

    # ----------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
        atom_type_id: Optional[torch.Tensor] = None,
        global_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        obs         : (B, obs_dim) or (obs_dim,)
        atom_type_id: (B,) or None
        global_feat : (B, global_dim) or (global_dim,) or None

        Returns
        -------
        v : (B, 1)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        h_in = self._build_input_tensor(obs, atom_type_id, global_feat)
        v = self.backbone(h_in)

        logger.debug(
            "[CentralVNet.forward] obs_shape=%s, v_shape=%s",
            tuple(obs.shape),
            tuple(v.shape),
        )
        return v


###############################################################
# TwinQCentral — Twin Critic wrapper (SAC용)
###############################################################
class TwinQCentral(nn.Module):
    """
    Twin Central Q-network for SAC.

    - 내부에 Q1, Q2 두 개의 CentralQNet 보유
    - forward()에서 둘 다 계산해서 반환
    - obs/acts/atom_type_id/global_feat는 per-atom 튜플 단위

    사용 예
    -------
    twin_q = TwinQCentral(obs_dim, act_dim, hidden=(256, 256),
                          n_atom_types=n_types, global_dim=g_dim)

    q1, q2 = twin_q(obs_batch, act_batch, atom_type_id_batch, global_batch)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (256, 256),
        n_atom_types: Optional[int] = None,
        type_embed_dim: int = 8,
        global_dim: int = 0,
    ):
        super().__init__()

        self.q1 = CentralQNet(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=hidden,
            n_atom_types=n_atom_types,
            type_embed_dim=type_embed_dim,
            global_dim=global_dim,
        )
        self.q2 = CentralQNet(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=hidden,
            n_atom_types=n_atom_types,
            type_embed_dim=type_embed_dim,
            global_dim=global_dim,
        )

        logger.info(
            "[TwinQCentral.__init__] Created TwinQ with obs_dim=%d, act_dim=%d, "
            "hidden=%s, n_atom_types=%s, type_embed_dim=%d, global_dim=%d",
            obs_dim,
            act_dim,
            str(hidden),
            str(n_atom_types),
            type_embed_dim,
            global_dim,
        )

    # ----------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
        acts: torch.Tensor,
        atom_type_id: Optional[torch.Tensor] = None,
        global_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs         : (B, obs_dim) or (obs_dim,)
        acts        : (B, act_dim) or (act_dim,)
        atom_type_id: (B,) or None
        global_feat : (B, global_dim) or (global_dim,) or None

        Returns
        -------
        q1 : (B, 1)
        q2 : (B, 1)
        """
        q1 = self.q1(obs, acts, atom_type_id, global_feat)
        q2 = self.q2(obs, acts, atom_type_id, global_feat)

        logger.debug(
            "[TwinQCentral.forward] obs_shape=%s, acts_shape=%s, "
            "q1_shape=%s, q2_shape=%s",
            tuple(obs.shape) if obs.dim() > 1 else (obs.numel(),),
            tuple(acts.shape) if acts.dim() > 1 else (acts.numel(),),
            tuple(q1.shape),
            tuple(q2.shape),
        )
        return q1, q2
