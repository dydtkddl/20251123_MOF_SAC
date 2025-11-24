# sac/agent.py
###############################################################
# MultiAgentSAC — Parameter-Sharing Per-Atom SAC + CTDE Twin Q
#
# - Actor : AtomActor (공유 파라미터, per-atom Gaussian policy)
# - Critic: TwinQCentral (per-atom Q(s_i, a_i, g), CTDE-lite)
# - Replay: MultiAgentReplayBuffer (per-atom transition 튜플)
#
# 핵심 컨벤션 (env / replay_buffer와의 인터페이스):
# ------------------------------------------------
#   • obs_dim   : per-atom feature dim (MOFEnv._build_f 기반)
#   • act_dim   : per-atom action dim (3: Δx, Δy, Δz)
#
#   • ReplayBuffer.sample(batch_size) → (batch, idxs, weights)
#       batch["obs"]       : (B, obs_dim)          # s_i
#       batch["acts"]      : (B, act_dim)          # a_i
#       batch["rews"]      : (B, 1) or (B,)        # scalar r
#       batch["next_obs"]  : (B, obs_dim)          # s'_i
#       batch["done"]      : (B, 1) or (B,)        # 0/1
#       (optional)
#       batch["atom_type"] : (B,)                  # i의 atom_type_id
#       batch["global"]    : (B, global_dim)       # g_t
#       batch["next_global"]: (B, global_dim)      # g_{t+1}
#
#   • PER인 경우:
#       - idxs    : (B,) priority index
#       - weights : (B,) importance sampling weight
#
# MultiAgentSAC.update()는 위 batch를 사용해서:
#   - Q-loss, policy-loss, alpha-loss 계산
#   - replay_buffer.update_priority(idxs, td_errors) 호출 (선택)
#
###############################################################

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .actor import AtomActor
from .critic import TwinQCentral, CentralVNet

logger = logging.getLogger("sac.agent")


class MultiAgentSAC:
    """
    Multi-Agent SAC for per-atom MOF 구조 최적화.

    Parameter-sharing per-atom policy, CTDE-lite TwinQ.

    Parameters
    ----------
    obs_dim : int
        Per-atom observation dimension.
    act_dim : int
        Per-atom action dimension (default: 3).
    replay_buffer : object
        MultiAgentReplayBuffer-like 객체.
        sample(batch_size) → (batch, idxs, weights) 인터페이스 필요.
    n_atom_types : Optional[int]
        atom_type_id ∈ [0, n_atom_types-1], 없으면 None.
        지정하면 Actor / Critic에서 embedding 사용.
    global_dim : int
        global embedding g의 차원 (0이면 사용 안 함).
    actor_hidden : tuple[int, ...]
        Actor MLP hidden layer sizes.
    critic_hidden : tuple[int, ...]
        Critic MLP hidden layer sizes.
    gamma : float
        Discount factor.
    tau : float
        Soft-update 계수 (target networks).
    alpha : float
        초기 entropy temperature. auto_alpha=True일 때 learnable 초기값.
    target_entropy_scale : float
        target_entropy = -act_dim * target_entropy_scale
        (보통 1.0 사용, 값 높이면 exploration 강화).
    lr_actor : float
        Actor learning rate.
    lr_critic : float
        Critic learning rate.
    lr_alpha : float
        Alpha optimizer learning rate (auto_alpha=True일 때만 사용).
    device : str
        "cuda" or "cpu".
    auto_alpha : bool
        True면 entropy temperature α를 자동 튜닝.
    use_v_net : bool
        별도의 V-net을 쓰고 싶으면 True (기본 False).
        최신 SAC 스타일은 TwinQ + targetQ만 써도 충분.
    max_grad_norm : Optional[float]
        Gradient clipping max norm (None이면 미사용).
    per_use_weights : bool
        ReplayBuffer에서 제공하는 importance weights를 loss에 적용할지.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        replay_buffer,
        n_atom_types: Optional[int] = None,
        global_dim: int = 0,
        actor_hidden=(256, 256),
        critic_hidden=(256, 256),
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        target_entropy_scale: float = 1.0,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        device: Optional[str] = None,
        auto_alpha: bool = True,
        use_v_net: bool = False,
        max_grad_norm: Optional[float] = None,
        per_use_weights: bool = True,
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.replay_buffer = replay_buffer

        self.n_atom_types = n_atom_types
        self.global_dim = int(global_dim)

        self.actor_hidden = tuple(actor_hidden)
        self.critic_hidden = tuple(critic_hidden)
        self.gamma = float(gamma)
        self.tau = float(tau)

        self.auto_alpha = bool(auto_alpha)
        self.use_v_net = bool(use_v_net)
        self.max_grad_norm = max_grad_norm
        self.per_use_weights = per_use_weights

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ------------------------------------------------------
        # Networks
        # ------------------------------------------------------
        # Actor: 파라미터 공유 per-atom policy
        self.actor = AtomActor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden=self.actor_hidden,
            n_atom_types=self.n_atom_types,
            type_embed_dim=8,
            global_dim=self.global_dim,
        ).to(self.device)

        # Twin Q Critic (CTDE-lite)
        self.critic = TwinQCentral(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden=self.critic_hidden,
            n_atom_types=self.n_atom_types,
            type_embed_dim=8,
            global_dim=self.global_dim,
        ).to(self.device)

        self.critic_target = TwinQCentral(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden=self.critic_hidden,
            n_atom_types=self.n_atom_types,
            type_embed_dim=8,
            global_dim=self.global_dim,
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optional V network
        if self.use_v_net:
            self.v_net = CentralVNet(
                obs_dim=self.obs_dim,
                hidden=self.critic_hidden,
                n_atom_types=self.n_atom_types,
                type_embed_dim=8,
                global_dim=self.global_dim,
            ).to(self.device)
            self.v_target = CentralVNet(
                obs_dim=self.obs_dim,
                hidden=self.critic_hidden,
                n_atom_types=self.n_atom_types,
                type_embed_dim=8,
                global_dim=self.global_dim,
            ).to(self.device)
            self.v_target.load_state_dict(self.v_net.state_dict())
        else:
            self.v_net = None
            self.v_target = None

        # ------------------------------------------------------
        # Entropy temperature α (auto-tuning)
        # ------------------------------------------------------
        self.log_alpha = torch.tensor(
            np.log(alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )
        # Target entropy (per-atom action)
        self.target_entropy = -float(self.act_dim) * float(target_entropy_scale)

        # ------------------------------------------------------
        # Optimizers
        # ------------------------------------------------------
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        if self.use_v_net:
            self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=lr_critic)
        else:
            self.v_optimizer = None

        if self.auto_alpha:
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.alpha_optimizer = None

        # ------------------------------------------------------
        # Bookkeeping
        # ------------------------------------------------------
        self.train_step = 0  # 총 update step 카운터
        logger.info(
            "[MultiAgentSAC.__init__] obs_dim=%d, act_dim=%d, n_atom_types=%s, "
            "global_dim=%d, gamma=%.4f, tau=%.4f, auto_alpha=%s, use_v_net=%s, "
            "device=%s",
            self.obs_dim,
            self.act_dim,
            str(self.n_atom_types),
            self.global_dim,
            self.gamma,
            self.tau,
            str(self.auto_alpha),
            str(self.use_v_net),
            str(self.device),
        )

    # ==========================================================
    # Utilities
    # ==========================================================
    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    def _soft_update(self, target: nn.Module, source: nn.Module):
        with torch.no_grad():
            for p_t, p in zip(target.parameters(), source.parameters()):
                p_t.data.mul_(1.0 - self.tau)
                p_t.data.add_(self.tau * p.data)

    # ==========================================================
    # Action selection (per-atom)
    # ==========================================================
    @torch.no_grad()
    def act(
        self,
        obs_atom: torch.Tensor,
        atom_type_id: Optional[torch.Tensor] = None,
        global_feat: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Policy로부터 per-atom action을 샘플.

        Parameters
        ----------
        obs_atom : (N, obs_dim) or (obs_dim,)
        atom_type_id : (N,) or None
        global_feat : (global_dim,) or (N, global_dim) or None
        deterministic : bool
            True면 mean action (no noise), False면 reparameterized sample.

        Returns
        -------
        actions : (N, act_dim)
        """
        self.actor.eval()

        if isinstance(obs_atom, np.ndarray):
            obs_atom = torch.as_tensor(obs_atom, dtype=torch.float32, device=self.device)
        else:
            obs_atom = obs_atom.to(self.device)

        if atom_type_id is not None:
            if isinstance(atom_type_id, np.ndarray):
                atom_type_id = torch.as_tensor(
                    atom_type_id, dtype=torch.long, device=self.device
                )
            else:
                atom_type_id = atom_type_id.to(self.device)

        if global_feat is not None:
            if isinstance(global_feat, np.ndarray):
                global_feat = torch.as_tensor(
                    global_feat, dtype=torch.float32, device=self.device
                )
            else:
                global_feat = global_feat.to(self.device)

        if determinant := deterministic:
            actions, _, _, _ = self.actor(
                obs_atom, atom_type_id=atom_type_id, global_feat=global_feat, deterministic=True
            )
        else:
            actions, _, _, _ = self.actor(
                obs_atom, atom_type_id=atom_type_id, global_feat=global_feat, deterministic=False
            )

        self.actor.train()
        return actions

    # ==========================================================
    # Sample from replay buffer
    # ==========================================================
    def _sample_batch(self, batch_size: int):
        """
        Replay buffer에서 batch를 샘플하고 device로 이동.

        Returns
        -------
        batch : Dict[str, torch.Tensor]
        idxs  : Optional[torch.Tensor]  (PER priority update용)
        weights : Optional[torch.Tensor]
        """
        batch, idxs, weights = self.replay_buffer.sample(batch_size)

        def to_tensor(x, dtype=torch.float32):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            if isinstance(x, np.ndarray):
                return torch.as_tensor(x, dtype=dtype, device=self.device)
            return torch.as_tensor(x, dtype=dtype, device=self.device)

        obs = to_tensor(batch["obs"], dtype=torch.float32)
        acts = to_tensor(batch["acts"], dtype=torch.float32)
        rews = to_tensor(batch["rews"], dtype=torch.float32).view(-1, 1)
        next_obs = to_tensor(batch["next_obs"], dtype=torch.float32)
        done = to_tensor(batch["done"], dtype=torch.float32).view(-1, 1)

        atom_type = to_tensor(batch.get("atom_type", None), dtype=torch.long)
        next_atom_type = to_tensor(batch.get("next_atom_type", None), dtype=torch.long)
        global_feat = to_tensor(batch.get("global", None), dtype=torch.float32)
        next_global = to_tensor(batch.get("next_global", None), dtype=torch.float32)

        if weights is not None:
            weights = to_tensor(weights, dtype=torch.float32).view(-1, 1)
        if idxs is not None:
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.to(self.device)
            else:
                idxs = torch.as_tensor(idxs, dtype=torch.long, device=self.device)

        batch_torch = {
            "obs": obs,
            "acts": acts,
            "rews": rews,
            "next_obs": next_obs,
            "done": done,
            "atom_type": atom_type,
            "next_atom_type": next_atom_type,
            "global": global_feat,
            "next_global": next_global,
        }

        return batch_torch, idxs, weights

    # ==========================================================
    # Single SAC update step
    # ==========================================================
    def update(self, batch_size: int) -> Dict[str, float]:
        """
        SAC update (1 step).

        Parameters
        ----------
        batch_size : int
            샘플링할 per-atom transition 수.

        Returns
        -------
        metrics : dict
            loss / alpha / Q값 등 모니터링용.
        """
        if len(self.replay_buffer) < batch_size:
            logger.debug(
                "[MultiAgentSAC.update] Buffer size (%d) < batch_size (%d), skip.",
                len(self.replay_buffer),
                batch_size,
            )
            return {}

        batch, idxs, weights = self._sample_batch(batch_size)

        obs = batch["obs"]
        acts = batch["acts"]
        rews = batch["rews"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        atom_type = batch["atom_type"]
        next_atom_type = batch["next_atom_type"]
        global_feat = batch["global"]
        next_global = batch["next_global"]

        if weights is None:
            weights = torch.ones_like(rews, device=self.device)

        # ------------------------------------------------------
        # 1) Critic update
        # ------------------------------------------------------
        self.critic_optimizer.zero_grad()

        with torch.no_grad():
            # Next actions and log probs
            next_pi, next_logp, _, _ = self.actor(
                next_obs,
                atom_type_id=next_atom_type,
                global_feat=next_global,
                deterministic=False,
            )

            # Q target from target critic
            q1_next, q2_next = self.critic_target(
                next_obs,
                next_pi,
                atom_type_id=next_atom_type,
                global_feat=next_global,
            )
            q_next = torch.min(q1_next, q2_next)

            alpha_value = self.log_alpha.exp()
            target_q = rews + self.gamma * (1.0 - done) * (q_next - alpha_value * next_logp)

        # Current Q estimates
        q1, q2 = self.critic(
            obs,
            acts,
            atom_type_id=atom_type,
            global_feat=global_feat,
        )

        # TD error
        td_error1 = q1 - target_q
        td_error2 = q2 - target_q
        td_error = 0.5 * (td_error1 + td_error2)

        critic_loss = (weights * (td_error1.pow(2) + td_error2.pow(2))).mean()

        critic_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # PER priority update
        if idxs is not None and hasattr(self.replay_buffer, "update_priority"):
            with torch.no_grad():
                new_priorities = td_error.abs().squeeze(-1).detach().cpu().numpy()
            self.replay_buffer.update_priority(idxs, new_priorities)

        # ------------------------------------------------------
        # 2) Actor update
        # ------------------------------------------------------
        self.actor_optimizer.zero_grad()

        pi, logp_pi, _, _ = self.actor(
            obs,
            atom_type_id=atom_type,
            global_feat=global_feat,
            deterministic=False,
        )
        q1_pi, q2_pi = self.critic(
            obs,
            pi,
            atom_type_id=atom_type,
            global_feat=global_feat,
        )
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (weights * (alpha_value * logp_pi - q_pi)).mean()

        actor_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # ------------------------------------------------------
        # 3) Alpha (entropy temperature) update
        # ------------------------------------------------------
        alpha_loss_value = 0.0
        if self.auto_alpha and self.alpha_optimizer is not None:
            self.alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = alpha_loss.item()
            alpha_value = self.log_alpha.exp().item()
        else:
            alpha_value = self.alpha

        # ------------------------------------------------------
        # 4) Optional V-net update (사용 시)
        # ------------------------------------------------------
        v_loss_value = 0.0
        if self.use_v_net and self.v_net is not None and self.v_optimizer is not None:
            self.v_optimizer.zero_grad()

            v_pred = self.v_net(
                obs,
                atom_type_id=atom_type,
                global_feat=global_feat,
            )
            # Soft value target: Q - alpha * logπ
            v_target = (q_pi - alpha_value * logp_pi).detach()
            v_loss = (weights * (v_pred - v_target).pow(2)).mean()

            v_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), self.max_grad_norm)
            self.v_optimizer.step()

            v_loss_value = v_loss.item()

        # ------------------------------------------------------
        # 5) Target network soft-update
        # ------------------------------------------------------
        self._soft_update(self.critic_target, self.critic)
        if self.use_v_net and self.v_target is not None:
            self._soft_update(self.v_target, self.v_net)

        self.train_step += 1

        # ------------------------------------------------------
        # Metrics for logging
        # ------------------------------------------------------
        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(alpha_value),
            "alpha_loss": float(alpha_loss_value),
            "q1_mean": float(q1.detach().mean().item()),
            "q2_mean": float(q2.detach().mean().item()),
            "q_pi_mean": float(q_pi.detach().mean().item()),
            "logp_pi_mean": float(logp_pi.detach().mean().item()),
            "v_loss": float(v_loss_value),
        }

        logger.debug(
            "[MultiAgentSAC.update] step=%d | "
            "critic_loss=%.4e, actor_loss=%.4e, alpha=%.4e, alpha_loss=%.4e, "
            "q1_mean=%.4e, q2_mean=%.4e, q_pi_mean=%.4e, logp_pi_mean=%.4e, v_loss=%.4e",
            self.train_step,
            metrics["critic_loss"],
            metrics["actor_loss"],
            metrics["alpha"],
            metrics["alpha_loss"],
            metrics["q1_mean"],
            metrics["q2_mean"],
            metrics["q_pi_mean"],
            metrics["logp_pi_mean"],
            metrics["v_loss"],
        )

        return metrics

    # ==========================================================
    # Save / Load
    # ==========================================================
    def save(self, path: str):
        """
        전체 SAC 상태 저장 (모델 + 옵티마이저 + alpha 등).
        """
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "train_step": self.train_step,
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
        }
        if self.use_v_net and self.v_net is not None:
            state["v_net"] = self.v_net.state_dict()
            state["v_target"] = self.v_target.state_dict()
            state["v_opt"] = self.v_optimizer.state_dict()
        if self.auto_alpha and self.alpha_optimizer is not None:
            state["alpha_opt"] = self.alpha_optimizer.state_dict()

        torch.save(state, path)
        logger.info("[MultiAgentSAC.save] Saved checkpoint to %s", path)

    def load(self, path: str, strict: bool = True):
        """
        SAC 상태 로드.

        strict=False 로 두면 일부 키 누락 시에도 최대한 로드.
        """
        state = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(state["actor"], strict=strict)
        self.critic.load_state_dict(state["critic"], strict=strict)
        self.critic_target.load_state_dict(state["critic_target"], strict=strict)

        if "log_alpha" in state:
            self.log_alpha.data = state["log_alpha"].to(self.device).log().clamp_(
                min=-20.0, max=2.0
            )

        self.train_step = int(state.get("train_step", 0))

        if "actor_opt" in state:
            self.actor_optimizer.load_state_dict(state["actor_opt"])
        if "critic_opt" in state:
            self.critic_optimizer.load_state_dict(state["critic_opt"])

        if self.use_v_net and self.v_net is not None:
            if "v_net" in state:
                self.v_net.load_state_dict(state["v_net"], strict=strict)
            if "v_target" in state:
                self.v_target.load_state_dict(state["v_target"], strict=strict)
            if "v_opt" in state and self.v_optimizer is not None:
                self.v_optimizer.load_state_dict(state["v_opt"])

        if self.auto_alpha and self.alpha_optimizer is not None:
            if "alpha_opt" in state:
                self.alpha_optimizer.load_state_dict(state["alpha_opt"])

        logger.info("[MultiAgentSAC.load] Loaded checkpoint from %s", path)
