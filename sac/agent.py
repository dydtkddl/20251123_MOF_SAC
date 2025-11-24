###############################################################
# MultiAgentSAC — Parameter-Sharing Per-Atom SAC + CTDE Twin Q
#
# - Actor : AtomActor (공유 파라미터, per-atom Gaussian policy)
# - Critic: TwinQCentral (per-atom Q(s_i, a_i, g), CTDE-lite)
# - Replay: MultiAgentReplayBuffer (per-atom transition 튜플)
#
# 인터페이스 컨벤션:
# ------------------------------------------------
#   • obs_dim   : per-atom feature dim
#   • act_dim   : per-atom action dim (3: Δx, Δy, Δz)
#
#   • ReplayBuffer.sample(batch_size) → (batch, idxs, weights)
#       batch["obs"]        : (B, obs_dim)           # s_i
#       batch["acts"]       : (B, act_dim)           # a_i
#       batch["rews"]       : (B,)                   # scalar r
#       batch["next_obs"]   : (B, obs_dim)           # s'_i
#       batch["done"]       : (B,)                   # 0/1
#       (optional)
#       batch["atom_type"]  : (B,)                   # i의 atom_type_id
#       batch["next_atom_type"]: (B,)
#       batch["global"]     : (B, global_dim)
#       batch["next_global"]: (B, global_dim)
#
#   • PER:
#       idxs    : (B,) priority index (numpy)
#       weights : (B,) importance sampling weight (numpy)
#
#   • PER priority update:
#       replay_buffer.update_priority(idxs, td_errors)
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

    Parameter-sharing per-atom policy + CTDE-lite Twin Q.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        replay_buffer,
        n_atom_types: Optional[int] = None,
        global_dim: int = 0,
        actor_hidden: Tuple[int, int] = (256, 256),
        critic_hidden: Tuple[int, int] = (256, 256),
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
            np.log(alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
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
        obs_atom,
        atom_type_id: Optional[torch.Tensor] = None,
        global_feat: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Policy로부터 per-atom action을 샘플.

        Parameters
        ----------
        obs_atom : np.ndarray or torch.Tensor, shape (N, obs_dim) or (obs_dim,)
        atom_type_id : (N,) or None
        global_feat : (global_dim,) or (N, global_dim) or None
        deterministic : bool
            True면 mean action (no noise),
            False면 reparameterized sample.

        Returns
        -------
        actions : (N, act_dim)
        """
        self.actor.eval()

        # obs
        if isinstance(obs_atom, np.ndarray):
            obs_atom = torch.as_tensor(
                obs_atom, dtype=torch.float32, device=self.device
            )
        else:
            obs_atom = obs_atom.to(self.device)

        # atom_type
        if atom_type_id is not None:
            if isinstance(atom_type_id, np.ndarray):
                atom_type_id = torch.as_tensor(
                    atom_type_id, dtype=torch.long, device=self.device
                )
            else:
                atom_type_id = atom_type_id.to(self.device)

        # global_feat
        if global_feat is not None:
            if isinstance(global_feat, np.ndarray):
                global_feat = torch.as_tensor(
                    global_feat, dtype=torch.float32, device=self.device
                )
            else:
                global_feat = global_feat.to(self.device)

        if deterministic:
            actions, _, _, _ = self.actor(
                obs_atom,
                atom_type_id=atom_type_id,
                global_feat=global_feat,
                deterministic=True,
            )
        else:
            actions, _, _, _ = self.actor(
                obs_atom,
                atom_type_id=atom_type_id,
                global_feat=global_feat,
                deterministic=False,
            )

        self.actor.train()
        return actions

    # ==========================================================
    # Sample from replay buffer
    # ==========================================================
    def _sample_batch(self, batch_size: int):
        """
        Replay buffer에서 batch를 샘플하고 torch.Tensor로 변환.

        Returns
        -------
        batch_torch : Dict[str, torch.Tensor]
        idxs        : np.ndarray or None  (PER priority update용)
        weights_t   : torch.Tensor or None
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
            weights_t = to_tensor(weights, dtype=torch.float32).view(-1, 1)
        else:
            weights_t = None

        # per_use_weights=False면 importance weight를 1로 처리
        if not self.per_use_weights or weights_t is None:
            weights_t = torch.ones_like(rews, device=self.device)

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
            "weights": weights_t,
        }

        # idxs는 PER 업데이트용으로 numpy 그대로 유지 (GPU 텐서로 바꾸지 않음!)
        return batch_torch, idxs, weights_t

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

            alpha_tensor = self.log_alpha.exp()
            target_q = rews + self.gamma * (1.0 - done) * (
                q_next - alpha_tensor * next_logp
            )

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

        # PER priority update (alias: update_priority → update_priorities)
        if idxs is not None and hasattr(self.replay_buffer, "update_priority"):
            with torch.no_grad():
                # numpy (CPU) 배열로 변환
                new_priorities = (
                    td_error.abs().squeeze(-1).detach().cpu().numpy()
                )
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

        # alpha는 critic에서 썼던 log_alpha와 동일한 값 사용
        alpha_tensor = self.log_alpha.exp()
        actor_loss = (weights * (alpha_tensor * logp_pi - q_pi)).mean()

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
            alpha_loss = -(
                self.log_alpha * (logp_pi + self.target_entropy).detach()
            ).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = alpha_loss.item()

        alpha_value = self.alpha  # scalar

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
            # 저장된 값이 이미 log(alpha)인지, alpha인지에 따라 다를 수 있어
            # 여기서는 alpha 값이라고 가정하고 log 변환
            loaded_alpha = state["log_alpha"]
            if isinstance(loaded_alpha, torch.Tensor):
                loaded_alpha = loaded_alpha.to(self.device)
            self.log_alpha.data = loaded_alpha.log().clamp_(min=-20.0, max=2.0)

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
