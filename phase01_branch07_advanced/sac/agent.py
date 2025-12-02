# sac/agent.py

import numpy as np
import logging

import torch
import torch.nn.functional as F
import torch.optim as optim

from .actor import Actor
from .critic import CriticQ, CriticV

logger = logging.getLogger(__name__)


class SACAgent:
    """
    Stable per-atom SAC agent for MACS-MOF RL (4D action 지원)

    - obs_dim: per-atom observation dimension
    - act_dim: action dimension (default 4: gate + dx,dy,dz)
    - replay_buffer: per-atom ReplayBuffer
    """

    def __init__(
        self,
        obs_dim: int,
        replay_buffer,
        act_dim: int = 4,
        device: str = "cuda",
        gamma: float = 0.995,
        tau: float = 5e-3,
        batch_size: int = 256,
        lr: float = 3e-4,
                target_entropy: float = -1.0,   # ★ 추가
    ):
        self.replay = replay_buffer
        self.batch_size = batch_size

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau

        # ---------------------------
        # NETWORKS (FP32)
        # ---------------------------
        self.actor = Actor(obs_dim, act_dim).to(self.device).float()
        self.v = CriticV(obs_dim).to(self.device).float()
        self.v_tgt = CriticV(obs_dim).to(self.device).float()
        self.q1 = CriticQ(obs_dim, act_dim).to(self.device).float()
        self.q2 = CriticQ(obs_dim, act_dim).to(self.device).float()

        self.v_tgt.load_state_dict(self.v.state_dict())

        # ---------------------------
        # OPTIMIZERS
        # ---------------------------
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.v_opt = optim.Adam(self.v.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)

        # ---------------------------
        # ENTROPY (TARGET)
        # ---------------------------

        self.total_steps = 0
                # ---------------------------
        # ENTROPY (TARGET)
        # ---------------------------
        self.target_entropy = float(target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

        logger.info(
            f"[SACAgent] Initialized: obs_dim={obs_dim}, act_dim={act_dim}, "
            f"batch_size={batch_size}, gamma={gamma}, tau={tau}, lr={lr}, "
            f"device={self.device}"
        )

    # -------------------------------------------------------------
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # -------------------------------------------------------------
    # ACTION SELECTION
    # -------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: (obs_dim,) numpy array (per-atom)
        return: action (act_dim,) numpy array
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a, _, _, _ = self.actor(obs_t.unsqueeze(0))  # (1, act_dim)
        return a.squeeze(0).cpu().numpy()

    # -------------------------------------------------------------
    # UPDATE SAC
    # -------------------------------------------------------------
    def update(self):
        """
        One SAC update step using a mini-batch sampled from replay buffer.

        Returns
        -------
        dict
            policy_loss, q1_loss, q2_loss, v_loss, alpha_loss
        """

        if len(self.replay) < self.batch_size:
            logger.debug(
                f"[SACAgent.update] replay size {len(self.replay)} < batch_size {self.batch_size}, skip update."
            )
            return {
                "policy_loss": None,
                "q1_loss": None,
                "q2_loss": None,
                "v_loss": None,
                "alpha_loss": None,
            }

        policy_loss = None

        batch = self.replay.sample(self.batch_size)

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device).unsqueeze(1)
        nobs = torch.as_tensor(batch["nobs"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).unsqueeze(1)

        # ===========================
        # α update
        # ===========================
        new_action, logp, _, _ = self.actor(obs)
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ===========================
        # Q update
        # ===========================
        with torch.no_grad():
            v_next = self.v_tgt(nobs)
            q_target = rew + (1.0 - done) * self.gamma * v_next

        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)

        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # ===========================
        # V update
        # ===========================
        v_pred = self.v(obs)

        with torch.no_grad():
            q_new = torch.min(
                self.q1(obs, new_action),
                self.q2(obs, new_action),
            )
        v_tgt = q_new - self.alpha * logp

        v_pred = v_pred.float()
        v_tgt = v_tgt.float()

        v_loss = F.mse_loss(v_pred, v_tgt)

        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        # ===========================
        # Policy update every 2 steps
        # ===========================
        if self.total_steps % 2 == 0:
            aa, lp, _, _ = self.actor(obs)

            q_new2 = torch.min(
                self.q1(obs, aa),
                self.q2(obs, aa),
            )

            policy_loss = (self.alpha * lp - q_new2).mean()

            self.actor_opt.zero_grad()
            policy_loss.backward()
            self.actor_opt.step()

            self.soft_update()

        self.total_steps += 1

        losses = {
            "policy_loss": float(policy_loss) if policy_loss is not None else None,
            "q1_loss": float(q1_loss),
            "q2_loss": float(q2_loss),
            "v_loss": float(v_loss),
            "alpha_loss": float(alpha_loss),
        }

        logger.debug(
            "[SACAgent.update] total_steps=%d | "
            "policy_loss=%s, q1_loss=%.6f, q2_loss=%.6f, v_loss=%.6f, alpha_loss=%.6f, alpha=%.5f",
            self.total_steps,
            f"{losses['policy_loss']:.6f}" if losses["policy_loss"] is not None else "None",
            losses["q1_loss"],
            losses["q2_loss"],
            losses["v_loss"],
            losses["alpha_loss"],
            float(self.alpha.detach().cpu().item()),
        )

        return losses

    # -------------------------------------------------------------
    def soft_update(self):
        """Polyak averaging for target V-network."""
        with torch.no_grad():
            for t, s in zip(self.v_tgt.parameters(), self.v.parameters()):
                t.data.copy_(self.tau * s.data + (1.0 - self.tau) * t.data)
