import logging
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from .actor import GaussianPolicy
from .critic import TwinQNetwork

logger = logging.getLogger(__name__)


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        target_entropy: float = None,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = automatic_entropy_tuning

        # 네트워크
        self.actor = GaussianPolicy(obs_dim, act_dim).to(device)
        self.critic = TwinQNetwork(obs_dim, act_dim).to(device)
        self.critic_target = TwinQNetwork(obs_dim, act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 옵티마이저
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # entropy temperature
        if target_entropy is None:
            # 일반적으로 -|A|
            target_entropy = -float(act_dim)
        self.target_entropy = target_entropy

        if self.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.log_alpha = None
            self.alpha_opt = None
            self.alpha = alpha

        logger.info(
            f"[SACAgent] obs_dim={obs_dim}, act_dim={act_dim}, "
            f"gamma={gamma}, tau={tau}, auto_alpha={self.auto_alpha}, "
            f"target_entropy={self.target_entropy}"
        )

    # --------------------------------------------------------
    # Action 선택
    # --------------------------------------------------------

    def select_action(self, obs_np, deterministic: bool = False):
        """
        obs_np: (N, obs_dim)
        반환: (N, act_dim)
        """
        self.actor.eval()
        with torch.no_grad():
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
            actions, _, mean_action = self.actor.sample(obs, deterministic=deterministic)
            if deterministic:
                out = mean_action
            else:
                out = actions
            actions_np = out.cpu().numpy()
        self.actor.train()
        return actions_np

    # --------------------------------------------------------
    # 파라미터 업데이트 (one gradient step)
    # --------------------------------------------------------

    def update_parameters(
        self, replay_buffer, batch_size: int
    ) -> Dict[str, Any]:
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = replay_buffer.sample(batch_size)

        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        # 1. Critic 업데이트
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_state_batch)
            q1_next, q2_next = self.critic_target(next_state_batch, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = reward_batch + (1.0 - done_batch) * self.gamma * q_next

        q1, q2 = self.critic(state_batch, action_batch)
        q1_loss = torch.mean((q1 - q_target) ** 2)
        q2_loss = torch.mean((q2 - q_target) ** 2)
        critic_loss = q1_loss + q2_loss

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # 2. Actor 업데이트
        pi, log_pi, _ = self.actor.sample(state_batch)
        q1_pi, q2_pi = self.critic(state_batch, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_pi - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # 3. Alpha 업데이트
        alpha_loss_val = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_val = alpha_loss.item()

        # 4. Target network soft update
        with torch.no_grad():
            for p, p_tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_tgt.data.mul_(1.0 - self.tau)
                p_tgt.data.add_(self.tau * p.data)

        info = {
            "critic_loss": float(critic_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha),
            "alpha_loss": float(alpha_loss_val),
        }

        return info
