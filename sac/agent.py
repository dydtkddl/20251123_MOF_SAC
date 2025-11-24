###############################################################
# sac/agent.py — MACS 3D-Action Fully Compatible FINAL VERSION
###############################################################

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from .actor import Actor
from .critic import CriticQ, CriticV


class SACAgent:
    """
    MACS-compatible Soft Actor-Critic Agent
    - Supports 3D vector action (act_dim = 3)
    - PER replay buffer integration
    - Alpha auto-tuning (entropy regularization)
    - V-network + twin Q-networks
    """

    def __init__(
        self,
        obs_dim,
        replay_buffer,
        act_dim=3,                   # ★ MACS: 3D action
        device="cuda",
        gamma=0.995,
        tau=5e-3,
        batch_size=256,
        lr=3e-4,
        n_step=1,
        target_entropy=None          # SAC default: -|A|  => 여기선 -3 추천
    ):

        # Device ------------------------------------------------------------------
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Replay ------------------------------------------------------------------
        self.replay = replay_buffer
        self.batch_size = batch_size

        # Discount ----------------------------------------------------------------
        self.gamma = gamma
        self.n_step = n_step
        self.gamma_n = gamma ** n_step

        # Target entropy -----------------------------------------------------------
        if target_entropy is None:
            # 3D action이면 entropy 하한 더 낮게 설정해야 안정적임
            target_entropy = -act_dim
        self.target_entropy = float(target_entropy)

        # Networks ----------------------------------------------------------------
        self.actor = Actor(obs_dim, act_dim).to(self.device)

        self.q1 = CriticQ(obs_dim, act_dim).to(self.device)
        self.q2 = CriticQ(obs_dim, act_dim).to(self.device)

        self.v = CriticV(obs_dim).to(self.device)
        self.v_target = CriticV(obs_dim).to(self.device)
        self.v_target.load_state_dict(self.v.state_dict())

        # Optimizers --------------------------------------------------------------
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)
        self.v_opt = optim.Adam(self.v.parameters(), lr=lr)

        # α (entropy temperature) --------------------------------------------------
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

        # Soft update rate
        self.tau = tau
        self.total_steps = 0


    # ---------------------------------------------------------------------------
    @property
    def alpha(self):
        return self.log_alpha.exp()


    # ---------------------------------------------------------------------------
    # Deterministic action for environment
    # ---------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs):
        """
        obs: (obs_dim,) or (1, obs_dim)
        returns: (3,) np.float32
        """
        if obs.ndim == 1:
            obs = obs[None, :]

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action_vec = self.actor.act_tensor(obs_t)   # → (1,3)
        return action_vec.squeeze(0).cpu().numpy()


    # ---------------------------------------------------------------------------
    # SAC Update Step
    # ---------------------------------------------------------------------------
    def update(self):

        batch = self.replay.sample(self.batch_size)

        obs   = torch.as_tensor(batch["obs"],  dtype=torch.float32, device=self.device)
        act   = torch.as_tensor(batch["act"],  dtype=torch.float32, device=self.device)
        rew   = torch.as_tensor(batch["rew"],  dtype=torch.float32, device=self.device).unsqueeze(-1)
        nobs  = torch.as_tensor(batch["nobs"], dtype=torch.float32, device=self.device)
        done  = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).unsqueeze(-1)

        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        idxs = batch["idx"]

        # =======================================================================
        # 1) α update  (entropy temperature)
        # =======================================================================
        a_sample, logp_a, _, _ = self.actor(obs)

        alpha_loss = -(self.log_alpha * (logp_a + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # clamp α in reasonable range
        with torch.no_grad():
            self.log_alpha.clamp_(min=-4.0, max=1.0)


        # =======================================================================
        # 2) Q-function update (twin critics)
        # =======================================================================
        with torch.no_grad():
            v_next = self.v_target(nobs)
            q_backup = rew + (1 - done) * self.gamma_n * v_next   # Bellman backup

        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)

        td1 = q1_pred - q_backup
        td2 = q2_pred - q_backup

        q1_loss = (weights * (td1 ** 2)).mean()
        q2_loss = (weights * (td2 ** 2)).mean()

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # PER priority update
        with torch.no_grad():
            td_err = torch.max(td1.abs(), td2.abs()).cpu().numpy().flatten()
            for i, v in zip(idxs, td_err):
                self.replay.update_priority(i, float(v + 1e-6))


        # =======================================================================
        # 3) V-network update
        # =======================================================================
        v_pred = self.v(obs)

        with torch.no_grad():
            q_min = torch.min(
                self.q1(obs, a_sample),
                self.q2(obs, a_sample)
            )
            v_target = q_min - self.alpha * logp_a

        v_loss = (weights * (v_pred - v_target) ** 2).mean()

        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()


        # =======================================================================
        # 4) Policy (actor) update  — every 2 steps for stability
        # =======================================================================
        policy_loss = None
        if self.total_steps % 2 == 0:

            a_smpl2, logp2, _, _ = self.actor(obs)

            q_min2 = torch.min(
                self.q1(obs, a_smpl2),
                self.q2(obs, a_smpl2)
            )

            policy_loss = (self.alpha * logp2 - q_min2)
            policy_loss = (policy_loss * weights).mean()

            self.actor_opt.zero_grad()
            policy_loss.backward()
            self.actor_opt.step()

            # soft-update V-target
            self.soft_update()

        self.total_steps += 1

        # =======================================================================
        # Return losses
        # =======================================================================
        return dict(
            policy_loss=float(policy_loss) if policy_loss is not None else None,
            q1_loss=float(q1_loss),
            q2_loss=float(q2_loss),
            v_loss=float(v_loss),
            alpha_loss=float(alpha_loss),
            alpha=float(self.alpha.detach().cpu().numpy()),
        )


    # ---------------------------------------------------------------------------
    def soft_update(self):
        with torch.no_grad():
            for tgt, src in zip(self.v_target.parameters(), self.v.parameters()):
                tgt.data.mul_(1 - self.tau).add_(self.tau * src.data)
