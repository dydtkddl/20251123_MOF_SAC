###############################################################
# sac/agent.py — Structure-Level SAC (MACS Global Policy)
# -------------------------------------------------------------
# 완전 구조-level RL로 재구현된 최종 버전
#
# state : obs_global_dim (flatten)
# action: act_global_dim = 3*N_atoms
#
# 구성요소:
#   Actor(obs_global) → action_global, logp
#   CriticQ(obs_global, act_global)
#   CriticV(obs_global)
#   Target V network
#   α auto-tuning
#   PER ReplayBuffer 지원
###############################################################

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from .actor import Actor
from .critic import CriticV, CriticQ, TwinCriticQ


class SACAgent:
    """
    Structure-Level Soft Actor Critic Agent
    ---------------------------------------
    obs_global_dim : flatten OBS dimension (N * obs_dim_atom)
    act_global_dim : flatten ACTION dimension (N * 3)

    replay_buffer :
        Structure-level ReplayBuffer (obs_global, act_global)
    """

    def __init__(
        self,
        obs_global_dim: int,
        act_global_dim: int,         # = 3 * N_atoms
        replay_buffer,
        device="cuda",
        gamma=0.995,
        tau=5e-3,
        batch_size=256,
        lr=3e-4,
        n_step=1,
        target_entropy=None          # set to -sqrt(act_dim)
    ):

        # ----------------------------------------------------------
        # Device 설정
        # ----------------------------------------------------------
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ----------------------------------------------------------
        # Replay Buffer
        # ----------------------------------------------------------
        self.replay = replay_buffer
        self.batch_size = batch_size

        # ----------------------------------------------------------
        # Discount
        # ----------------------------------------------------------
        self.gamma = gamma
        self.n_step = n_step
        self.gamma_n = gamma ** n_step

        # ----------------------------------------------------------
        # Target Entropy 정의
        # ----------------------------------------------------------
        if target_entropy is None:
            # ✔ 구조-level act_dim = 3*N → sqrt로 조정
            target_entropy = -np.sqrt(act_global_dim)
        self.target_entropy = float(target_entropy)

        # ----------------------------------------------------------
        # Networks (Actor, Critic)
        # ----------------------------------------------------------
        self.actor = Actor(obs_global_dim, n_atoms=act_global_dim // 3).to(self.device)

        self.critic = TwinCriticQ(obs_global_dim, act_global_dim).to(self.device)
        self.v = CriticV(obs_global_dim).to(self.device)
        self.v_target = CriticV(obs_global_dim).to(self.device)
        self.v_target.load_state_dict(self.v.state_dict())

        # ----------------------------------------------------------
        # Optimizers
        # ----------------------------------------------------------
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.critic.Q2.parameters(), lr=lr)
        self.v_opt = optim.Adam(self.v.parameters(), lr=lr)

        # ----------------------------------------------------------
        # Entropy temperature α
        # ----------------------------------------------------------
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

        # ----------------------------------------------------------
        # Soft-update parameter
        # ----------------------------------------------------------
        self.tau = tau
        self.total_steps = 0


    # ----------------------------------------------------------
    @property
    def alpha(self):
        return self.log_alpha.exp()


    # ----------------------------------------------------------
    # Deterministic Action (Evaluation / Environment)
    # ----------------------------------------------------------
    @torch.no_grad()
    def act(self, obs_global):
        """
        obs_global: numpy (obs_global_dim,)
        return: numpy (act_global_dim,)
        """
        if isinstance(obs_global, np.ndarray):
            obs_t = torch.as_tensor(obs_global, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            obs_t = obs_global.to(self.device).unsqueeze(0)

        action = self.actor.act_tensor(obs_t)  # returns (N,3) numpy
        return action.flatten().astype(np.float32)



    # ----------------------------------------------------------
    # Main SAC Update step
    # ----------------------------------------------------------
    def update(self):

        batch = self.replay.sample(self.batch_size)

        obs   = torch.as_tensor(batch["obs"],  dtype=torch.float32, device=self.device)
        act   = torch.as_tensor(batch["act"],  dtype=torch.float32, device=self.device)
        rew   = torch.as_tensor(batch["rew"],  dtype=torch.float32, device=self.device).unsqueeze(-1)
        nobs  = torch.as_tensor(batch["nobs"], dtype=torch.float32, device=self.device)
        done  = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).unsqueeze(-1)

        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        idxs = batch["idx"]

        # ======================================================
        # 1) α update (entropy temperature)
        # ======================================================
        a_sample, logp_a, _, _ = self.actor(obs)

        alpha_loss = -(self.log_alpha * (logp_a + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # alpha clamp (stability)
        with torch.no_grad():
            self.log_alpha.clamp_(min=-6.0, max=-1.6)


        # ======================================================
        # 2) Q-function Update (Twin-Q)
        # ======================================================
        with torch.no_grad():
            v_next = self.v_target(nobs)
            q_backup = rew + (1 - done) * self.gamma_n * v_next

        q1_pred, q2_pred = self.critic(obs, act)

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
            self.replay.update_priority(idxs, td_err)


        # ======================================================
        # 3) V-network Update
        # ======================================================
        v_pred = self.v(obs)

        with torch.no_grad():
            q_min = torch.min(
                self.critic.Q1(obs, a_sample),
                self.critic.Q2(obs, a_sample)
            )
            v_target = q_min - self.alpha * logp_a

        v_loss = (weights * (v_pred - v_target) ** 2).mean()

        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()


        # ======================================================
        # 4) Policy Update (every 2 steps)
        # ======================================================
        policy_loss = None

        if self.total_steps % 2 == 0:

            a_new, logp_new, _, _ = self.actor(obs)

            q_min_new = torch.min(
                self.critic.Q1(obs, a_new),
                self.critic.Q2(obs, a_new)
            )

            policy_loss = (self.alpha * logp_new - q_min_new)
            policy_loss = (policy_loss * weights).mean()

            self.actor_opt.zero_grad()
            policy_loss.backward()
            self.actor_opt.step()

            # --------------------------------------------------
            # Soft-update V-target
            # --------------------------------------------------
            self.soft_update(self.v_target, self.v)

        self.total_steps += 1


        # ======================================================
        # 결과 반환 (logging)
        # ======================================================
        return dict(
            policy_loss=float(policy_loss) if policy_loss is not None else None,
            q1_loss=float(q1_loss),
            q2_loss=float(q2_loss),
            v_loss=float(v_loss),
            alpha_loss=float(alpha_loss),
            alpha=float(self.alpha.detach().cpu().item()),
        )



    # ----------------------------------------------------------
    def soft_update(self, target, source):
        with torch.no_grad():
            for t, s in zip(target.parameters(), source.parameters()):
                t.data.mul_(1 - self.tau).add_(self.tau * s.data)
