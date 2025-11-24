###############################################################
# sac/agent.py — Structure-Level Soft Actor Critic (MACS Global)
# -------------------------------------------------------------
# 옵션 1 완전 적용:
#   - ReplayBuffer는 obs_atom만 저장
#   - update()에서 obs_atom → obs_global (flatten)
#   - 구조-level SAC(Actor: N*F → 3N)
#   - PER + n-step + twin-Q + V-target 완전 지원
###############################################################

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from .actor import Actor
from .critic import CriticV, TwinCriticQ


class SACAgent:
    """
    Structure-Level SAC Agent (Global Policy)

    obs_global_dim = N_atoms * per_atom_feature_dim
    act_global_dim = 3 * N_atoms

    replay_buffer는 다음을 제공함:
        batch:
            obs_atom     : (B, N, F)
            nobs_atom    : (B, N, F)
            act          : (B, 3N)
            rew          : (B,)
            done         : (B,)
            weights      : (B,)
            idx          : (B,)
    """

    def __init__(
        self,
        obs_global_dim: int,
        act_global_dim: int,
        replay_buffer,

        device="cuda",
        gamma=0.995,
        tau=5e-3,
        batch_size=256,
        lr=3e-4,
        n_step=1,

        # target entropy = -sqrt(act_dim) 권장
        target_entropy=None
    ):

        # device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # replay buffer
        self.replay = replay_buffer
        self.batch_size = batch_size

        # discount (n-step)
        self.gamma = gamma
        self.n_step = n_step
        self.gamma_n = gamma ** n_step

        # target entropy
        if target_entropy is None:
            target_entropy = -np.sqrt(act_global_dim)
        self.target_entropy = float(target_entropy)

        # ------------------------------------------------------
        # Networks
        # ------------------------------------------------------
        n_atoms = act_global_dim // 3

        self.actor = Actor(obs_global_dim, n_atoms=n_atoms).to(self.device)
        self.critic = TwinCriticQ(obs_global_dim, act_global_dim).to(self.device)
        self.v = CriticV(obs_global_dim).to(self.device)
        self.v_target = CriticV(obs_global_dim).to(self.device)
        self.v_target.load_state_dict(self.v.state_dict())

        # ------------------------------------------------------
        # Optimizers
        # ------------------------------------------------------
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.critic.Q2.parameters(), lr=lr)
        self.v_opt = optim.Adam(self.v.parameters(), lr=lr)

        # ------------------------------------------------------
        # Entropy α
        # ------------------------------------------------------
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

        # soft update parameter
        self.tau = tau
        self.total_steps = 0


    #######################################################################
    @property
    def alpha(self):
        return self.log_alpha.exp()


    #######################################################################
    # Deterministic Action (Evaluation & Environment)
    #######################################################################
    @torch.no_grad()
    def act(self, obs_global):
        """
        obs_global: numpy vector (obs_global_dim,)
        returns: flattened action (3N,)
        """
        if isinstance(obs_global, np.ndarray):
            obs_t = torch.as_tensor(obs_global, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            obs_t = obs_global.to(self.device).unsqueeze(0)

        action = self.actor.act_tensor(obs_t)  # returns (N, 3)
        return action.flatten().astype(np.float32)


    #######################################################################
    # SAC update
    #######################################################################
    def update(self):

        batch = self.replay.sample(self.batch_size)

        # ------------------------------------------------------
        # 1) Load batch & flatten obs_atom → obs_global
        # ------------------------------------------------------
        obs_atom  = torch.as_tensor(batch["obs_atom"],  dtype=torch.float32, device=self.device)
        nobs_atom = torch.as_tensor(batch["nobs_atom"], dtype=torch.float32, device=self.device)

        # (B, N, F) → (B, N*F)
        B = obs_atom.shape[0]
        obs_global  = obs_atom.reshape(B, -1)
        nobs_global = nobs_atom.reshape(B, -1)

        act   = torch.as_tensor(batch["act"],  dtype=torch.float32, device=self.device)
        rew   = torch.as_tensor(batch["rew"],  dtype=torch.float32, device=self.device).unsqueeze(-1)
        done  = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        w     = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        idxs  = batch["idx"]

        # ======================================================
        # 2) Entropy temperature α update
        # ======================================================
        a_sample, logp_a, _, _ = self.actor(obs_global)

        alpha_loss = -(self.log_alpha * (logp_a + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # α clamp for stability
        with torch.no_grad():
            self.log_alpha.clamp_(min=-6.0, max=-1.6)

        # ======================================================
        # 3) Q-function Update (Twin-Q)
        # ======================================================
        with torch.no_grad():
            v_next = self.v_target(nobs_global)
            q_backup = rew + (1 - done) * self.gamma_n * v_next

        q1_pred, q2_pred = self.critic(obs_global, act)

        td1 = q1_pred - q_backup
        td2 = q2_pred - q_backup

        q1_loss = (w * td1.pow(2)).mean()
        q2_loss = (w * td2.pow(2)).mean()

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
        # 4) V-network update
        # ======================================================
        v_pred = self.v(obs_global)

        with torch.no_grad():
            q_min = torch.min(
                self.critic.Q1(obs_global, a_sample),
                self.critic.Q2(obs_global, a_sample)
            )
            v_target = q_min - self.alpha * logp_a

        v_loss = (w * (v_pred - v_target).pow(2)).mean()

        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        # ======================================================
        # 5) Policy update (every 2 steps)
        # ======================================================
        policy_loss = None

        if self.total_steps % 2 == 0:

            a_new, logp_new, _, _ = self.actor(obs_global)

            q_min_new = torch.min(
                self.critic.Q1(obs_global, a_new),
                self.critic.Q2(obs_global, a_new)
            )

            policy_loss = (self.alpha * logp_new - q_min_new)
            policy_loss = (policy_loss * w).mean()

            self.actor_opt.zero_grad()
            policy_loss.backward()
            self.actor_opt.step()

            # Soft update of target V
            self.soft_update(self.v_target, self.v)

        self.total_steps += 1

        # ======================================================
        # Return logging dict
        # ======================================================
        return dict(
            policy_loss=float(policy_loss) if policy_loss is not None else None,
            q1_loss=float(q1_loss),
            q2_loss=float(q2_loss),
            v_loss=float(v_loss),
            alpha_loss=float(alpha_loss),
            alpha=float(self.alpha.detach().cpu().item()),
        )


    #######################################################################
    def soft_update(self, target, source):
        with torch.no_grad():
            for t, s in zip(target.parameters(), source.parameters()):
                t.data.mul_(1 - self.tau).add_(self.tau * s.data)
