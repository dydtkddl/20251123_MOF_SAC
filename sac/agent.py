###############################################################
# sac/agent.py — Hybrid-MACS Stable SAC Agent (Final Version)
# Fully compatible with:
#   - Dynamic obs_dim from Hybrid-MACS env
#   - ReplayBuffer (PER + n-step + episode-aware)
#   - actor.act_tensor()
#   - critic networks
###############################################################

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from .actor import Actor
from .critic import CriticQ, CriticV


#######################################################################
# SAC Agent for Hybrid-MACS (scale-action, multi-agent implicit)
#######################################################################
class SACAgent:

    def __init__(
        self,
        obs_dim,
        replay_buffer,
        act_dim=1,
        device="cuda",
        gamma=0.995,
        tau=5e-3,
        batch_size=256,
        lr=3e-4,
        n_step=1,
        target_entropy=-0.3
    ):
        #############################################
        # Device
        #############################################
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        #############################################
        # Replay Buffer
        #############################################
        self.replay = replay_buffer
        self.batch_size = batch_size

        #############################################
        # Discount (n-step compatible)
        #############################################
        self.gamma = gamma
        self.n_step = n_step
        self.gamma_n = gamma ** n_step  # for n-step TD target

        #############################################
        # Networks
        #############################################
        self.actor = Actor(obs_dim, act_dim).to(self.device)

        self.q1 = CriticQ(obs_dim, act_dim).to(self.device)
        self.q2 = CriticQ(obs_dim, act_dim).to(self.device)

        self.v = CriticV(obs_dim).to(self.device)
        self.v_target = CriticV(obs_dim).to(self.device)
        self.v_target.load_state_dict(self.v.state_dict())  # initial sync

        #############################################
        # Optimizers
        #############################################
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)
        self.v_opt = optim.Adam(self.v.parameters(), lr=lr)

        #############################################
        # Entropy Temperature α
        #############################################
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = target_entropy

        #############################################
        # counters
        #############################################
        self.tau = tau
        self.total_steps = 0


    ###################################################################
    @property
    def alpha(self):
        """ α = exp(log_alpha) """
        return self.log_alpha.exp()


    ###################################################################
    # Deterministic action interface for environment stepping
    ###################################################################
    @torch.no_grad()
    def act(self, obs):
        """
        obs: numpy (1D or 2D) → return scale in [0, 1]
        Uses actor.act_tensor() for consistency.
        """
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.actor.act_tensor(obs_t)  # → (N_atom, 1)


    ###################################################################
    # SAC UPDATE
    ###################################################################
    def update(self):

        batch = self.replay.sample(self.batch_size)

        obs   = torch.as_tensor(batch["obs"],  dtype=torch.float32, device=self.device)
        act   = torch.as_tensor(batch["act"],  dtype=torch.float32, device=self.device)
        rew   = torch.as_tensor(batch["rew"],  dtype=torch.float32, device=self.device).unsqueeze(-1)
        nobs  = torch.as_tensor(batch["nobs"], dtype=torch.float32, device=self.device)
        done  = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).unsqueeze(-1)

        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        idxs = batch["idx"]

        ###############################################################
        # 1. α update (entropy temperature)
        ###############################################################
        scale_s, logp_s, _, _ = self.actor(obs)

        alpha_loss = -(self.log_alpha * (logp_s + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.alpha_opt.param_groups[0]["params"], 0.5)
        self.alpha_opt.step()

        # clamp alpha for stability
        with torch.no_grad():
            self.log_alpha.data.clamp_(min=-4.0, max=-1.2)


        ###############################################################
        # 2. Q1, Q2 Update
        ###############################################################
        with torch.no_grad():
            v_next = self.v_target(nobs)
            q_target = rew + (1 - done) * self.gamma_n * v_next

        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)

        td1 = q1_pred - q_target
        td2 = q2_pred - q_target

        q1_loss = (weights * td1.pow(2)).mean()
        q2_loss = (weights * td2.pow(2)).mean()

        # Q1
        self.q1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 0.5)
        self.q1_opt.step()

        # Q2
        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 0.5)
        self.q2_opt.step()

        # PER priority update (max TD-error per sample)
        with torch.no_grad():
            td_err = torch.max(td1.abs(), td2.abs()).cpu().numpy().flatten()
            for i, v in zip(idxs, td_err):
                self.replay.update_priority(i, float(v + 1e-6))


        ###############################################################
        # 3. V-network Update
        ###############################################################
        v_pred = self.v(obs)

        with torch.no_grad():
            q_min = torch.min(
                self.q1(obs, scale_s),
                self.q2(obs, scale_s)
            )
            v_target = q_min - self.alpha * logp_s

        v_loss = (weights * (v_pred - v_target).pow(2)).mean()

        self.v_opt.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), 0.5)
        self.v_opt.step()


        ###############################################################
        # 4. Policy Update (every 2 steps)
        ###############################################################
        policy_loss = None

        if self.total_steps % 2 == 0:

            scale2, logp2, _, _ = self.actor(obs)

            q_min2 = torch.min(
                self.q1(obs, scale2),
                self.q2(obs, scale2)
            )

            policy_loss = (self.alpha * logp2 - q_min2)
            policy_loss = (policy_loss * weights).mean()

            self.actor_opt.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()

            # target V-network update
            self.soft_update()

        self.total_steps += 1


        ###############################################################
        # Return logs
        ###############################################################
        return dict(
            policy_loss=float(policy_loss) if policy_loss is not None else None,
            q1_loss=float(q1_loss),
            q2_loss=float(q2_loss),
            v_loss=float(v_loss),
            alpha_loss=float(alpha_loss),
            alpha=float(self.alpha.detach().cpu().numpy())
        )


    ###################################################################
    # Soft Update Target Network
    ###################################################################
    def soft_update(self):
        with torch.no_grad():
            for tgt, src in zip(self.v_target.parameters(), self.v.parameters()):
                tgt.data.mul_(1 - self.tau).add_(self.tau * src.data)
