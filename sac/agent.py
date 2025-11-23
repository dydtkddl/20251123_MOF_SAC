# sac/agent.py

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .actor import Actor
from .critic import CriticQ, CriticV


class SACAgent:

    def __init__(
        self,
        obs_dim,
        act_dim,
        replay_buffer,
        device="cuda",
        gamma=0.995,
        tau=5e-3,
        batch_size=256,
        lr=3e-4,
    ):

        self.replay = replay_buffer
        self.batch_size = batch_size

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        self.gamma = gamma
        self.tau = tau

        # -----------------
        # models
        # -----------------
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.v     = CriticV(obs_dim).to(self.device)
        self.v_tgt = CriticV(obs_dim).to(self.device)
        self.q1    = CriticQ(obs_dim, act_dim).to(self.device)
        self.q2    = CriticQ(obs_dim, act_dim).to(self.device)

        self.v_tgt.load_state_dict(self.v.state_dict())

        # -----------------
        # optimizers
        # -----------------
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.v_opt     = optim.Adam(self.v.parameters(), lr=lr)
        self.q1_opt    = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt    = optim.Adam(self.q2.parameters(), lr=lr)

        # -----------------
        # entropy tuning
        # -----------------
        self.target_entropy = -act_dim
        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=self.device
        )
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

        self.total_steps = 0


        ########################################################
        # FORCE ALL NETWORK PARAMETERS TO FP32
        ########################################################
        self.actor.float()
        self.v.float()
       	self.v_tgt.float()
        self.q1.float()
        self.q2.float()



    @property
    def alpha(self):
        return self.log_alpha.exp()



    @torch.no_grad()
    def act(self, obs):

        obs = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        )

        a, _, _, _ = self.actor(obs)
        return a.cpu().numpy()



    ############################################################
    def update(self):

        batch = self.replay.sample(self.batch_size)

        # FP32 -- critical
        obs  = torch.as_tensor(batch["obs"],  dtype=torch.float32, device=self.device)
        act  = torch.as_tensor(batch["act"],  dtype=torch.float32, device=self.device)
        rew  = torch.as_tensor(batch["rew"],  dtype=torch.float32, device=self.device).unsqueeze(1)
        nobs = torch.as_tensor(batch["nobs"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).unsqueeze(1)


        # α update
        new_action, logp, _, _ = self.actor(obs)
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()


        # Q update
        with torch.no_grad():
            v_next = self.v_tgt(nobs)
            q_target = rew + (1 - done) * self.gamma * v_next

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


        # V update
        v_pred = self.v(obs)
        with torch.no_grad():
            q_new = torch.min(
                self.q1(obs, new_action),
                self.q2(obs, new_action),
            )
        v_tgt = q_new - self.alpha * logp

        v_loss = F.mse_loss(v_pred, v_tgt)

        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()


        # Policy update
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



    ############################################################
    def soft_update(self):

        with torch.no_grad():
            for t, s in zip(self.v_tgt.parameters(), self.v.parameters()):
                t.data.copy_(
                    self.tau * s.data + (1 - self.tau) * t.data
                )
