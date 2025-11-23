# sac/agent.py

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .actor import Actor
from .critic import CriticQ, CriticV


class SACAgent:

    def __init__(
            self,
            obs_dim,
            act_dim,
            replay_buffer,
            device='cuda',
            gamma=0.995,
            tau=5e-3,
            batch_size=256,
            lr=3e-4,
    ):

        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        # device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.tau = tau

        # actor
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)

        # critics
        self.vf = CriticV(obs_dim).to(self.device)
        self.vf_tgt = CriticV(obs_dim).to(self.device)
        self.vf_tgt.load_state_dict(self.vf.state_dict())

        self.vf_opt = optim.Adam(self.vf.parameters(), lr=lr)

        self.q1 = CriticQ(obs_dim, act_dim).to(self.device)
        self.q2 = CriticQ(obs_dim, act_dim).to(self.device)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)

        # entropy tuning
        self.target_entropy = -act_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

        self.total_steps = 0
        self.last_loss = 0.0


    @property
    def alpha(self):
        return self.log_alpha.exp()



    # =========================================================
    #   ACTION
    # =========================================================

    @torch.no_grad()
    def act(self, obs):

        obs = torch.FloatTensor(obs).to(self.device)

        a, _, _, _ = self.actor(obs)
        return a.cpu().numpy()



    # =========================================================
    #   TRAIN
    # =========================================================

    def update(self):

        batch = self.replay_buffer.sample()

        obs = torch.FloatTensor(batch['obs']).to(self.device)
        act = torch.FloatTensor(batch['act']).to(self.device)
        rew = torch.FloatTensor(batch['rew']).unsqueeze(1).to(self.device)
        nobs = torch.FloatTensor(batch['nobs']).to(self.device)
        done = torch.FloatTensor(batch['done']).unsqueeze(1).to(self.device)

        # ---------------------------------------
        # train alpha (dual optimization)
        with torch.no_grad():
            a_new, logp_new, _, _ = self.actor(obs)

        alpha_loss = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()


        # ---------------------------------------
        # Q functions

        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)

        with torch.no_grad():
            v_next = self.vf_tgt(nobs)
            q_target = rew + (1-done)*self.gamma*v_next


        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()


        # ---------------------------------------
        # V function

        v_pred = self.vf(obs)

        with torch.no_grad():
            q_new = torch.min(self.q1(obs, a_new), self.q2(obs, a_new))

        v_target = q_new - self.alpha * logp_new
        v_loss = F.mse_loss(v_pred, v_target)

        self.vf_opt.zero_grad()
        v_loss.backward()
        self.vf_opt.step()


        # ---------------------------------------
        # Actor (policy) update

        if self.total_steps % 2 == 0:

            a_new2, logp2, _, _ = self.actor(obs)
            q_new2 = torch.min(self.q1(obs, a_new2), self.q2(obs, a_new2))

            policy_loss = (self.alpha * logp2 - q_new2).mean()

            self.actor_opt.zero_grad()
            policy_loss.backward()
            self.actor_opt.step()

            self.soft_update()


        self.last_loss = float(v_loss.item())



    # =========================================================

    def soft_update(self):
        with torch.no_grad():
            for tgt, src in zip(self.vf_tgt.parameters(), self.vf.parameters()):
                tgt.data.mul_(1-self.tau)
                tgt.data.add_(self.tau * src.data)



    # =========================================================

    def train_loop(self, env, total_steps):

        obs = env.reset()

        for _ in tqdm(range(total_steps)):

            self.total_steps += 1

            act = self.act(obs)
            nobs, rew, done = env.step(act)

            # per-atom push
            for i in range(len(act)):
                self.replay_buffer.store(obs[i], act[i], rew[i], nobs[i], done)

            obs = nobs

            if self.replay_buffer.size > self.batch_size:
                self.update()

            if done:
                obs = env.reset()
