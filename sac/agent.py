import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .actor import Actor
from .critic import CriticQ, CriticV


class SACAgent:
    """
    Stable per-atom SAC agent for MACS-MOF RL with:
        - target entropy = -1.0
        - update frequency K=4
        - target smoothing (EMA)
        - advantage normalization
        - FP32 enforced across entire chain
    """

    def __init__(
        self,
        obs_dim,
        replay_buffer,
        act_dim=3,
        device="cuda",
        gamma=0.995,
        tau=5e-3,
        batch_size=256,
        lr=3e-4,
        update_every=4,           # ★ update every K steps
        normalize_adv=True,       # ★ advantage normalization switch
        ema_beta=0.10             # ★ EMA smoothing factor
    ):

        self.replay = replay_buffer
        self.batch_size = batch_size
        self.update_every = update_every
        self.normalize_adv = normalize_adv
        self.ema_beta = ema_beta

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau

        # ---------------------------------------------------------
        # NETWORKS (FP32)
        # ---------------------------------------------------------
        self.actor = Actor(obs_dim, act_dim).to(self.device).float()
        self.v     = CriticV(obs_dim).to(self.device).float()
        self.v_tgt = CriticV(obs_dim).to(self.device).float()
        self.q1    = CriticQ(obs_dim, act_dim).to(self.device).float()
        self.q2    = CriticQ(obs_dim, act_dim).to(self.device).float()

        self.v_tgt.load_state_dict(self.v.state_dict())

        # ---------------------------------------------------------
        # OPTIMIZERS
        # ---------------------------------------------------------
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.v_opt     = optim.Adam(self.v.parameters(), lr=lr)
        self.q1_opt    = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt    = optim.Adam(self.q2.parameters(), lr=lr)

        # ---------------------------------------------------------
        # ENTROPY (target = -1.0)
        # ---------------------------------------------------------
        self.target_entropy = -1.0
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

        # ---------------------------------------------------------
        # Internal counters
        # ---------------------------------------------------------
        self.total_steps = 0
        self.q_target_ema = None  # ★ EMA buffer


    # ==============================================================
    @property
    def alpha(self):
        return self.log_alpha.exp()


    # ==============================================================
    @torch.no_grad()
    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a, _, _, _ = self.actor(obs)
        return a.cpu().numpy()


    # ==============================================================
    # SAC UPDATE
    # ==============================================================
    def update(self):

        # ----------------------------------------------------------
        # Update every K steps
        # ----------------------------------------------------------
        if self.total_steps % self.update_every != 0:
            self.total_steps += 1
            return None

        batch = self.replay.sample(self.batch_size)

        obs  = torch.as_tensor(batch["obs"],  dtype=torch.float32, device=self.device)
        act  = torch.as_tensor(batch["act"],  dtype=torch.float32, device=self.device)
        rew  = torch.as_tensor(batch["rew"],  dtype=torch.float32, device=self.device).unsqueeze(1)
        nobs = torch.as_tensor(batch["nobs"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).unsqueeze(1)


        # ===========================================================
        # 1) α update
        # ===========================================================
        new_action, logp, _, _ = self.actor(obs)

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()


        # ===========================================================
        # 2) Q update (with EMA target)
        # ===========================================================
        with torch.no_grad():
            v_next = self.v_tgt(nobs)
            q_target_raw = rew + (1 - done) * self.gamma * v_next  # original target

            # ★ EMA TARGET SMOOTHING
            if self.q_target_ema is None:
                self.q_target_ema = q_target_raw.clone()

            self.q_target_ema = (1 - self.ema_beta) * self.q_target_ema + \
                                 self.ema_beta * q_target_raw

            q_target = self.q_target_ema


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


        # ===========================================================
        # 3) V update
        # ===========================================================
        v_pred = self.v(obs)

        with torch.no_grad():
            q_new = torch.min(
                self.q1(obs, new_action),
                self.q2(obs, new_action)
            )
            v_tgt = q_new - self.alpha * logp

            # ★ Advantage Normalization
            if self.normalize_adv:
                v_tgt_mean = v_tgt.mean()
                v_tgt_std  = v_tgt.std() + 1e-6
                v_tgt = (v_tgt - v_tgt_mean) / v_tgt_std

        v_loss = F.mse_loss(v_pred.float(), v_tgt.float())

        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()


        # ===========================================================
        # 4) POLICY update (same frequency as critic)
        # ===========================================================
        aa, lp, _, _ = self.actor(obs)
        q_new2 = torch.min(self.q1(obs, aa), self.q2(obs, aa))

        policy_loss = (self.alpha * lp - q_new2).mean()

        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()


        # ===========================================================
        # 5) Soft-update V_target
        # ===========================================================
        self.soft_update()

        self.total_steps += 1

        return {
            "policy_loss": float(policy_loss),
            "q1_loss": float(q1_loss),
            "q2_loss": float(q2_loss),
            "v_loss": float(v_loss),
            "alpha_loss": float(alpha_loss),
        }


    # ==============================================================
    def soft_update(self):
        with torch.no_grad():
            for tgt, src in zip(self.v_tgt.parameters(), self.v.parameters()):
                tgt.data.copy_(self.tau * src.data + (1 - self.tau) * tgt.data)
