import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .actor import Actor
from .critic import CriticQ, CriticV


class SACAgent:
    """
    Upgraded per-atom SAC agent:
    - PER (TD-error priority)
    - N-step return (via ReplayBuffer)
    - PER importance sampling weights
    - Temperature α clamp for stability
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
        n_step=1
    ):

        self.replay = replay_buffer
        self.batch_size = batch_size

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # n-step discount (n_step=1 strongly recommended for structure optimization)
        self.n_step = n_step
        self.gamma = gamma
        self.gamma_n = gamma ** n_step

        self.tau = tau

        # ---------------------------
        # NETWORKS
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
        # SAC entropy temperature α
        # ---------------------------
        self.target_entropy = -1.0
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

        self.total_steps = 0


    @property
    def alpha(self):
        return self.log_alpha.exp()


    # -------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a, _, _, _ = self.actor(obs)
        return a.cpu().numpy()


    # -------------------------------------------------------------
    # UPDATE SAC (with PER + TD-error priority update)
    # -------------------------------------------------------------
    def update(self):

        batch = self.replay.sample(self.batch_size)

        obs   = torch.as_tensor(batch["obs"],  dtype=torch.float32, device=self.device)
        act   = torch.as_tensor(batch["act"],  dtype=torch.float32, device=self.device)
        rew   = torch.as_tensor(batch["rew"],  dtype=torch.float32, device=self.device).unsqueeze(1)
        nobs  = torch.as_tensor(batch["nobs"], dtype=torch.float32, device=self.device)
        done  = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).unsqueeze(1)

        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device).unsqueeze(1)
        idxs = batch["idx"]

        ###############################################################
        # α update (entropy temperature)
        ###############################################################
        new_action, logp, _, _ = self.actor(obs)
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ★ temperature clamp → structure optimization 안정화
        with torch.no_grad():
            self.log_alpha.data.clamp_(min=-4.0, max=-1.0)


        ###############################################################
        # Q update (TD target)
        ###############################################################
        with torch.no_grad():
            v_next = self.v_tgt(nobs)
            q_target = rew + (1 - done) * self.gamma_n * v_next

        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)

        td1 = q1_pred - q_target
        td2 = q2_pred - q_target

        q1_loss = (weights * td1.pow(2)).mean()
        q2_loss = (weights * td2.pow(2)).mean()

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # ★ PER TD-error priority update
        with torch.no_grad():
            new_priority = (td1.abs() + td2.abs()).cpu().numpy().flatten()
            for i, p in zip(idxs, new_priority):
                self.replay.update_priority(i, p)


        ###############################################################
        # V update
        ###############################################################
        v_pred = self.v(obs)

        with torch.no_grad():
            q_new = torch.min(
                self.q1(obs, new_action),
                self.q2(obs, new_action)
            )
            v_target = q_new - self.alpha * logp

        v_loss = (weights * (v_pred - v_target).pow(2)).mean()

        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()


        ###############################################################
        # Policy update (every 2 steps)
        ###############################################################
        policy_loss = None

        if self.total_steps % 2 == 0:

            aa, lp, _, _ = self.actor(obs)

            q_new2 = torch.min(
                self.q1(obs, aa),
                self.q2(obs, aa),
            )

            policy_loss = (self.alpha * lp - q_new2)
            policy_loss = (policy_loss * weights).mean()

            self.actor_opt.zero_grad()
            policy_loss.backward()
            self.actor_opt.step()

            self.soft_update()

        self.total_steps += 1

        return {
            "policy_loss": float(policy_loss) if policy_loss is not None else None,
            "q1_loss": float(q1_loss),
            "q2_loss": float(q2_loss),
            "v_loss": float(v_loss),
            "alpha_loss": float(alpha_loss),
        }


    # -------------------------------------------------------------
    def soft_update(self):
        with torch.no_grad():
            for t, s in zip(self.v_tgt.parameters(), self.v.parameters()):
                t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)
