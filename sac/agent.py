import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .actor import Actor
from .critic import CriticQ, CriticV


###############################################################################
# Soft Actor-Critic Agent for MOF Structure Optimization
###############################################################################
class SACAgent:
    """
    SAC agent for MOF structure optimization using:
        - scalar action = RL scale-factor (0~1)
        - displacement = scale * (-force_direction)
        - PER using TD-error priority
        - deep actor/critic networks
        - soft target V update
    """

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
        target_entropy=-0.5
    ):
        self.replay = replay_buffer
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ============================
        # Hyperparameters
        # ============================
        self.gamma = gamma
        self.tau = tau
        self.n_step = n_step
        self.gamma_n = gamma ** n_step

        # ============================
        # Networks
        # ============================
        self.actor = Actor(obs_dim, act_dim).to(self.device).float()

        self.q1 = CriticQ(obs_dim, act_dim).to(self.device).float()
        self.q2 = CriticQ(obs_dim, act_dim).to(self.device).float()

        self.v = CriticV(obs_dim).to(self.device).float()
        self.v_tgt = CriticV(obs_dim).to(self.device).float()
        self.v_tgt.load_state_dict(self.v.state_dict())

        # ============================
        # Optimizers
        # ============================
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)
        self.v_opt = optim.Adam(self.v.parameters(), lr=lr)

        # ============================
        # Temperature α
        # ============================
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = target_entropy

        self.total_steps = 0


    ###########################################################################
    # α (entropy temperature)
    ###########################################################################
    @property
    def alpha(self):
        return self.log_alpha.exp()


    ###########################################################################
    # Deterministic action (evaluation)
    ###########################################################################
    @torch.no_grad()
    def act(self, obs):
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        scale, _, _, _ = self.actor(obs_t)
        return scale.cpu().numpy()


    ###########################################################################
    # Main SAC update step
    ###########################################################################
    def update(self):

        batch = self.replay.sample(self.batch_size)

        obs   = torch.as_tensor(batch["obs"],  dtype=torch.float32, device=self.device)
        act   = torch.as_tensor(batch["act"],  dtype=torch.float32, device=self.device)
        rew   = torch.as_tensor(batch["rew"],  dtype=torch.float32, device=self.device).unsqueeze(-1)
        nobs  = torch.as_tensor(batch["nobs"], dtype=torch.float32, device=self.device)
        done  = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).unsqueeze(-1)

        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        idxs = batch["idx"]

        ###################################################################
        # 1. Update α (entropy temperature)
        ###################################################################
        scale_s, logp_s, _, _ = self.actor(obs)

        alpha_loss = -(self.log_alpha * (logp_s + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Clamp α for stability
        with torch.no_grad():
            self.log_alpha.data.clamp_(min=-4.0, max=0.0)


        ###################################################################
        # 2. Update Q networks (TD target)
        ###################################################################
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

        # PER priority update
        with torch.no_grad():
            priority = torch.max(td1.abs(), td2.abs()).cpu().numpy().flatten()
            for i, p in zip(idxs, priority):
                self.replay.update_priority(i, p)


        ###################################################################
        # 3. Update V(s)
        ###################################################################
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
        self.v_opt.step()


        ###################################################################
        # 4. Policy update (every 2 steps for stability)
        ###################################################################
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
            self.actor_opt.step()

            self.soft_update()

        self.total_steps += 1


        ###################################################################
        # Output log dictionary
        ###################################################################
        return dict(
            policy_loss=float(policy_loss) if policy_loss is not None else None,
            q1_loss=float(q1_loss),
            q2_loss=float(q2_loss),
            v_loss=float(v_loss),
            alpha_loss=float(alpha_loss),
            alpha=float(self.alpha.detach().cpu().numpy()),
        )


    ###########################################################################
    # Soft target update: V_target ← τ V + (1−τ) V_target
    ###########################################################################
    def soft_update(self):
        with torch.no_grad():
            for tgt, src in zip(self.v_tgt.parameters(), self.v.parameters()):
                tgt.data.mul_(1 - self.tau).add_(self.tau * src.data)
