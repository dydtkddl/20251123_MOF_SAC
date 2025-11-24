###############################################################
# main_train.py — HYBRID-MACS FINAL VERSION (3D actions)
# Author: ChatGPT for Yongsang
###############################################################

import os
import time
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
import torch

from ase.io import read
from mace.calculators import MACECalculator

from env.mof_env import MOFEnv
from sac.agent import SACAgent
from utils.replay_buffer import ReplayBuffer


###############################################################
# LOGGING
###############################################################
log_handler = RotatingFileHandler("train.log", maxBytes=25_000_000, backupCount=10)
log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)


###############################################################
# CHECKPOINT SAVE
###############################################################
def save_checkpoint(ep, agent, tag="auto"):
    os.makedirs("checkpoints", exist_ok=True)

    ckpt = {
        "epoch": ep,
        "actor": agent.actor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "v": agent.v.state_dict(),
        "v_target": agent.v_target.state_dict(),
        "log_alpha": float(agent.log_alpha.detach().cpu())
    }

    path = f"checkpoints/ckpt_ep{ep:04d}_{tag}.pt"
    torch.save(ckpt, path)
    logger.info(f"[CHECKPOINT] saved => {path}")


###############################################################
# CIF Sampling
###############################################################
POOL_DIR = "mofs/train_pool_valid"

def sample_cif():
    if not os.path.exists(POOL_DIR):
        raise FileNotFoundError(f"POOL_DIR not found: {POOL_DIR}")

    cifs = [
        os.path.join(POOL_DIR, f)
        for f in os.listdir(POOL_DIR)
        if f.endswith(".cif")
    ]
    if len(cifs) == 0:
        raise RuntimeError(f"No CIF found in {POOL_DIR}")

    return np.random.choice(cifs)


###############################################################
# MACE Calculator
###############################################################
calc = MACECalculator(
    model_paths=["mofs_v2.model"],
    head="pbe_d3",
    device="cuda",
    default_dtype="float32"
)


###############################################################
# TRAIN CONFIG
###############################################################
EPOCHS      = 1500
BASE_STEPS  = 200
FINAL_STEPS = 900
HORIZON_SCH = 500

ACT_DIM     = 3         # ★★ MACS-style 3D action ★★
FMAX_THRESH = 0.05

BUFFER_SIZE = 10_000_000
BATCH_SIZE  = 256
CKPT_INT    = 5


###############################################################
# GLOBAL HOLDERS
###############################################################
OBS_DIM  = None
agent    = None
replay   = None


###############################################################
# TRAINING START
###############################################################
logger.info(f"======== Hybrid-MACS RL TRAIN START (epochs={EPOCHS}) ========")
global_start = time.time()

for ep in range(EPOCHS):

    logger.info("\n" + "=" * 90)
    logger.info(f"[EP {ep}] START")

    ###############################################################
    # Curriculum horizon schedule
    ###############################################################
    r = min(ep / HORIZON_SCH, 1.0)
    max_steps = int(BASE_STEPS + (FINAL_STEPS - BASE_STEPS) * r)
    logger.info(f"[EP {ep}] max_steps = {max_steps}")

    ###############################################################
    # Snapshot dirs
    ###############################################################
    snap_dir = f"snapshots/EP{ep:04d}"
    os.makedirs(snap_dir, exist_ok=True)

    traj_xyz   = os.path.join(snap_dir, "traj.xyz")
    energy_log = os.path.join(snap_dir, "energy.txt")

    if os.path.exists(traj_xyz):  os.remove(traj_xyz)
    if os.path.exists(energy_log): os.remove(energy_log)

    ###############################################################
    # CIF Load
    ###############################################################
    cif = sample_cif()
    atoms = read(cif)
    atoms.calc = calc

    logger.info(f"[EP {ep}] CIF selected: {cif}")
    logger.info(f"[EP {ep}] atoms = {len(atoms)}")

    # loader closure
    def loader():
        a = atoms.copy()
        a.calc = calc
        return a

    ###############################################################
    # Create environment
    ###############################################################
    env = MOFEnv(
        atoms_loader=loader,
        fmax_threshold=FMAX_THRESH,
        max_steps=max_steps,
        k_neighbors=12,
        disp_scale=0.03,
    )

    obs = env.reset()
    N_atom = env.N

    obs_dim = obs.shape[1]
    logger.info(f"[EP {ep}] OBS_DIM = {obs_dim}")

    ###############################################################
    # Observation dimension consistency check
    ###############################################################
    if ep == 0:
        OBS_DIM = obs_dim
    else:
        if obs_dim != OBS_DIM:
            logger.warning(
                f"[EP {ep}] OBS_DIM mismatch: expected={OBS_DIM}, got={obs_dim} — skipping episode."
            )
            continue

    ###############################################################
    # Initialize Agent + Replay (first episode only)
    ###############################################################
    if agent is None:
        replay = ReplayBuffer(
            obs_dim=OBS_DIM,
            max_size=BUFFER_SIZE,
            n_step=1,
            gamma=0.995,
            alpha=0.6,
            beta=0.4
        )

        agent = SACAgent(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,   # ★ 3D action
            replay_buffer=replay,
            gamma=0.995,
            n_step=1,
            lr=3e-4,
            tau=5e-3,
            device="cuda",
            batch_size=BATCH_SIZE
        )

        logger.info("[INIT] SACAgent + ReplayBuffer initialized.")

    ###############################################################
    # BEGIN EPISODE
    ###############################################################
    replay.new_episode()

    ep_ret = 0.0
    done_reason = "none"
    com_drift_sum = 0.0

    energy_hist = []
    tqdm_bar = tqdm(range(max_steps), desc=f"[EP {ep}]", ncols=120)

    for step in tqdm_bar:

        ###############################################################
        # Compute per-atom stochastic actions (3D)
        ###############################################################
        # actor.act() returns (3,) vector for each atom
        act_list = [agent.act(obs[i]) for i in range(N_atom)]
        action_arr = np.array(act_list, np.float32).reshape(N_atom, ACT_DIM)

        ###############################################################
        # ENV STEP
        ###############################################################
        next_obs, reward, done, done_reason, Etot, Fmax = env.step(action_arr)
        reward = reward.astype(np.float32)

        ###############################################################
        # Logging values
        ###############################################################
        Fnorm = np.linalg.norm(env.forces, axis=1)
        E_avg = float(Etot) / N_atom

        energy_hist.append(E_avg)
        Emean_hist = float(np.mean(energy_hist))
        Estd_hist = float(np.std(energy_hist))

        com_now = env.atoms.positions.mean(axis=0)
        COM_step = float(np.linalg.norm(com_now - env.com_prev))
        com_drift_sum += COM_step

        disp_mag = np.linalg.norm(env.disp_last, axis=1)

        ###############################################################
        # Replay Buffer Store (per atom)
        ###############################################################
        for i in range(N_atom):
            replay.store(obs[i], action_arr[i], reward[i], next_obs[i], done)

        ###############################################################
        # Save trajectory and energy
        ###############################################################
        env.atoms.write(traj_xyz, append=True)
        with open(energy_log, "a") as f:
            f.write(f"{step} {Etot:.8f} {E_avg:.8f}\n")

        ###############################################################
        # Rich step logging
        ###############################################################
        logger.info(
            f"[EP {ep}][STEP {step}] "
            f"N={N_atom} | "
            f"Fmax={Fmax:.3e} Fmean={float(np.mean(Fnorm)):.3e} | "
            f"E={Etot:.4f}, E/atom={E_avg:.6f} mean_hist={Emean_hist:.6f} | "
            f"COM_step={COM_step:.5f} COM_total={com_drift_sum:.5f} | "
            f"Disp_mean={float(np.mean(disp_mag)):.3e} Disp_max={float(np.max(disp_mag)):.3e} | "
            f"rew_mean={float(np.mean(reward)):.6f} rmax={float(np.max(reward)):.6f} | "
            f"alpha={float(agent.alpha):.5f} | "
            f"buffer={len(replay):,}"
        )

        ep_ret += float(np.mean(reward))
        obs = next_obs

        if done:
            logger.info(f"[EP {ep}] TERMINATED at step={step}, reason={done_reason}")
            break

    ###############################################################
    # END EPISODE
    ###############################################################
    logger.info(f"[EP {ep}] return={ep_ret:.6f}")

    if done_reason in ["com", "bond"]:
        replay.end_episode(keep=False)
        logger.info(f"[EP {ep}] BAD episode → transitions dropped")
    else:
        replay.end_episode(keep=True)

    logger.info(f"[EP {ep}] Replay size = {len(replay):,}")

    ###############################################################
    # SAC UPDATE
    ###############################################################
    if len(replay) > agent.batch_size:
        losses = agent.update()
        logger.info(
            f"[UPDATE] "
            f"q1={losses['q1_loss']:.6f} | "
            f"q2={losses['q2_loss']:.6f} | "
            f"v={losses['v_loss']:.6f} | "
            f"pi={losses['policy_loss']} | "
            f"alpha={losses['alpha']:.5f}"
        )

    ###############################################################
    # PERIODIC CHECKPOINT
    ###############################################################
    if ep % CKPT_INT == 0 and ep > 0:
        save_checkpoint(ep, agent, tag="interval")

###############################################################
# FINAL SAVE
###############################################################
save_checkpoint(EPOCHS, agent, tag="final")

logger.info(f"======= TRAINING DONE (wallclock={(time.time()-global_start)/3600:.3f} hr) =======")
print("== TRAINING FINISHED ==")
