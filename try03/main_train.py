##############################
# train_mof_multi_env.py  
# Multi-ENV + Short Episode + Low Perturb + Full Logging + XYZ trajectory dump
##############################

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


############################################################
# ROTATING LOGGING
############################################################
log_handler = RotatingFileHandler(
    "train.log",
    maxBytes=10_000_000,
    backupCount=10,
)
log_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
))
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)


############################################################
# CHECKPOINT SAVE FUNCTION
############################################################
def save_checkpoint(ep, agent, tag="auto"):
    os.makedirs("checkpoints", exist_ok=True)

    ckpt = {
        "epoch": ep,
        "actor": agent.actor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "v": agent.v.state_dict(),
        "v_tgt": agent.v_tgt.state_dict(),
        "log_alpha": float(agent.log_alpha.detach().cpu()),
    }
    p = f"checkpoints/ckpt_ep{ep:04d}_{tag}.pt"
    torch.save(ckpt, p)
    logger.info(f"[CHECKPOINT] saved => {p}")


############################################################
# pool DIR
############################################################
POOL_DIR = "mofs/train_pool_valid"

def sample_cif():
    cifs = [
        os.path.join(POOL_DIR, f)
        for f in os.listdir(POOL_DIR)
        if f.endswith(".cif")
    ]
    return np.random.choice(cifs)


############################################################
# LOW PERTURB
############################################################
def perturb(a, sigma=0.01):
    p = a.get_positions()
    p += np.random.normal(0, sigma, p.shape)
    a.set_positions(p)
    return a


############################################################
# MACE surrogate
############################################################
calc = MACECalculator(
    model_paths=["mofs_v2.model"],
    head="pbe_d3",
    device="cuda",
    default_dtype="float32"
)


############################################################
# CONFIG
############################################################
EPOCHS      = 200
MAX_STEPS   = 1000
SWITCH_N    = 1000
FMAX_THRESH = 0.05

BUFFER_SIZE = 3_000_000
BATCH_SIZE  = 256

OBS_DIM     = 204
ACT_DIM     = 3
CHECKPOINT_INTERVAL = 5


############################################################
# replay + agent
############################################################
replay = ReplayBuffer(obs_dim=OBS_DIM, act_dim=ACT_DIM, max_size=BUFFER_SIZE)

agent = SACAgent(
    obs_dim=OBS_DIM,
    act_dim=ACT_DIM,
    replay_buffer=replay,
    device="cuda",
    lr=3e-4,
    gamma=0.995,
    tau=5e-3,
    batch_size=BATCH_SIZE,
)


############################################################
# TRAIN LOOP (MULTI-ENV)
############################################################
logger.info(f"[MACS-MOF][Multi-ENV] Start training EPOCHS={EPOCHS}, STEPS={MAX_STEPS}, SWITCH_N={SWITCH_N}")

global_start = time.time()


for ep in range(EPOCHS):

    logger.info("\n" + "="*80)
    logger.info(f"[EP {ep}] START")

    obs = None
    env = None
    ep_ret = 0.0

    # ------------------------------
    # 🔥 EPISODE SNAPSHOT FOLDER
    # ------------------------------
    snap_dir = f"snapshots/EP{ep:04d}"
    os.makedirs(snap_dir, exist_ok=True)

    traj_path = os.path.join(snap_dir, "traj.xyz")
    energy_path = os.path.join(snap_dir, "energies.txt")

    if os.path.exists(traj_path):
        os.remove(traj_path)

    if os.path.exists(energy_path):
        os.remove(energy_path)

    # ------------------------------------------------------
    # MULTI-ENV LOOP
    # ------------------------------------------------------
    for step in tqdm(range(MAX_STEPS), desc=f"[EP {ep}]", ncols=120):

        # 🔥 STRUCTURE SWITCH
        if (step % SWITCH_N == 0) or (obs is None):
            cif = sample_cif()
            atoms = read(cif)
            atoms = perturb(atoms)
            atoms.calc = calc

            env = MOFEnv(
                atoms_loader=lambda: atoms,
                k_neighbors=12,
                fmax_threshold=FMAX_THRESH,
                max_steps=MAX_STEPS,
            )

            obs = env.reset()

            logger.info(f"[EP {ep}] [SWITCH] step={step} new CIF={cif}")

        # ---- RL INTERACTION ----
        act = agent.act(obs)
        next_obs, rew, done = env.step(act)

        # Store transitions
        for i in range(env.N):
            replay.store(obs[i], act[i], rew[i], next_obs[i], done)

        if len(replay) > agent.batch_size:
            agent.update()

        obs = next_obs
        ep_ret += float(np.mean(rew))

        # ------------------------------------------------------
        # 🔥 SAVE XYZ FRAME (append)
        # ------------------------------------------------------
        env.atoms.write(traj_path, append=True)

        # ------------------------------------------------------
        # 🔥 SAVE ENERGY
        # ------------------------------------------------------
        E_total = env.atoms.get_potential_energy()
        E_pa = E_total / env.N

        with open(energy_path, "a") as f:
            f.write(f"{step} {E_total:.8f} {E_pa:.8f}\n")

        # Force stats
        f = np.linalg.norm(env.forces, axis=1)
        f_avg = float(np.mean(f))
        f_max = float(np.max(f))
        f_min = float(np.min(f))

        # Per-step log
        logger.info(
            f"[EP {ep}][STEP {step}] "
            f"Natom={env.N} | "
            f"Favg={f_avg:.6f} Fmax={f_max:.6f} Fmin={f_min:.6f} | "
            f"rew={float(np.mean(rew)):.6f} | "
            f"replay={len(replay):,} | "
            f"alpha={float(agent.alpha):.6f}"
        )

        if done:
            break


    logger.info(f"[EP {ep}] return={ep_ret:.6f}")
    logger.info(f"[EP {ep}] replay={len(replay):,}")

    if ep % CHECKPOINT_INTERVAL == 0 and ep > 0:
        save_checkpoint(ep, agent, tag="interval")


# ----------------------------
# final checkpoint
# ----------------------------
save_checkpoint(EPOCHS, agent, tag="final")

logger.info("[TRAIN DONE]")
logger.info(f"wallclock={(time.time()-global_start)/3600:.3f} hr")

print("== training finished ==")
