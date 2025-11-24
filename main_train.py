##############################
# train_mof_multi_env.py  
# Multi-ENV + Curriculum Horizon + Full Logging + XYZ trajectory dump
# ★ per-structure RL version (critical fix)
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
    maxBytes=20_000_000,
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
# CIF SAMPLING
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
EPOCHS       = 1500
BASE_STEPS   = 300
FINAL_STEPS  = 1000
HORIZON_SCH  = 500

FMAX_THRESH  = 0.05
BUFFER_SIZE  = 1_000_000      # per-structure buffer → much smaller OK
BATCH_SIZE   = 64

CHECKPOINT_INTERVAL = 5

############################################################
# TRAIN LOOP
############################################################
logger.info(f"[MACS-MOF] Training start (EPOCHS={EPOCHS})")

global_start = time.time()

# replay buffer (will reallocate after env.init)
replay = None
agent = None

for ep in range(EPOCHS):

    logger.info("\n" + "="*80)
    logger.info(f"[EP {ep}] START")

    ############################################################
    # Horizon schedule
    ############################################################
    ratio = min(ep / HORIZON_SCH, 1.0)
    max_steps = int(BASE_STEPS + (FINAL_STEPS - BASE_STEPS) * ratio)
    max_steps = min(max_steps, FINAL_STEPS)

    logger.info(f"[EP {ep}] max_steps = {max_steps}")

    ################################################################
    # Snapshot directory
    ################################################################
    snap_dir = f"snapshots/EP{ep:04d}"
    os.makedirs(snap_dir, exist_ok=True)

    traj_path = os.path.join(snap_dir, "traj.xyz")
    energy_path = os.path.join(snap_dir, "energies.txt")

    if os.path.exists(traj_path): os.remove(traj_path)
    if os.path.exists(energy_path): os.remove(energy_path)

    ############################################################
    # Load CIF + init ENV
    ############################################################
    cif = sample_cif()
    atoms = read(cif)
    atoms.calc = calc

    env = MOFEnv(
        atoms_loader=lambda: atoms,
        k_neighbors=12,
        fmax_threshold=FMAX_THRESH,
        max_steps=max_steps,
        cmax=0.03,
    )

    obs = env.reset()
    logger.info(f"[EP {ep}] CIF loaded: {cif}")

    N_atom = env.N
    single_obs_dim = obs.shape[1]
    OBS_DIM = N_atom * single_obs_dim      # per-structure obs
    ACT_DIM = N_atom * 3                   # per-structure action

    ############################################################
    # allocate replay + agent (only at EP 0)
    ############################################################
    if ep == 0:
        logger.info(f"[INIT] OBS_DIM={OBS_DIM:,}, ACT_DIM={ACT_DIM:,}")
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

        logger.info("[INIT] Agent + ReplayBuffer allocated.")

    ############################################
    # EPISODE LOOP
    ############################################
    ep_ret = 0.0

    for step in tqdm(range(max_steps), desc=f"[EP {ep}]", ncols=120):

        ############################################################
        # per-structure action (flattened)
        ############################################################
        obs_flat = obs.reshape(-1)
        act_flat = agent.act(obs_flat)         # shape (ACT_DIM,)
        act = act_flat.reshape(N_atom, 3)

        next_obs, rew, done = env.step(act)

        ############################################################
        # per-structure reward (critical)
        ############################################################
        global_rew = float(np.mean(rew))

        ############################################################
        # Store one transition per structure
        ############################################################
        next_obs_flat = next_obs.reshape(-1)

        replay.store(
            obs_flat,
            act_flat,
            global_rew,
            next_obs_flat,
            done
        )

        if len(replay) > agent.batch_size:
            agent.update()

        obs = next_obs
        ep_ret += global_rew

        ############################################################
        # Save trajectory
        ############################################################
        env.atoms.write(traj_path, append=True)

        # Save energy
        E_total = env.atoms.get_potential_energy()
        E_pa = E_total / env.N
        with open(energy_path, "a") as f:
            f.write(f"{step} {E_total:.8f} {E_pa:.8f}\n")

        # Logging step stats
        f = np.linalg.norm(env.forces, axis=1)
        logger.info(
            f"[EP {ep}][STEP {step}] "
            f"Natom={env.N} | Favg={np.mean(f):.6f} Fmax={np.max(f):.6f} "
            f"rew={global_rew:.6f} | replay={len(replay):,} | alpha={float(agent.alpha):.6f}"
        )

        if done:
            logger.info(f"[EP {ep}] terminated early at step={step}")
            break

    ############################################################
    # EPISODE END
    ############################################################
    logger.info(f"[EP {ep}] return={ep_ret:.6f}")
    logger.info(f"[EP {ep}] replay_size={len(replay):,}")

    if ep % CHECKPOINT_INTERVAL == 0 and ep > 0:
        save_checkpoint(ep, agent, tag="interval")


############################################################
# FINAL CHECKPOINT
############################################################
save_checkpoint(EPOCHS, agent, tag="final")

logger.info("[TRAIN DONE]")
logger.info(f"wallclock={(time.time()-global_start)/3600:.3f} hr")

print("== training finished ==")
