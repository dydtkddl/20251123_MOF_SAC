##############################################
# train_mof_multi_env.py  
# Per-Atom RL version (MACS-style)
# Stable SAC + Warm-up + Reward Clipping + Update Frequency
##############################################

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


##############################################
# LOGGING SETUP
##############################################
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


##############################################
# CHECKPOINT
##############################################
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
    logger.info(f"[CHECKPOINT] Saved => {p}")


##############################################
# CIF SAMPLING
##############################################
POOL_DIR = "mofs/train_pool_valid"

def sample_cif():
    cifs = [
        os.path.join(POOL_DIR, f)
        for f in os.listdir(POOL_DIR)
        if f.endswith(".cif")
    ]
    return np.random.choice(cifs)


##############################################
# MACE Surrogate
##############################################
calc = MACECalculator(
    model_paths=["mofs_v2.model"],
    head="pbe_d3",
    device="cuda",
    default_dtype="float32"
)


##############################################
# CONFIG
##############################################
EPOCHS       = 1500
BASE_STEPS   = 300
FINAL_STEPS  = 1000
HORIZON_SCH  = 500

FMAX_THRESH  = 0.05
BUFFER_SIZE  = 200_000          # ★ smaller buffer for stable RL
BATCH_SIZE   = 256

CHECKPOINT_INTERVAL = 5


##############################################
# GLOBALS
##############################################
OBS_DIM = None
ACT_DIM = 3   # per-atom action = 3-dim
replay = None
agent = None


##############################################
# TRAIN START
##############################################
logger.info(f"[MACS-MOF] Training start (EPOCHS={EPOCHS})")
global_start = time.time()


for ep in range(EPOCHS):

    logger.info("\n" + "="*80)
    logger.info(f"[EP {ep}] START")

    ##################################
    # Curriculum Horizon
    ##################################
    ratio = min(ep / HORIZON_SCH, 1.0)
    max_steps = int(BASE_STEPS + (FINAL_STEPS - BASE_STEPS) * ratio)
    logger.info(f"[EP {ep}] max_steps = {max_steps}")

    ##################################
    # Snapshot folders
    ##################################
    snap_dir = f"snapshots/EP{ep:04d}"
    os.makedirs(snap_dir, exist_ok=True)

    traj_path = os.path.join(snap_dir, "traj.xyz")
    en_path   = os.path.join(snap_dir, "energies.txt")

    if os.path.exists(traj_path): os.remove(traj_path)
    if os.path.exists(en_path): os.remove(en_path)

    ##################################
    # Load CIF and Init Env
    ##################################
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
    obs_dim = obs.shape[1]

    ##################################
    # EP0: Initialize Replay + Agent
    ##################################
    if ep == 0:
        OBS_DIM = obs_dim

        logger.info(f"[INIT] OBS_DIM={OBS_DIM}, ACT_DIM=3 (per-atom)")

        replay = ReplayBuffer(
            obs_dim=OBS_DIM,
            max_size=BUFFER_SIZE,
            reward_weight=2.0,
            warmup=10_000,
        )

        agent = SACAgent(
            obs_dim=OBS_DIM,
            act_dim=3,
            replay_buffer=replay,
            device="cuda",
            lr=3e-4,
            gamma=0.995,
            tau=5e-3,
            batch_size=BATCH_SIZE,
            update_every=4,           # ★ update frequency
            normalize_adv=True,
        )
        logger.info("[INIT] Agent + ReplayBuffer allocated (per-atom).")


    ##################################
    # EPISODE
    ##################################
    ep_ret = 0.0

    for step in tqdm(range(max_steps), desc=f"[EP {ep}]", ncols=120):

        ########################
        # ACTION (per-atom)
        ########################
        action = np.zeros((N_atom, 3), float)

        for i in range(N_atom):
            action[i] = agent.act(obs[i])

        ########################
        # STEP ENV
        ########################
        next_obs, reward, done = env.step(action)

        # ------------------------------
        # ★ reward clipping for stability
        # ------------------------------
        clipped_reward = np.clip(reward, -5.0, 5.0)

        ########################
        # STORE PER-ATOM
        ########################
        for i in range(N_atom):
            replay.store(
                obs[i],
                action[i],
                clipped_reward[i],
                next_obs[i],
                done,
            )

        ########################
        # UPDATE SAC (after warm-up)
        ########################
        if replay.ready():
            agent.update()

        ########################
        # TRAJECTORY SAVE
        ########################
        env.atoms.write(traj_path, append=True)

        Etot = env.atoms.get_potential_energy()
        E_pa = Etot / N_atom

        with open(en_path, "a") as f:
            f.write(f"{step} {Etot:.8f} {E_pa:.8f}\n")

        f_norm = np.linalg.norm(env.forces, axis=1)

        logger.info(
            f"[EP {ep}][STEP {step}] "
            f"N={N_atom} | "
            f"Favg={np.mean(f_norm):.6f} Fmax={np.max(f_norm):.6f} "
            f"rew_mean={float(np.mean(clipped_reward)):.6f} | "
            f"replay={len(replay):,} | "
            f"alpha={float(agent.alpha):.5f}"
        )

        ep_ret += float(np.mean(clipped_reward))
        obs = next_obs

        if done:
            logger.info(f"[EP {ep}] terminated early at step={step}")
            break


    ##################################
    # EP END
    ##################################
    logger.info(f"[EP {ep}] return={ep_ret:.6f}")
    logger.info(f"[EP {ep}] replay_size={len(replay):,}")

    if ep % CHECKPOINT_INTERVAL == 0 and ep > 0:
        save_checkpoint(ep, agent, tag="interval")


##############################################
# FINAL SAVE
##############################################
save_checkpoint(EPOCHS, agent, tag="final")

logger.info("[TRAIN DONE]")
logger.info(f"wallclock={(time.time() - global_start)/3600:.3f} hr")

print("== training finished ==")
