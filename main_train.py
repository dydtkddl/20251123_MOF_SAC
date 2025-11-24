###############################################################
# train_mof_scale_rl.py (FINAL, CLEAN, READY-TO-RUN)
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
# LOGGING SETUP
###############################################################
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)

log_handler = RotatingFileHandler(
    "train.log",
    maxBytes=20_000_000,
    backupCount=10,
)
log_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
))
logger.addHandler(log_handler)


###############################################################
# CHECKPOINT
###############################################################
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


###############################################################
# CIF SAMPLING
###############################################################
POOL_DIR = "mofs/train_pool"

def sample_cif():
    cifs = [
        os.path.join(POOL_DIR, f)
        for f in os.listdir(POOL_DIR)
        if f.endswith(".cif")
    ]
    return np.random.choice(cifs)


###############################################################
# MACE Surrogate
###############################################################
calc = MACECalculator(
    model_paths=["mofs_v2.model"],
    head="pbe_d3",
    device="cuda",
    default_dtype="float32"
)


###############################################################
# CONFIG
###############################################################
EPOCHS        = 1500
BASE_STEPS    = 200
FINAL_STEPS   = 900
HORIZON_SCH   = 500

FMAX_THRESH   = 0.05
BUFFER_SIZE   = 2_000_000
BATCH_SIZE    = 256
CHECKPOINT_INTERVAL = 5


###############################################################
# TRAIN LOOP
###############################################################
logger.info(f"[MOF-SCALE-RL] Training start (epochs = {EPOCHS})")
global_start = time.time()

agent = None
replay = None


for ep in range(EPOCHS):

    logger.info("\n" + "="*80)
    logger.info(f"[EP {ep}] START")

    ###############################################################
    # Curriculum Horizon
    ###############################################################
    ratio = min(ep / HORIZON_SCH, 1.0)
    max_steps = int(BASE_STEPS + (FINAL_STEPS - BASE_STEPS) * ratio)
    logger.info(f"[EP {ep}] max_steps = {max_steps}")

    ###############################################################
    # Snapshot dirs
    ###############################################################
    snap_dir = f"snapshots/EP{ep:04d}"
    os.makedirs(snap_dir, exist_ok=True)

    traj_xyz = os.path.join(snap_dir, "traj.xyz")
    if os.path.exists(traj_xyz):
        os.remove(traj_xyz)

    ###############################################################
    # Load CIF + Env
    ###############################################################
    cif = sample_cif()
    atoms = read(cif)
    atoms.calc = calc

    env = MOFEnv(
        atoms_loader=lambda: atoms.copy(),  # fresh copy each reset
        max_steps=max_steps,
        disp_scale=0.03,
        fmax_threshold=FMAX_THRESH,
    )

    obs = env.reset()
    N_atom = env.N
    obs_dim = obs.shape[1]

    logger.info(f"[EP {ep}] CIF loaded: {cif}")
    logger.info(f"[EP {ep}] obs_dim = {obs_dim} | N_atom = {N_atom}")

    ###############################################################
    # Initialize Agent + Replay (EP=0)
    ###############################################################
    if ep == 0:

        replay = ReplayBuffer(
            obs_dim=obs_dim,
            max_size=BUFFER_SIZE,
            n_step=1,
            gamma=0.995,
            alpha=0.6,
            beta=0.4
        )

        agent = SACAgent(
            obs_dim=obs_dim,
            act_dim=1,
            replay_buffer=replay,
            device="cuda",
            lr=3e-4,
            gamma=0.995,
            tau=5e-3,
            batch_size=BATCH_SIZE,
            n_step=1
        )

        logger.info("[INIT] Agent + ReplayBuffer allocated.")


    ###############################################################
    # EPISODE
    ###############################################################
    ep_ret = 0.0
    done_reason = "none"
    replay.new_episode()

    for step in tqdm(range(max_steps), desc=f"[EP {ep}]", ncols=110):

        ###########################################################
        # ACTION: batched per-atom scale prediction
        ###########################################################
        obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device)
        scale_batch, _, _, _ = agent.actor(obs_t)  # (N,1)
        scale_arr = scale_batch.detach().cpu().numpy()

        ###########################################################
        # ENV STEP
        ###########################################################
        next_obs, reward, done, reason = env.step(scale_arr)

        rew_scalar = float(reward)
        rew_atom = np.full((N_atom,), rew_scalar, dtype=np.float32)

        ###########################################################
        # Replay store (per atom)
        ###########################################################
        done_flag = 1.0 if done else 0.0

        for i in range(N_atom):
            replay.store(
                obs[i],
                scale_arr[i],
                rew_scalar,
                next_obs[i],
                done_flag,
            )

        ###########################################################
        # Dump snapshot
        ###########################################################
        env.atoms.write(traj_xyz, append=True)

        ###########################################################
        # Step logging
        ###########################################################
        logger.info(
            f"[EP {ep}][STEP {step}] "
            f"reward={rew_scalar:.6f} | "
            f"alpha={float(agent.alpha):.5f} | "
            f"buffer={len(replay):,}"
        )

        ep_ret += rew_scalar
        obs = next_obs

        if done:
            logger.info(f"[EP {ep}] terminated early at step={step} reason={reason}")
            break


    ###############################################################
    # EP END
    ###############################################################
    logger.info(f"[EP {ep}] return={ep_ret:.6f}")

    if reason in ("com", "bond"):
        logger.info(f"[EP {ep}] BAD episode → discard transitions")
        replay.end_episode(keep=False)
    else:
        logger.info(f"[EP {ep}] GOOD episode")
        replay.end_episode(keep=True)

    ###############################################################
    # TRAIN
    ###############################################################
    if len(replay) > agent.batch_size:
        losses = agent.update()
        logger.info(
            f"[UPDATE] q1={losses['q1_loss']:.5f} "
            f"q2={losses['q2_loss']:.5f} "
            f"v={losses['v_loss']:.5f} "
            f"pi={losses['policy_loss']} "
            f"alpha={losses['alpha_loss']:.5f}"
        )

    ###############################################################
    # CHECKPOINT
    ###############################################################
    if ep % CHECKPOINT_INTERVAL == 0 and ep > 0:
        save_checkpoint(ep, agent, tag="interval")


###############################################################
# FINAL SAVE
###############################################################
save_checkpoint(EPOCHS, agent, tag="final")

logger.info("[TRAIN DONE]")
logger.info(f"wallclock={(time.time() - global_start)/3600:.3f} hr")

print("== training finished ==")
