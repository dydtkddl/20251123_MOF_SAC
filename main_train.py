###############################################################
# main_train.py — Hybrid-MACS Full Version
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
log_handler = RotatingFileHandler(
    "train.log",
    maxBytes=20_000_000,
    backupCount=10,
)
log_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

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
        raise RuntimeError(f"No CIF files found in {POOL_DIR}")

    return np.random.choice(cifs)


###############################################################
# MACE MODEL
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

FMAX_THRESH = 0.05
BUFFER_SIZE = 10_000_000        # Hybrid-MACS: 10M buffer recommended
BATCH_SIZE  = 256
CKPT_INT    = 5


###############################################################
# GLOBALS
###############################################################
OBS_DIM  = None
ACT_DIM  = 1
agent    = None
replay   = None


###############################################################
# TRAINING START
###############################################################
logger.info(f"======== Hybrid-MACS RL TRAIN START (epochs={EPOCHS}) ========")
global_start = time.time()


for ep in range(EPOCHS):

    logger.info("\n" + "=" * 80)
    logger.info(f"[EP {ep}] START")

    ###############################################################
    # Curriculum horizon
    ###############################################################
    r = min(ep / HORIZON_SCH, 1.0)
    max_steps = int(BASE_STEPS + (FINAL_STEPS - BASE_STEPS) * r)
    logger.info(f"[EP {ep}] max_steps = {max_steps}")

    ###############################################################
    # Snapshot dirs
    ###############################################################
    snap_dir = f"snapshots/EP{ep:04d}"
    os.makedirs(snap_dir, exist_ok=True)

    traj_xyz  = os.path.join(snap_dir, "traj.xyz")
    energy_log = os.path.join(snap_dir, "energy.txt")

    for f in [traj_xyz, energy_log]:
        if os.path.exists(f):
            os.remove(f)

    ###############################################################
    # CIF Load
    ###############################################################
    cif = sample_cif()
    atoms = read(cif)
    atoms.calc = calc

    logger.info(f"[EP {ep}] CIF selected: {cif}")
    logger.info(f"[EP {ep}] CIF atoms: {len(atoms)}")

    def loader():
        a = atoms.copy()
        a.calc = calc
        return a

    ###############################################################
    # ENVIRONMENT (Hybrid-MACS)
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

    # Hybrid-MACS obs_dim auto
    obs_dim = obs.shape[1]
    logger.info(f"[EP {ep}] OBS_DIM from env.reset = {obs_dim}")

    ###############################################################
    # Hybrid-MACS: DEBUG FEATURE LOGGING
    ###############################################################
    # logger.info(f"[EP {ep}] feature_dim(center)     = {env.feature_dim_center}")
    # logger.info(f"[EP {ep}] feature_dim(neighbor)  = {env.feature_dim_neighbor}")
    # logger.info(f"[EP {ep}] k_neighbors            = {env.k}")
    # logger.info(f"[EP {ep}] Final obs_dim          = {env.obs_dim_total}")

    ###############################################################
    # obs_dim mismatch check (CIF-to-CIF consistency must hold)
    ###############################################################
    if ep == 0:
        OBS_DIM = obs_dim
    else:
        if obs_dim != OBS_DIM:
            logger.warning(
                f"[EP {ep}] OBS_DIM mismatch! expected={OBS_DIM}, got={obs_dim}. "
                f"This CIF will be skipped."
            )
            continue

    ###############################################################
    # INITIALIZE AGENT + BUFFER (first episode only)
    ###############################################################
    if ep == 0:

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
            act_dim=ACT_DIM,
            replay_buffer=replay,
            gamma=0.995,
            n_step=1,
            lr=3e-4,
            tau=5e-3,
            device="cuda",
            batch_size=BATCH_SIZE
        )

        logger.info("[INIT] SACAgent + ReplayBuffer created.")
        logger.info(f"[INIT] BUFFER={BUFFER_SIZE:,}, OBS_DIM={OBS_DIM}, ACT_DIM={ACT_DIM}")

    ###############################################################
    # EPISODE START
    ###############################################################
    replay.new_episode()
    ep_ret = 0.0
    done_reason = "none"

    energy_hist = []
    com_drift_sum = 0.0

    tqdm_bar = tqdm(range(max_steps), desc=f"[EP {ep}]", ncols=120)

    for step in tqdm_bar:

        ###############################################################
        # Compute per-atom actions (Hybrid-MACS: scale factor only)
        ###############################################################
        scale_list = [agent.act(obs[i]) for i in range(N_atom)]
        scale_arr = np.array(scale_list, np.float32).reshape(-1, 1)

        ###############################################################
        # ENV STEP (returns per-atom obs & reward)
        ###############################################################
        next_obs, reward, done, done_reason, Etot, Fmax = env.step(scale_arr)
        reward = reward.astype(np.float32)

        ###############################################################
        # Logging forces, energy, COM drift, displacement
        ###############################################################
        Fnorm = np.linalg.norm(env.forces, axis=1)
        E_avg = Etot / N_atom

        energy_hist.append(E_avg)
        Emean_hist = float(np.mean(energy_hist))
        Estd_hist  = float(np.std(energy_hist))

        com_now = env.atoms.positions.mean(axis=0)
        COM_step = float(np.linalg.norm(com_now - env.com_prev))
        com_drift_sum += COM_step

        disp_mag = np.linalg.norm(env.disp_last, axis=1)

        ###############################################################
        # Replay Buffer: per-atom store
        ###############################################################
        for i in range(N_atom):
            replay.store(obs[i], scale_arr[i], reward[i], next_obs[i], done)

        ###############################################################
        # Save trajectory XYZ + energy log
        ###############################################################
        env.atoms.write(traj_xyz, append=True)
        with open(energy_log, "a") as f:
            f.write(f"{step} {Etot:.8f} {E_avg:.8f}\n")

        ###############################################################
        # EXTENDED LOGGING
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
            logger.info(f"[EP {ep}] TERMINATED step={step}, reason={done_reason}")
            break

    ###############################################################
    # EPISODE END
    ###############################################################
    logger.info(f"[EP {ep}] return={ep_ret:.6f}")

    # bad episode?
    if done_reason in ["com", "bond"]:
        logger.info(f"[EP {ep}] BAD episode → drop transitions")
        replay.end_episode(keep=False)
    else:
        replay.end_episode(keep=True)

    logger.info(f"[EP {ep}] replay size = {len(replay):,}")

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
            f"alpha={losses['alpha']:.4f}"
        )

    ###############################################################
    # CHECKPOINT
    ###############################################################
    if ep % CKPT_INT == 0 and ep > 0:
        save_checkpoint(ep, agent, tag="interval")


###############################################################
# FINAL SAVE
###############################################################
save_checkpoint(EPOCHS, agent, tag="final")
logger.info(f"======= TRAINING DONE — wallclock={(time.time()-global_start)/3600:.3f} hr =======")
print("== training finished ==")
