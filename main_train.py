###############################################################
# train_mof_scale_rl.py (Enhanced Monitoring Version)
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

log_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
))

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
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
POOL_DIR = "mofs/train_pool_valid"

def sample_cif():
    if not os.path.exists(POOL_DIR):
        raise FileNotFoundError(f"[ERROR] CIF directory not found: {POOL_DIR}")

    cifs = [
        os.path.join(POOL_DIR, f)
        for f in os.listdir(POOL_DIR)
        if f.endswith(".cif")
    ]

    if len(cifs) == 0:
        raise RuntimeError(
            f"[ERROR] No CIF files found in {POOL_DIR}. "
            "Add training CIFs first."
        )

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
# GLOBALS
###############################################################
OBS_DIM = None
ACT_DIM = 1
agent = None
replay = None


###############################################################
# TRAINING START
###############################################################
logger.info(f"[MOF-SCALE-RL] Training start (epochs = {EPOCHS})")
global_start = time.time()


for ep in range(EPOCHS):

    logger.info("\n" + "="*80)
    logger.info(f"[EP {ep}] START")

    ###############################################################
    # 1. Curriculum Horizon
    ###############################################################
    ratio = min(ep / HORIZON_SCH, 1.0)
    max_steps = int(BASE_STEPS + (FINAL_STEPS - BASE_STEPS) * ratio)
    logger.info(f"[EP {ep}] max_steps = {max_steps}")

    ###############################################################
    # 2. Snapshot Directory
    ###############################################################
    snap_dir = f"snapshots/EP{ep:04d}"
    os.makedirs(snap_dir, exist_ok=True)

    traj_xyz = os.path.join(snap_dir, "traj.xyz")
    energy_log = os.path.join(snap_dir, "energy.txt")

    if os.path.exists(traj_xyz):
        os.remove(traj_xyz)
    if os.path.exists(energy_log):
        os.remove(energy_log)

    ###############################################################
    # 3. Load CIF + Env Init
    ###############################################################
    cif = sample_cif()
    atoms = read(cif)
    atoms.calc = calc
    logger.info(f"[EP {ep}] CIF loaded: {cif}")

    def loader():
        a = atoms.copy()
        a.calc = calc
        return a

    env = MOFEnv(
        atoms_loader=loader,
        fmax_threshold=FMAX_THRESH,
        max_steps=max_steps,
        disp_scale=0.03,
    )

    obs = env.reset()

    N_atom = env.N
    obs_dim = obs.shape[1]

    ###############################################################
    # 4. First episode only
    ###############################################################
    if ep == 0:
        OBS_DIM = obs_dim
        logger.info(f"[INIT] OBS_DIM={OBS_DIM} | ACT_DIM={ACT_DIM}")

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
    # 5. EPISODE
    ###############################################################
    replay.new_episode()

    ep_ret = 0.0
    done_reason = "none"

    com_drift_cum = 0.0
    energy_history = []

    for step in tqdm(range(max_steps), desc=f"[EP {ep}]", ncols=120):

        scale_list = []
        for i in range(N_atom):
            s = agent.act(obs[i])
            scale_list.append(s)

        scale_arr = np.array(scale_list, dtype=np.float32).reshape(-1, 1)

        ###########################################################
        # ENV STEP
        ###########################################################
        next_obs, reward, done, done_reason, Etot, Fmax = env.step(scale_arr)
        reward = reward.astype(np.float32)

        Fnorm = np.linalg.norm(env.forces, axis=1)
        Fmin  = float(np.min(Fnorm))
        Fmean = float(np.mean(Fnorm))
        Fstd  = float(np.std(Fnorm))
        Fmed  = float(np.median(Fnorm))

        E_avg = Etot / N_atom
        energy_history.append(E_avg)
        Emean_hist = float(np.mean(energy_history))
        Estd_hist = float(np.std(energy_history))

        com_now = env.atoms.positions.mean(axis=0)
        COM_drift_step = float(np.linalg.norm(com_now - env.com_prev))
        com_drift_cum += COM_drift_step

        disp_mag = np.linalg.norm(env.disp_last, axis=1)
        Disp_mean = float(np.mean(disp_mag))
        Disp_max  = float(np.max(disp_mag))

        rew_mean = float(np.mean(reward))
        rew_std  = float(np.std(reward))
        rew_min  = float(np.min(reward))
        rew_max  = float(np.max(reward))

        ###########################################################
        # STORE (per-atom)
        ###########################################################
        for i in range(N_atom):
            replay.store(
                obs[i], scale_arr[i], reward[i], next_obs[i], done
            )

        ###########################################################
        # TRAJECTORY SAVE
        ###########################################################
        env.atoms.write(traj_xyz, append=True)
        with open(energy_log, "a") as f:
            f.write(f"{step} {Etot:.8f} {E_avg:.8f}\n")

        ###########################################################
        # EXTENDED LOGGING (NEW)
        ###########################################################
        logger.info(
            f"[EP {ep}][STEP {step}] "
            f"N={N_atom} | "
            f"Fmax={Fmax:.3e} Fmin={Fmin:.3e} Fmean={Fmean:.3e} Fstd={Fstd:.3e} Fmed={Fmed:.3e} | "
            f"E={Etot:.3f} E/atom={E_avg:.5f} Emean_hist={Emean_hist:.5f} Estd_hist={Estd_hist:.5f} | "
            f"COM_step={COM_drift_step:.4f} COM_cum={com_drift_cum:.4f} | "
            f"Disp_mean={Disp_mean:.4e} Disp_max={Disp_max:.4e} | "
            f"rew_mean={rew_mean:.5f} rew_std={rew_std:.5f} rmin={rew_min:.5f} rmax={rew_max:.5f} | "
            f"alpha={float(agent.alpha):.5f} | buffer={len(replay):,}"
        )

        ep_ret += rew_mean
        obs = next_obs

        if done:
            logger.info(f"[EP {ep}] terminated at step={step} reason={done_reason}")
            break


    ###############################################################
    # 6. EP END
    ###############################################################
    logger.info(f"[EP {ep}] return={ep_ret:.6f}")

    BAD = ["com", "bond"]
    if done_reason in BAD:
        logger.info(f"[EP {ep}] BAD → discard episode")
        replay.end_episode(keep=False)
    else:
        logger.info(f"[EP {ep}] GOOD → keep transitions")
        replay.end_episode(keep=True)

    logger.info(f"[EP {ep}] replay_size={len(replay):,}")

    ###############################################################
    # 7. UPDATE
    ###############################################################
    if len(replay) > agent.batch_size:
        losses = agent.update()
        logger.info(
            f"[UPDATE] q1={losses['q1_loss']:.5f} "
            f"q2={losses['q2_loss']:.5f} "
            f"v={losses['v_loss']:.5f} "
            f"pi={losses['policy_loss']}"
        )

    ###############################################################
    # 8. CHECKPOINT
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


