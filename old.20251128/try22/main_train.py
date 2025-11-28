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
# LOGGING SETUP
###############################################################
log_handler = RotatingFileHandler(
    "train.log", maxBytes=30_000_000, backupCount=20
)
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
        "log_alpha": float(agent.log_alpha.detach().cpu()),
    }

    path = f"checkpoints/ckpt_ep{ep:04d}_{tag}.pt"
    torch.save(ckpt, path)
    logger.info(f"[CHECKPOINT] Saved: {path}")



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
EPOCHS = 1500

BASE_STEPS  = 200
FINAL_STEPS = 900
HORIZON_SCH = 500     # curriculum schedule

ACT_DIM = 3            # ★★★ MACS 3D action ★★★
FMAX_THRESH = 0.05

BUFFER_SIZE = 10_000_000
BATCH_SIZE = 256
CKPT_INT = 5



###############################################################
# GLOBAL HOLDERS
###############################################################
OBS_DIM = None
agent = None
replay = None



###############################################################
# TRAINING START
###############################################################
logger.info(f"======== Hybrid-MACS RL TRAIN START (epochs={EPOCHS}) ========")
global_start = time.time()


for ep in range(EPOCHS):

    logger.info("\n" + "=" * 100)
    logger.info(f"[EP {ep}] START")

    #######################################################################
    # Curriculum Horizon Schedule
    #######################################################################
    r = min(ep / HORIZON_SCH, 1.0)
    max_steps = int(BASE_STEPS + (FINAL_STEPS - BASE_STEPS) * r)
    logger.info(f"[EP {ep}] max_steps = {max_steps}")

    #######################################################################
    # Snapshot Directory
    #######################################################################
    snap_dir = f"snapshots/EP{ep:04d}"
    os.makedirs(snap_dir, exist_ok=True)

    traj_xyz = os.path.join(snap_dir, "traj.xyz")
    energy_log = os.path.join(snap_dir, "energy.txt")

    if os.path.exists(traj_xyz):  os.remove(traj_xyz)
    if os.path.exists(energy_log): os.remove(energy_log)

    #######################################################################
    # CIF LOAD
    #######################################################################
    cif = sample_cif()
    atoms = read(cif)
    atoms.calc = calc

    logger.info(f"[EP {ep}] CIF = {cif}")
    logger.info(f"[EP {ep}] N_atom = {len(atoms)}")

    # loader closure for env
    def loader():
        a = atoms.copy()
        a.calc = calc
        return a

    #######################################################################
    # Create Environment
    #######################################################################
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

    logger.info(f"[EP {ep}] OBS_DIM(observed) = {obs_dim}")


    #######################################################################
    # Observation dimension consistency check
    #######################################################################
    if ep == 0:
        OBS_DIM = obs_dim
        logger.info(f"[INIT] OBS_DIM set = {OBS_DIM}")

    else:
        if obs_dim != OBS_DIM:
            logger.warning(
                f"[EP {ep}] OBS_DIM mismatch: expected={OBS_DIM}, got={obs_dim} "
                f"=> SKIP EPISODE"
            )
            continue


    #######################################################################
    # INIT AGENT + REPLAY ON FIRST EP
    #######################################################################
    if agent is None:

        replay = ReplayBuffer(
            obs_dim=OBS_DIM,
            max_size=BUFFER_SIZE,
            n_step=1,
            gamma=0.995,
            alpha=0.6,
            beta=0.4,
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
            batch_size=BATCH_SIZE,
        )

        logger.info("[INIT] SACAgent + ReplayBuffer Initialized.")


    #######################################################################
    # EPISODE START
    #######################################################################
    replay.new_episode()

    ep_ret = 0.0
    done_reason = "none"
    com_drift_sum = 0.0

    energy_hist = []
    tqdm_bar = tqdm(range(max_steps), desc=f"[EP {ep}]", ncols=120)


    for step in tqdm_bar:

        ###############################################################
        # 1) ACTOR → per-atom 3D action
        ###############################################################
        action_arr = np.zeros((N_atom, ACT_DIM), np.float32)

        for i in range(N_atom):
            action_arr[i] = agent.act(obs[i])   # returns (3,)

        ###############################################################
        # 2) ENV STEP
        ###############################################################
        next_obs, reward, done, done_reason, Etot, Fmax = env.step(action_arr)
        reward = reward.astype(np.float32)

        ###############################################################
        # 3) Logging variables
        ###############################################################
        Fnorm = np.linalg.norm(env.forces, axis=1)
        E_avg = float(Etot) / N_atom

        energy_hist.append(E_avg)
        Emean_hist = float(np.mean(energy_hist))
        Estd_hist = float(np.std(energy_hist))

        # COM drift
        com_now = env.atoms.positions.mean(axis=0)
        COM_step = float(np.linalg.norm(com_now - env.com_prev))
        com_drift_sum += COM_step

        disp_mag = np.linalg.norm(env.disp_last, axis=1)


        ###############################################################
        # 4) STORE to ReplayBuffer (per atom)
        ###############################################################
        for i in range(N_atom):
            replay.store(obs[i], action_arr[i], reward[i], next_obs[i], done)


        ###############################################################
        # 5) Save trajectory
        ###############################################################
        env.atoms.write(traj_xyz, append=True)

        with open(energy_log, "a") as f:
            f.write(f"{step} {Etot:.8f} {E_avg:.8f}\n")


        ###############################################################
        # 6) Step Logging
        ###############################################################
        logger.info(
            f"[EP {ep}][STEP {step}] "
            f"N={N_atom} | "
            f"Fmax={Fmax:.3e}, Fmean={float(np.mean(Fnorm)):.3e} | "
            f"E={Etot:.4f}, E/atom={E_avg:.6f}, mean_hist={Emean_hist:.6f} ± {Estd_hist:.6f} | "
            f"COM_step={COM_step:.6f}, COM_sum={com_drift_sum:.6f} | "
            f"Disp_mean={float(np.mean(disp_mag)):.3e}, Disp_max={float(np.max(disp_mag)):.3e} | "
            f"rew_mean={float(np.mean(reward)):.6f}, rmax={float(np.max(reward)):.6f} | "
            f"alpha={float(agent.alpha):.5f} | "
            f"buffer={len(replay):,}"
        )

        ep_ret += float(np.mean(reward))
        obs = next_obs

        if done:
            logger.info(f"[EP {ep}] DONE at step={step}, reason={done_reason}")
            break



    ###############################################################
    # EPISODE END
    ###############################################################
    logger.info(f"[EP {ep}] return={ep_ret:.6f}")

    if done_reason in ["com", "bond"]:
        replay.end_episode(keep=False)
        logger.info(f"[EP {ep}] BAD EPISODE → transitions dropped.")
    else:
        replay.end_episode(keep=True)

    logger.info(f"[EP {ep}] Replay size = {len(replay):,}")



    ###############################################################
    # SAC UPDATE
    ###############################################################
    if len(replay) > agent.batch_size:
        update_info = agent.update()

        policy_loss = update_info["policy_loss"]
        q1_loss     = update_info["q1_loss"]
        q2_loss     = update_info["q2_loss"]
        v_loss      = update_info["v_loss"]
        alpha_loss  = update_info["alpha_loss"]
        alpha_val   = update_info["alpha"]

        # policy_loss가 None이면 숫자로 출력하면 crash → 방어적으로 처리
        policy_str = f"{policy_loss:.6f}" if policy_loss is not None else "None"

        logger.info(
            "[UPDATE] "
            f"policy={policy_str} | "
            f"q1={q1_loss:.6f} | "
            f"q2={q2_loss:.6f} | "
            f"v={v_loss:.6f} | "
            f"alpha={alpha_val:.5f}"
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

logger.info(
    f"======= TRAINING COMPLETE "
    f"(elapsed={(time.time()-global_start)/3600:.3f} hr) ======="
)

print("== TRAINING DONE ==")
