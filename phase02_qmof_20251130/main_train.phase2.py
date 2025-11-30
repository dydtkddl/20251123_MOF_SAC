#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################
# train_mof_phase2.py
# Phase 2: QMOF + Small Perturb + Warm-up
# - Load Phase1 SAC checkpoint
# - New ReplayBuffer + Warm-up
# - Reset-time random perturb (sigma 0.02~0.08 Å, max 0.3 Å)
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
# LOGGING SETUP (Phase2 전용 log 파일)
##############################################
log_handler = RotatingFileHandler(
    "train_phase2.log",
    maxBytes=20_000_000,
    backupCount=10,
)
log_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
))

logger = logging.getLogger("train_phase2")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)


##############################################
# CHECKPOINT I/O
##############################################
PHASE1_CKPT = "phase01_qmof_20251129/checkpoints/ckpt_ep1500_final.pt"#  "checkpoints/ckpt_ep1500_final.pt"  # <- 여기만 필요시 변경

def save_checkpoint(ep, agent, tag="phase2"):
    os.makedirs("checkpoints_phase2", exist_ok=True)
    ckpt = {
        "epoch": ep,
        "actor": agent.actor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "v": agent.v.state_dict(),
        "v_tgt": agent.v_tgt.state_dict(),
        "log_alpha": float(agent.log_alpha.detach().cpu()),
    }
    p = f"checkpoints_phase2/ckpt_ep{ep:04d}_{tag}.pt"
    torch.save(ckpt, p)
    logger.info(f"[CHECKPOINT] Saved => {p}")


def load_phase1_checkpoint(agent, ckpt_path):
    device = agent.device
    ckpt = torch.load(ckpt_path, map_location=device)

    agent.actor.load_state_dict(ckpt["actor"])
    agent.q1.load_state_dict(ckpt["q1"])
    agent.q2.load_state_dict(ckpt["q2"])
    agent.v.load_state_dict(ckpt["v"])
    agent.v_tgt.load_state_dict(ckpt["v_tgt"])

    # log_alpha 복원
    with torch.no_grad():
        agent.log_alpha.data.fill_(ckpt["log_alpha"])

    logger.info(f"[LOAD] Loaded Phase1 checkpoint from {ckpt_path}")
    logger.info(f"[LOAD] log_alpha={ckpt['log_alpha']:.6f}")


##############################################
# CIF SAMPLING (QMOF pool)
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
# CONFIG (Phase2)
##############################################
EPOCHS       = 1500          # Phase2는 조금 짧게 잡아도 됨 (원하면 조정)
BASE_STEPS   = 300
FINAL_STEPS  = 1000
HORIZON_SCH  = 500

FMAX_THRESH  = 0.05
BUFFER_SIZE  = 5_000_000
BATCH_SIZE   = 256

CHECKPOINT_INTERVAL   = 10
WARMUP_TRANSITIONS    = 50_000   # Phase2에서도 새로 warm-up

# Perturb 스케줄 범위
SIGMA_MIN = 0.02   # Å
SIGMA_MAX = 0.08   # Å
MAX_PERTURB = 0.20 # Å


def get_perturb_sigma(ep):
    """
    ep=0에서 SIGMA_MIN,
    ep=(EPOCHS/2) 이상에서 SIGMA_MAX
    그 사이에는 선형 증가.
    """
    t = min(ep / (EPOCHS / 2.0), 1.0)
    return SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * t


##############################################
# TRAIN START
##############################################
logger.info(f"[MACS-MOF Phase2] Training start (EPOCHS={EPOCHS})")
logger.info(
    f"[CONFIG] BATCH_SIZE={BATCH_SIZE}, BUFFER_SIZE={BUFFER_SIZE:,}, "
    f"WARMUP_TRANSITIONS={WARMUP_TRANSITIONS:,}, "
    f"SIGMA_MIN={SIGMA_MIN:.3f}, SIGMA_MAX={SIGMA_MAX:.3f}, "
    f"MAX_PERTURB={MAX_PERTURB:.3f}"
)
global_start = time.time()


##############################################
# GLOBALS
##############################################
OBS_DIM = None
ACT_DIM = 3   # ALWAYS 3 for per-atom action
replay = None
agent = None


for ep in range(EPOCHS):

    logger.info("\n" + "=" * 80)
    logger.info(f"[EP {ep}] START")

    ##################################
    # Curriculum Horizon (동일)
    ##################################
    ratio = min(ep / HORIZON_SCH, 1.0)
    max_steps = int(BASE_STEPS + (FINAL_STEPS - BASE_STEPS) * ratio)
    logger.info(f"[EP {ep}] max_steps = {max_steps}")

    ##################################
    # Perturb sigma (Phase2 핵심)
    ##################################
    sigma_ep = get_perturb_sigma(ep)
    logger.info(f"[EP {ep}] perturb_sigma = {sigma_ep:.4f} Å, max_perturb={MAX_PERTURB:.3f} Å")

    ##################################
    # Snapshot folders
    ##################################
    snap_dir = f"snapshots_phase2/EP{ep:04d}"
    os.makedirs(snap_dir, exist_ok=True)

    traj_path = os.path.join(snap_dir, "traj.xyz")
    en_path = os.path.join(snap_dir, "energies.txt")

    if os.path.exists(traj_path):
        os.remove(traj_path)
    if os.path.exists(en_path):
        os.remove(en_path)

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
        # Phase2: perturb 옵션 켜기
        random_perturb=True,
        perturb_sigma=sigma_ep,
        max_perturb=MAX_PERTURB,
    )

    obs = env.reset()
    logger.info(f"[EP {ep}] CIF loaded: {cif}")

    N_atom = env.N
    obs_dim = obs.shape[1]   # per-atom feature dim

    ##################################
    # EP0: Initialize Replay + Agent + Load Phase1 ckpt
    ##################################
    if ep == 0:
        OBS_DIM = obs_dim
        logger.info(f"[INIT] OBS_DIM={OBS_DIM}, ACT_DIM=3 (per-atom)")

        replay = ReplayBuffer(
            obs_dim=OBS_DIM,
            max_size=BUFFER_SIZE
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
        )
        logger.info("[INIT] Agent + ReplayBuffer allocated (per-atom).")

        # Phase1 checkpoint 로드
        load_phase1_checkpoint(agent, PHASE1_CKPT)


    ##################################
    # EPISODE
    ##################################
    ep_ret = 0.0

    for step in tqdm(range(max_steps), desc=f"[EP {ep}]", ncols=120):

        ########################
        # ACTION (per-atom)
        ########################
        obs_tensor = obs  # shape = (N_atom, obs_dim)

        action_list = []
        for i in range(N_atom):
            a = agent.act(obs_tensor[i])  # → (3,)
            action_list.append(a)

        action_arr = np.stack(action_list, axis=0)  # (N_atom, 3)

        ########################
        # STEP ENV
        ########################
        next_obs, reward, done = env.step(action_arr)
        # reward = per-atom reward shape (N_atom,)

        ########################
        # STORE (per-atom)
        ########################
        next_reward = reward.astype(np.float32)

        for i in range(N_atom):
            replay.store(
                obs[i],            # (obs_dim,)
                action_arr[i],     # (3,)
                next_reward[i],    # scalar
                next_obs[i],       # (obs_dim,)
                done,
            )

        # ------------------------
        # SAC 업데이트 (Replay warm-up 적용)
        # ------------------------
        if len(replay) > max(agent.batch_size, WARMUP_TRANSITIONS):
            losses = agent.update()
            logger.info(f"[EP {ep}][STEP {step}] losses={losses}")
        else:
            if len(replay) % 10_000 == 0:
                logger.info(
                    f"[WARMUP] replay={len(replay):,} / {WARMUP_TRANSITIONS:,} "
                    f"(batch={agent.batch_size})"
                )

        ########################
        # LOG & SAVE TRAJECTORY
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
            f"rew_mean={float(np.mean(next_reward)):.6f} | "
            f"replay={len(replay):,} | "
            f"alpha={float(agent.alpha):.5f}"
        )

        ep_ret += float(np.mean(next_reward))
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

logger.info("[TRAIN DONE] (Phase2)")
logger.info(f"wallclock={(time.time() - global_start)/3600:.3f} hr")

print("== Phase2 training finished ==")



