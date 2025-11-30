#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################
# train_mof_phase2.py
# Phase 2: QMOF + (옵션) Small Perturb + Warm-up
# - Load Phase1 SAC checkpoint
# - New ReplayBuffer + Warm-up
# - Reset-time random perturb (ON/OFF + 스케줄 가능)
# - cmax도 EP 기반 스케줄로 조절 가능
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
    "train.log",
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
PHASE2_CKPT = "/home/yongsang/20251123_MOF_SAC/phase02_qmof_20251130/checkpoints_phase2/ckpt_ep1500_final.pt"

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

    logger.info(f"[LOAD] Loaded Phase2 checkpoint from {ckpt_path}")
    logger.info(f"[LOAD] log_alpha={ckpt['log_alpha']:.6f}")


##############################################
# CIF SAMPLING (QMOF pool)
##############################################
POOL_DIR = "mofs/train_pool_valid.hmof"


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
EPOCHS       = 1500
BASE_STEPS   = 300
FINAL_STEPS  = 1000
HORIZON_SCH  = 500

FMAX_THRESH  = 0.05
BUFFER_SIZE  = 5_000_000
BATCH_SIZE   = 256

CHECKPOINT_INTERVAL   = 10
WARMUP_TRANSITIONS    = 15_000   # ★ 요구사항: Warm-up 15000으로 감소


##############################################
# PERTURB / CMAX SCHEDULE CONFIG
##############################################

# ---------- Perturb 스케줄 ----------
# 기본은 perturb 끔. 나중에 HMOF/Phase3 등에서 켜고 싶으면
# PERTURB_ENABLE = True로 바꾸고 EP 구간/시그마만 조정하면 됨.
PERTURB_ENABLE     = False     # ★ 현재는 완전 OFF
PERTURB_START_EP   = 200       # 이 EP부터 perturb 서서히 켜기 시작 (ENABLE=True일 때)
PERTURB_END_EP     = 800       # 이 EP 이후로는 최댓값 유지
PERTURB_SIGMA_MIN  = 0.01      # 시작 시그마
PERTURB_SIGMA_MAX  = 0.05      # 마지막 시그마
MAX_PERTURB_MIN    = 0.10      # 시작 max_perturb
MAX_PERTURB_MAX    = 0.20      # 마지막 max_perturb


def get_perturb_params(ep: int):
    """
    EP 기반 perturb 설정 함수.

    Returns
    -------
    use_perturb : bool
    sigma_ep    : float
    max_pert_ep : float
    """
    # 완전 OFF 모드
    if not PERTURB_ENABLE:
        return False, 0.0, MAX_PERTURB_MIN

    # 아직 perturb 켜기 전 구간
    if ep < PERTURB_START_EP:
        return False, 0.0, MAX_PERTURB_MIN

    # 이미 최댓값 구간
    if ep >= PERTURB_END_EP:
        return True, PERTURB_SIGMA_MAX, MAX_PERTURB_MAX

    # 선형 보간
    t = (ep - PERTURB_START_EP) / float(PERTURB_END_EP - PERTURB_START_EP)
    sigma = PERTURB_SIGMA_MIN + (PERTURB_SIGMA_MAX - PERTURB_SIGMA_MIN) * t
    max_p = MAX_PERTURB_MIN + (MAX_PERTURB_MAX - MAX_PERTURB_MIN) * t
    return True, sigma, max_p


# ---------- cmax 스케줄 ----------
# 기본 동작은 cmax = 0.03 고정.
# 나중에 특정 EP 범위에서 0.03 → 0.05 이런 식으로 선형 증가시키고 싶으면
# CMAX_TARGET, CMAX_START_EP, CMAX_END_EP를 수정하면 됨.
CMAX_BASE      = 0.03    # 초기/기본 cmax
CMAX_TARGET    = 0.08    # 목표 cmax (지금은 BASE와 같아서 스케줄 비활성화 상태)
CMAX_START_EP  = 200       # 이 EP부터 스케줄 시작
CMAX_END_EP    = 1000       # 이 EP부터는 TARGET 유지 (BASE==TARGET이면 의미 없음)


def get_cmax(ep: int) -> float:
    """
    EP 기반 cmax 스케줄 함수.

    현재 설정(CMAX_BASE == CMAX_TARGET 또는 START>=END)이면
    항상 CMAX_BASE를 반환해서 "스케줄 없음" 모드로 동작.
    """
    # 스케줄 비활성화 조건
    if CMAX_BASE == CMAX_TARGET or CMAX_START_EP >= CMAX_END_EP:
        return CMAX_BASE

    # 스케줄 시작 전
    if ep <= CMAX_START_EP:
        return CMAX_BASE

    # 스케줄 종료 후
    if ep >= CMAX_END_EP:
        return CMAX_TARGET

    # 그 사이: 선형 보간
    t = (ep - CMAX_START_EP) / float(CMAX_END_EP - CMAX_START_EP)
    return CMAX_BASE + (CMAX_TARGET - CMAX_BASE) * t


##############################################
# TRAIN START
##############################################
logger.info(f"[MACS-MOF Phase2] Training start (EPOCHS={EPOCHS})")
logger.info(
    "[CONFIG] "
    f"BATCH_SIZE={BATCH_SIZE}, BUFFER_SIZE={BUFFER_SIZE:,}, "
    f"WARMUP_TRANSITIONS={WARMUP_TRANSITIONS:,}"
)
logger.info(
    "[CONFIG-PERTURB] "
    f"ENABLE={PERTURB_ENABLE}, "
    f"EP_RANGE=({PERTURB_START_EP}→{PERTURB_END_EP}), "
    f"SIGMA=({PERTURB_SIGMA_MIN:.4f}→{PERTURB_SIGMA_MAX:.4f}), "
    f"MAX_PERT=({MAX_PERTURB_MIN:.3f}→{MAX_PERTURB_MAX:.3f})"
)
logger.info(
    "[CONFIG-CMAX] "
    f"BASE={CMAX_BASE:.4f}, TARGET={CMAX_TARGET:.4f}, "
    f"EP_RANGE=({CMAX_START_EP}→{CMAX_END_EP})"
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
    # Curriculum Horizon
    ##################################
    ratio = min(ep / HORIZON_SCH, 1.0)
    max_steps = int(BASE_STEPS + (FINAL_STEPS - BASE_STEPS) * ratio)
    logger.info(f"[EP {ep}] max_steps = {max_steps}")

    ##################################
    # cmax / Perturb 스케줄
    ##################################
    cmax_ep = get_cmax(ep)
    use_perturb, sigma_ep, max_pert_ep = get_perturb_params(ep)

    logger.info(
        f"[EP {ep}] cmax={cmax_ep:.4f}, "
        f"perturb_enable={use_perturb}, "
        f"perturb_sigma={sigma_ep:.4f} Å, "
        f"max_perturb={max_pert_ep:.3f} Å"
    )

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
        cmax=cmax_ep,
        random_perturb=use_perturb,
        perturb_sigma=sigma_ep,
        max_perturb=max_pert_ep,
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
        load_phase1_checkpoint(agent, PHASE2_CKPT)

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
            # 10,000 단위로 warm-up 상태 로깅
            if len(replay) > 0 and len(replay) % 10_000 == 0:
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

