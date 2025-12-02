#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################
# train_mof_phase2_gate4d.py
# Phase 2: QMOF + Small Perturb + Warm-up
# - 4D per-atom action: (gate, dx, dy, dz)
# - gate ∈ [-1,1] → Env에서 [0,1]로 맵핑 후 disp 스케일링
# - Load Phase1 SAC checkpoint (옵션)
# - New ReplayBuffer + Warm-up
# - Reset-time random perturb (sigma 0.02~0.08 Å, max 0.3 Å)
# - cmax curriculum: 0.05 → 0.40 over episodes
# - Reward component logging
#   (force / COM / bond / time / fail / bonus / force_inc)
# - Logging + tqdm 진행률 표시 (고수준 로그: 일정 스텝 간격 요약)
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
logger.propagate = False  # 상위 로거로 중복 전파 방지


##############################################
# CHECKPOINT I/O
##############################################
PHASE1_CKPT = "../phase02_qmof_branch03_terminate_reward_20251201/checkpoints_phase2/ckpt_ep1500_final.pt"

# Phase1 checkpoint 사용 여부 (True: 로드, False: 스킵 후 Phase2를 처음부터 학습)
USE_PHASE1_CKPT = False


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
POOL_DIR = "../mofs/train_pool_valid"


def sample_cif():
    cifs = [
        os.path.join(POOL_DIR, f)
        for f in os.listdir(POOL_DIR)
        if f.endswith(".cif")
    ]
    if not cifs:
        raise RuntimeError(f"[CIF] No .cif files found in {POOL_DIR}")
    return np.random.choice(cifs)


##############################################
# MACE Surrogate
##############################################
calc = MACECalculator(
    model_paths=["../mofs_v2.model"],
    head="pbe_d3",
    device="cuda",
    default_dtype="float32",
)


##############################################
# CONFIG (Phase2)
##############################################
EPOCHS = 3000          # Phase2 길이
BASE_STEPS = 200
FINAL_STEPS = 600
HORIZON_SCH = 1000      # step 수 curriculum

FMAX_THRESH = 0.05
BUFFER_SIZE = 5_000_000
BATCH_SIZE = 256

CHECKPOINT_INTERVAL = 10
WARMUP_TRANSITIONS = 20_000   # Phase2에서도 새로 warm-up

# Perturb 스케줄 범위
SIGMA_MIN = 0.02   # Å
SIGMA_MAX = 0.08   # Å
MAX_PERTURB = 0.30  # Å (헤더 주석대로 max 0.3 Å 적용)

# ---------- cmax 커리큘럼 설정 ----------
CMAX_MIN = 0.03         # 초기 EP에서 사용할 최소 cmax
CMAX_MAX = 0.40         # 후반 EP에서 사용할 최대 cmax
CMAX_SCH_START_EP = 1500       # cmax 증가 시작 EP
CMAX_SCH_END_EP = 2000         # cmax 증가 종료 EP

# ---------- Force 증가 패널티 하이퍼파라미터 (NEW) ----------
FORCE_INCREASE_SCALE = 3.0       # 비율 초과분에 곱해지는 스케일
FORCE_INCREASE_THRESHOLD = 0.03  # 5% 이상 증가 시만 벌점 (ratio > 1.05)

# ---------- LOGGING GRANULARITY (고수준 로그) ----------
LOG_STEP_INTERVAL = 20  # 몇 스텝마다 한 번씩 상세 로그를 찍을지 (예: 20)


def get_perturb_sigma(ep: int) -> float:
    """
    ep=0에서 SIGMA_MIN,
    ep=(EPOCHS/2) 이상에서 SIGMA_MAX
    그 사이에는 선형 증가.
    """
    t = min(ep / (EPOCHS / 2.0), 1.0)
    return SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * t


def get_cmax_ep(ep: int) -> float:
    """
    Episode index(ep)에 따라 cmax를 선형 증가시키는 함수.
    - ep <= CMAX_SCH_START_EP : CMAX_MIN
    - ep >= CMAX_SCH_END_EP   : CMAX_MAX
    - 그 사이 : 선형 보간
    """
    if ep <= CMAX_SCH_START_EP:
        return CMAX_MIN
    if ep >= CMAX_SCH_END_EP:
        return CMAX_MAX

    t = (ep - CMAX_SCH_START_EP) / float(CMAX_SCH_END_EP - CMAX_SCH_START_EP)
    return CMAX_MIN + t * (CMAX_MAX - CMAX_MIN)


##############################################
# TRAIN LOOP
##############################################
logger.info(f"[MACS-MOF Phase2] Training start (EPOCHS={EPOCHS})")
logger.info(
    f"[CONFIG] BATCH_SIZE={BATCH_SIZE}, BUFFER_SIZE={BUFFER_SIZE:,}, "
    f"WARMUP_TRANSITIONS={WARMUP_TRANSITIONS:,}, "
    f"SIGMA_MIN={SIGMA_MIN:.3f}, SIGMA_MAX={SIGMA_MAX:.3f}, "
    f"MAX_PERTURB={MAX_PERTURB:.3f}"
)
logger.info(
    f"[CONFIG] CMAX_MIN={CMAX_MIN:.3f}, CMAX_MAX={CMAX_MAX:.3f}, "
    f"CMAX_SCH_START_EP={CMAX_SCH_START_EP}, CMAX_SCH_END_EP={CMAX_SCH_END_EP}"
)
logger.info(
    f"[CONFIG] FORCE_INCREASE_SCALE={FORCE_INCREASE_SCALE:.3f}, "
    f"FORCE_INCREASE_THRESHOLD={FORCE_INCREASE_THRESHOLD:.3f}"
)
logger.info(
    f"[CONFIG] LOG_STEP_INTERVAL={LOG_STEP_INTERVAL} (steps between detailed logs)"
)

global_start = time.time()

##############################################
# GLOBALS
##############################################
OBS_DIM = None
ACT_DIM = 4   # 4D per-atom action: [gate, dx, dy, dz]
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
    # Perturb sigma (Phase2 핵심)
    ##################################
    sigma_ep = get_perturb_sigma(ep)
    logger.info(
        f"[EP {ep}] perturb_sigma = {sigma_ep:.4f} Å, "
        f"max_perturb={MAX_PERTURB:.3f} Å"
    )

    ##################################
    # cmax curriculum (NEW)
    ##################################
    cmax_ep = get_cmax_ep(ep)
    logger.info(f"[EP {ep}] cmax = {cmax_ep:.4f}")

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
        cmax=cmax_ep,          # EP별 cmax 적용
        # Phase2: perturb 옵션 켜기
        random_perturb=True,
        perturb_sigma=sigma_ep,
        max_perturb=MAX_PERTURB,
        # 종료/시간 관련 하이퍼파라미터
        terminal_bonus_base=10.0,  # 성공 시 남은 step 비율에 비례한 보너스
        time_penalty=0.05,         # 매 step -0.05 (튜닝 대상)
        fail_penalty=15.0,         # max_steps 도달 실패 패널티
        # Force 증가 패널티 (NEW)
        force_increase_scale=FORCE_INCREASE_SCALE,
        force_increase_threshold=FORCE_INCREASE_THRESHOLD,
    )

    obs = env.reset()
    logger.info(f"[EP {ep}] CIF loaded: {cif}")

    N_atom = env.N
    obs_dim = obs.shape[1]   # per-atom feature dim

    ##################################
    # EP0: Initialize Replay + Agent + (Optional) Load Phase1 ckpt
    ##################################
    if ep == 0:
        OBS_DIM = obs_dim
        logger.info(f"[INIT] OBS_DIM={OBS_DIM}, ACT_DIM={ACT_DIM} (per-atom)")

        replay = ReplayBuffer(
            obs_dim=OBS_DIM,
            max_size=BUFFER_SIZE,
        )

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
        logger.info("[INIT] Agent + ReplayBuffer allocated (per-atom).")

        # Phase1 checkpoint 로드 (옵션)
        if USE_PHASE1_CKPT and PHASE1_CKPT and os.path.exists(PHASE1_CKPT):
            try:
                load_phase1_checkpoint(agent, PHASE1_CKPT)
            except Exception as e:
                logger.warning(
                    f"[LOAD] Failed to load Phase1 checkpoint ({PHASE1_CKPT}): {e}"
                )
                logger.warning("[LOAD] Continue Phase2 from scratch (4D action).")
        else:
            logger.info(
                f"[LOAD] Skip Phase1 checkpoint. "
                f"Training Phase2 from scratch (ACT_DIM={ACT_DIM})."
            )

    ##################################
    # EPISODE
    ##################################
    ep_ret = 0.0
    ep_steps = 0

    for step in tqdm(range(max_steps), desc=f"[EP {ep}]", ncols=120):

        ########################
        # ACTION (per-atom, 4D)
        ########################
        obs_tensor = obs  # shape = (N_atom, obs_dim)

        action_list = []
        for i in range(N_atom):
            a = agent.act(obs_tensor[i])  # → (ACT_DIM,) = (4,)
            action_list.append(a)

        action_arr = np.stack(action_list, axis=0)  # (N_atom, 4)

        # ----------------------------------------
        # 0) gate / disp 분리 후 disp에만 mean 제거
        # ----------------------------------------
        gate_raw = action_arr[:, 0:1]   # (N, 1)
        disp_raw = action_arr[:, 1:4]   # (N, 3)

        mean_disp = disp_raw.mean(axis=0, keepdims=True)   # (1, 3)
        disp_centered = disp_raw - mean_disp               # (N, 3)

        # 다시 합치기: [gate, centered_disp]
        action_arr = np.concatenate([gate_raw, disp_centered], axis=1)  # (N, 4)

        # 모니터링용 gate 평균
        mean_gate = float(gate_raw.mean())

        ########################
        # STEP ENV
        ########################
        # ⚠️ Env 쪽 step()에서는 다음과 같이 해석:
        #   gate_raw = action[:, 0:1]
        #   disp_raw = action[:, 1:4]
        #   gate = (gate_raw + 1.0) / 2.0  # [-1,1] → [0,1]
        #   disp = 0.003 * gate * disp_raw * (scale / cmax)
        next_obs, reward, done = env.step(action_arr)

        ########################
        # STORE (per-atom)
        ########################
        next_reward = reward.astype(np.float32)

        for i in range(N_atom):
            replay.store(
                obs[i],            # (obs_dim,)
                action_arr[i],     # (4,)
                next_reward[i],    # scalar
                next_obs[i],       # (obs_dim,)
                done,
            )

        # ------------------------
        # SAC 업데이트 (Replay warm-up 적용)
        # ------------------------
        step_losses = None
        if len(replay) > max(agent.batch_size, WARMUP_TRANSITIONS):
            step_losses = agent.update()
        else:
            # warm-up 구간은 매우 드물게만 로깅
            if len(replay) != 0 and len(replay) % 50_000 == 0:
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

        # Reward component breakdown from env
        rf_mean = env.last_r_f_mean
        com_pen = env.last_com_penalty
        bond_pen = env.last_bond_penalty
        t_pen = env.last_time_penalty
        f_pen = env.last_fail_penalty
        t_bonus = env.last_terminal_bonus
        f_inc_pen = env.last_force_increase_penalty
        r_mean = env.last_reward_mean

        ep_ret += float(r_mean)
        ep_steps = step + 1
        obs = next_obs

        # --------- 고수준 STEP LOGGING (간격 기반) ----------
        if (
            (step % LOG_STEP_INTERVAL == 0)
            or done
            or (step == max_steps - 1)
        ):
            # losses 요약 문자열 (있을 때만)
            if step_losses is not None and step_losses["policy_loss"] is not None:
                loss_str = (
                    f" | losses="
                    f"pi={step_losses['policy_loss']:.4f}, "
                    f"q1={step_losses['q1_loss']:.4f}, "
                    f"q2={step_losses['q2_loss']:.4f}, "
                    f"v={step_losses['v_loss']:.4f}, "
                    f"alpha={step_losses['alpha_loss']:.4f}"
                )
            elif step_losses is not None:
                # warm-up 직후 등 policy_loss=None인 경우
                loss_str = (
                    f" | losses="
                    f"q1={step_losses['q1_loss']:.4f}, "
                    f"q2={step_losses['q2_loss']:.4f}, "
                    f"v={step_losses['v_loss']:.4f}, "
                    f"alpha={step_losses['alpha_loss']:.4f}"
                )
            else:
                loss_str = ""

            logger.info(
                f"[EP {ep}][STEP {step}] "
                f"N={N_atom} | "
                f"Favg={np.mean(f_norm):.6f} Fmax={np.max(f_norm):.6f} | "
                f"r_mean={r_mean:.6f} (rf={rf_mean:.6f}, "
                f"COM=-{com_pen:.6f}, bond=-{bond_pen:.6f}, "
                f"time=-{t_pen:.6f}, fail=-{f_pen:.6f}, "
                f"force_inc=-{f_inc_pen:.6f}, bonus=+{t_bonus:.6f}) | "
                f"gate_mean={mean_gate:.6f} | "
                f"replay={len(replay):,} | "
                f"alpha={float(agent.alpha):.5f}"
                f"{loss_str}"
            )

        if done:
            logger.info(f"[EP {ep}] terminated early at step={step}")
            break

    ##################################
    # EP END (에피소드 요약 로그)
    ##################################
    # 마지막 스텝에서의 force 상태를 이용한 간단 요약
    final_favg = float(np.mean(np.linalg.norm(env.forces, axis=1)))
    final_fmax = float(np.max(np.linalg.norm(env.forces, axis=1)))

    logger.info(
        f"[EP {ep}] END | steps={ep_steps} "
        f"| return={ep_ret:.6f} "
        f"| Favg_final={final_favg:.6f} Fmax_final={final_fmax:.6f} "
        f"| replay_size={len(replay):,}"
    )

    if ep % CHECKPOINT_INTERVAL == 0 and ep > 0:
        save_checkpoint(ep, agent, tag="interval")


##############################################
# FINAL SAVE
##############################################
save_checkpoint(EPOCHS, agent, tag="final")

logger.info("[TRAIN DONE] (Phase2, 4D action)")
logger.info(f"wallclock={(time.time() - global_start)/3600:.3f} hr")

print("== Phase2 training finished (4D gate + disp) ==")
