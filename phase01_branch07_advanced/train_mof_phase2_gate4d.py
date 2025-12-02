#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################
# train_mof_phase2_gate4d.py  (Config-based)
# Phase 2: QMOF + Small Perturb + Warm-up
# - YAML(configs/train_phase2.yaml) → Phase2Config
# - 4D per-atom action: (gate, dx, dy, dz)
# - gate ∈ [-1,1] → Env에서 [0,1]로 맵핑 후 disp 스케일링
# - Load Phase1 SAC checkpoint (옵션, CLI 인자)
# - New ReplayBuffer + Warm-up
# - Reset-time random perturb (sigma_min → sigma_max, max_perturb)
# - cmax curriculum: cmax_min → cmax_max (ep_sch_start → ep_sch_end)
# - Mode-based smoothing 사용 여부 / 모드 타입 등도 config 연동
# - Full logging (RotatingFileHandler) + tqdm 진행률 표시
##############################################

import os
import time
import argparse
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
from utils.config_utils import load_config, Phase2Config

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
# 중복 추가 방지
if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    logger.addHandler(log_handler)


##############################################
# CIF SAMPLING (QMOF pool)
##############################################
def sample_cif(pool_dir: str) -> str:
    cifs = [
        os.path.join(pool_dir, f)
        for f in os.listdir(pool_dir)
        if f.endswith(".cif")
    ]
    if not cifs:
        raise RuntimeError(f"[CIF] No .cif files found in {pool_dir}")
    return str(np.random.choice(cifs))


##############################################
# SCHEDULE FUNCTIONS (sigma, cmax)
##############################################
def get_perturb_sigma(ep: int, cfg: Phase2Config) -> float:
    """
    ep=0에서 env.sigma_min,
    ep >= (train.epochs/2) 에서 env.sigma_max.
    그 사이에서는 선형 증가.
    """
    total_ep = max(1, cfg.train.epochs)
    t = min(ep / (total_ep / 2.0), 1.0)
    return cfg.env.sigma_min + (cfg.env.sigma_max - cfg.env.sigma_min) * t


def get_cmax_ep(ep: int, cfg: Phase2Config) -> float:
    """
    Episode index(ep)에 따라 cmax를 선형 증가시키는 함수.
    - ep <= cmax_sch_start_ep : cmax_min
    - ep >= cmax_sch_end_ep   : cmax_max
    - 그 사이 : 선형 보간
    """
    env_cfg = cfg.env
    if ep <= env_cfg.cmax_sch_start_ep:
        return env_cfg.cmax_min
    if ep >= env_cfg.cmax_sch_end_ep:
        return env_cfg.cmax_max

    t = (ep - env_cfg.cmax_sch_start_ep) / float(
        env_cfg.cmax_sch_end_ep - env_cfg.cmax_sch_start_ep
    )
    return env_cfg.cmax_min + t * (env_cfg.cmax_max - env_cfg.cmax_min)


##############################################
# CHECKPOINT I/O
##############################################
def save_checkpoint(ep: int, agent: SACAgent, tag: str = "phase2") -> None:
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


def load_phase1_checkpoint(agent: SACAgent, ckpt_path: str) -> None:
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
# MAIN TRAIN LOOP (CONFIG-DRIVEN)
##############################################
def main():
    # ---------------------------
    # CLI Arguments
    # ---------------------------
    parser = argparse.ArgumentParser(
        description="MACS-MOF Phase2 Training (4D gate + disp, YAML config 기반)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_phase2.yaml",
        help="Phase2 YAML config path (default: configs/train_phase2.yaml)",
    )
    parser.add_argument(
        "--use-phase1-ckpt",
        action="store_true",
        help="If set, load Phase1 checkpoint before Phase2 training.",
    )
    parser.add_argument(
        "--phase1-ckpt",
        type=str,
        default="../phase02_qmof_branch03_terminate_reward_20251201/checkpoints_phase2/ckpt_ep1500_final.pt",
        help="Path to Phase1 checkpoint (.pt) (default: previous branch path).",
    )
    args = parser.parse_args()

    # ---------------------------
    # Load Config
    # ---------------------------
    cfg = load_config(args.config)
    train_cfg = cfg.train
    env_cfg = cfg.env
    sac_cfg = cfg.sac
    replay_cfg = cfg.replay
    modes_cfg = cfg.modes
    # bfgs_cfg, bc_cfg 등은 현재 이 스크립트에선 직접 사용 X (향후 확장용)

    logger.info(
        "[MAIN] Using config: %s | epochs=%d, buffer_size=%d, batch_size=%d",
        os.path.abspath(args.config),
        train_cfg.epochs,
        train_cfg.buffer_size,
        train_cfg.batch_size,
    )

    # ---------------------------
    # MACE Surrogate (from config)
    # ---------------------------
    if not train_cfg.mace_model_paths:
        raise ValueError(
            "[MACE] train.mace_model_paths 가 비어 있습니다. "
            "YAML 에 모델 경로를 지정해주세요."
        )

    logger.info(
        "[MACE] model_paths=%s, head=%s, device=%s",
        list(train_cfg.mace_model_paths),
        train_cfg.mace_head,
        train_cfg.device,
    )

    calc = MACECalculator(
        model_paths=list(train_cfg.mace_model_paths),
        head=train_cfg.mace_head,
        device=train_cfg.device,
        default_dtype="float32",
    )

    # ---------------------------
    # Basic Training Constants (from config)
    # ---------------------------
    EPOCHS = train_cfg.epochs
    BASE_STEPS = train_cfg.base_steps
    FINAL_STEPS = train_cfg.final_steps
    HORIZON_SCH = train_cfg.horizon_sch

    FMAX_THRESH = train_cfg.fmax_thresh
    BUFFER_SIZE = train_cfg.buffer_size
    BATCH_SIZE = train_cfg.batch_size
    CHECKPOINT_INTERVAL = train_cfg.checkpoint_interval
    WARMUP_TRANSITIONS = train_cfg.warmup_transitions

    # pool_dir / device / etc
    POOL_DIR = train_cfg.pool_dir

    # Mode-based smoothing: env.use_mode_basis + modes.use_mode_basis 함께 고려
    use_mode_smoothing = bool(env_cfg.use_mode_basis or modes_cfg.use_mode_basis)
    if env_cfg.use_mode_basis != modes_cfg.use_mode_basis:
        logger.warning(
            "[CONFIG] env.use_mode_basis(%s) != modes.use_mode_basis(%s) → "
            "use_mode_smoothing=%s (OR 기준)",
            env_cfg.use_mode_basis,
            modes_cfg.use_mode_basis,
            use_mode_smoothing,
        )

    MODE_TYPE = modes_cfg.type
    MODE_NUM_MODES = modes_cfg.num_modes
    # Laplacian cutoff 은 아직 config 에 없음 → 기존 값 유지
    MODE_EIG_CUTOFF = 4.0

    logger.info(
        "[CONFIG] Phase2 Summary | "
        "epochs=%d, base_steps=%d, final_steps=%d, horizon_sch=%d, "
        "fmax_thresh=%.4f, buffer_size=%d, batch_size=%d, warmup=%d",
        EPOCHS,
        BASE_STEPS,
        FINAL_STEPS,
        HORIZON_SCH,
        FMAX_THRESH,
        BUFFER_SIZE,
        BATCH_SIZE,
        WARMUP_TRANSITIONS,
    )
    logger.info(
        "[CONFIG] Env | k=%d, cmax=%.3f→%.3f (ep %d→%d), "
        "perturb_sigma=%.3f→%.3f, max_perturb=%.3f, "
        "time_penalty=%.4f, fail_penalty=%.3f, use_mode_basis(env)=%s",
        env_cfg.k_neighbors,
        env_cfg.cmax_min,
        env_cfg.cmax_max,
        env_cfg.cmax_sch_start_ep,
        env_cfg.cmax_sch_end_ep,
        env_cfg.sigma_min,
        env_cfg.sigma_max,
        env_cfg.max_perturb,
        env_cfg.time_penalty,
        env_cfg.fail_penalty,
        env_cfg.use_mode_basis,
    )
    logger.info(
        "[CONFIG] Modes | use_mode_basis(modes)=%s, type=%s, "
        "num_modes=%d, eig_cutoff=%.2f",
        modes_cfg.use_mode_basis,
        MODE_TYPE,
        MODE_NUM_MODES,
        MODE_EIG_CUTOFF,
    )
    logger.info(
        "[CONFIG] SAC | lr=%.6f, gamma=%.4f, tau=%.4f, target_entropy=%.3f, "
        "use_bc_loss=%s, bc_lambda=%.3f, bc_batch_ratio=%.3f",
        sac_cfg.lr,
        sac_cfg.gamma,
        sac_cfg.tau,
        sac_cfg.target_entropy,
        sac_cfg.use_bc_loss,
        sac_cfg.bc_lambda,
        sac_cfg.bc_batch_ratio,
    )
    logger.info(
        "[CONFIG] Replay | max_size=%d, log_interval=%d, "
        "use_expert_replay_seed=%s, expert_replay_path=%s",
        replay_cfg.max_size,
        replay_cfg.log_interval,
        replay_cfg.use_expert_replay_seed,
        str(replay_cfg.expert_replay_path),
    )

    global_start = time.time()

    ##############################################
    # GLOBALS
    ##############################################
    OBS_DIM = None
    ACT_DIM = 4   # 4D per-atom action: [gate, dx, dy, dz]
    replay = None
    agent = None

    logger.info(f"[MACS-MOF Phase2] Training start (EPOCHS={EPOCHS})")

    # ==========================================================
    # MAIN EPISODE LOOP
    # ==========================================================
    for ep in range(EPOCHS):

        logger.info("\n" + "=" * 80)
        logger.info(f"[EP {ep}] START")

        ##################################
        # Curriculum Horizon → max_steps
        ##################################
        ratio = min(ep / HORIZON_SCH, 1.0)
        max_steps = int(BASE_STEPS + (FINAL_STEPS - BASE_STEPS) * ratio)
        logger.info(f"[EP {ep}] max_steps = {max_steps}")

        ##################################
        # Perturb sigma (Phase2 핵심, config 기반)
        ##################################
        sigma_ep = get_perturb_sigma(ep, cfg)
        logger.info(
            f"[EP {ep}] perturb_sigma = {sigma_ep:.4f} Å, "
            f"max_perturb={env_cfg.max_perturb:.3f} Å "
            f"(random_perturb={env_cfg.random_perturb})"
        )

        ##################################
        # cmax curriculum (config 기반)
        ##################################
        cmax_ep = get_cmax_ep(ep, cfg)
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
        cif = sample_cif(POOL_DIR)
        atoms = read(cif)
        atoms.calc = calc

        cif_basename = os.path.splitext(os.path.basename(cif))[0]
        env = MOFEnv(
            atoms_loader=lambda: atoms,
            k_neighbors=env_cfg.k_neighbors,
            fmax_threshold=FMAX_THRESH,
            max_steps=max_steps,
            cmax=cmax_ep,          # EP별 cmax 적용
            # Phase2: perturb 옵션 (config 기반)
            random_perturb=env_cfg.random_perturb,
            perturb_sigma=sigma_ep,
            max_perturb=env_cfg.max_perturb,
            # 종료/시간 관련 하이퍼파라미터
            terminal_bonus_base=env_cfg.terminal_bonus_base,
            time_penalty=env_cfg.time_penalty,
            fail_penalty=env_cfg.fail_penalty,
            # Mode-based smoothing (4A/4B, graph_eig)
            use_mode_smoothing=use_mode_smoothing,
            mode_type=MODE_TYPE,
            mode_num_modes=MODE_NUM_MODES,
            mode_eig_cutoff=MODE_EIG_CUTOFF,
            mode_id=cif_basename,
            # ---------- 고급 패널티 / 로컬 프레임 (1A / 1B / 3A / 3B) ----------
            use_force_increase_penalty=env_cfg.use_force_increase_penalty,
            lambda_force_up=env_cfg.lambda_force_up,
            use_fd_direction_penalty=env_cfg.use_fd_direction_penalty,
            lambda_fd_penalty=env_cfg.lambda_fd_penalty,
            use_bond_projection=env_cfg.use_bond_projection,
            use_local_frame=env_cfg.use_local_frame,
            radial_scale=env_cfg.radial_scale,
            tangent_scale=env_cfg.tangent_scale,
        )

        obs = env.reset()
        logger.info(f"[EP {ep}] CIF loaded: {cif}")

        # Mode basis 상태 로그 (켰을 때만)
        if env.use_mode_smoothing:
            logger.info(
                "[EP %d] Mode smoothing enabled: type=%s, num_modes=%d, basis_shape=%s",
                ep,
                env.mode_type,
                env.mode_num_modes,
                "None" if env.mode_U is None else str(env.mode_U.shape),
            )

        N_atom = env.N
        obs_dim = obs.shape[1]   # per-atom feature dim

        ##################################
        # EP0: Initialize Replay + Agent (+ Optional Phase1 ckpt)
        ##################################
        if ep == 0:
            OBS_DIM = obs_dim
            logger.info(f"[INIT] OBS_DIM={OBS_DIM}, ACT_DIM={ACT_DIM} (per-atom)")

            # ReplayBuffer (config 기반)
            replay = ReplayBuffer(
                obs_dim=OBS_DIM,
                max_size=replay_cfg.max_size,
                log_interval=replay_cfg.log_interval,
            )

            # SACAgent (config 기반)
            agent = SACAgent(
                obs_dim=OBS_DIM,
                act_dim=ACT_DIM,
                replay_buffer=replay,
                device=train_cfg.device,
                lr=sac_cfg.lr,
                gamma=sac_cfg.gamma,
                tau=sac_cfg.tau,
                batch_size=BATCH_SIZE,
                target_entropy=sac_cfg.target_entropy,  # ★ config 연결
            )
            logger.info("[INIT] Agent + ReplayBuffer allocated (per-atom).")

            # Phase1 checkpoint 로드 (옵션: CLI 인자)
            if args.use_phase1_ckpt and args.phase1_ckpt and os.path.exists(args.phase1_ckpt):
                try:
                    load_phase1_checkpoint(agent, args.phase1_ckpt)
                except Exception as e:
                    logger.warning(
                        "[LOAD] Failed to load Phase1 checkpoint (%s): %s",
                        args.phase1_ckpt,
                        repr(e),
                    )
                    logger.warning("[LOAD] Continue Phase2 from scratch (4D action).")
            else:
                logger.info(
                    "[LOAD] Skip Phase1 checkpoint. "
                    "Training Phase2 from scratch (4D action)."
                )

            # (선택) expert replay seeding 은 utils/expert_replay 에서 별도 스크립트로 처리
            # cfg.replay.use_expert_replay_seed / expert_replay_path 는
            # 필요 시 추후 확장해서 여기서 불러 사용할 수 있음.

        ##################################
        # EPISODE LOOP
        ##################################
        ep_ret = 0.0

        for step in tqdm(range(max_steps), desc=f"[EP {ep}]", ncols=120):

            ########################
            # ACTION (per-atom, 4D)
            ########################
            obs_tensor = obs  # (N_atom, obs_dim)

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
            # Env.step() 내:
            #   gate_raw = action[:, 0:1]
            #   disp_raw = action[:, 1:4]
            #   gate = (gate_raw + 1.0) / 2.0  # [-1,1] → [0,1]
            #   disp = 0.003 * gate * disp_raw * (scale / cmax)
            #   (옵션) mode smoothing → atoms.positions += disp
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
            if len(replay) > max(agent.batch_size, WARMUP_TRANSITIONS):
                losses = agent.update()
                logger.info(f"[EP {ep}][STEP {step}] losses={losses}")
            else:
                # warm-up 구간에서도 tqdm 외에 간헐적으로 사이즈 로깅
                if len(replay) % 10_000 == 0 and len(replay) > 0:
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
            r_mean = env.last_reward_mean

            logger.info(
                f"[EP {ep}][STEP {step}] "
                f"N={N_atom} | "
                f"Favg={np.mean(f_norm):.6f} Fmax={np.max(f_norm):.6f} | "
                f"r_mean={r_mean:.6f} (rf={rf_mean:.6f}, "
                f"COM=-{com_pen:.6f}, bond=-{bond_pen:.6f}, "
                f"time=-{t_pen:.6f}, fail=-{f_pen:.6f}, bonus=+{t_bonus:.6f}) | "
                f"gate_mean={mean_gate:.6f} | "
                f"replay={len(replay):,} | "
                f"alpha={float(agent.alpha):.5f}"
            )

            ep_ret += float(r_mean)
            obs = next_obs

            if done:
                logger.info(f"[EP {ep}] terminated early at step={step}")
                break

        ##################################
        # EP END
        ##################################
        ##################################
        # EP END
        ##################################
        logger.info(f"[EP {ep}] return={ep_ret:.6f}")
        logger.info(f"[EP {ep}] replay_size={len(replay):,}")

        # 주기적 체크포인트 저장
        if (ep + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(ep, agent, tag="phase2")

    ##############################################
    # TRAINING DONE
    ##############################################
    total_time = time.time() - global_start
    logger.info(
        "[TRAIN DONE] epochs=%d, total_time=%.1f s (%.2f h)",
        EPOCHS,
        total_time,
        total_time / 3600.0,
    )

    # 마지막 에포크 기준 최종 체크포인트 한 번 더 저장
    try:
        save_checkpoint(EPOCHS - 1, agent, tag="phase2_final")
    except Exception as e:
        logger.warning("[CHECKPOINT] Failed to save final checkpoint: %s", repr(e))


if __name__ == "__main__":
    main()
