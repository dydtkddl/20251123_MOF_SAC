# main_train.py
###############################################################
# MOF Multi-Agent SAC Training Script
#
# - 여러 MOF CIF를 랜덤으로 샘플링하면서
#   per-atom Multi-Agent SAC (Parameter Sharing + CTDE Twin Q) 학습
# - env.MOFEnv / sac.MultiAgentSAC / utils.MultiAgentReplayBuffer 연동
# - logging + RotatingFileHandler + tqdm 진행률 표시
###############################################################

import os
import sys
import time
import glob
import random
import argparse
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Tuple

import numpy as np
from tqdm.auto import tqdm

import torch
from ase.io import read
from mace.calculators import MACECalculator

from env.mof_env import MOFEnv
from sac.agent import MultiAgentSAC
from utils.replay_buffer import MultiAgentReplayBuffer


# ============================================================
# Utility: seeding
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Utility: logger 설정
# ============================================================
def setup_logger(log_dir: str, log_level=logging.INFO) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    logger = logging.getLogger("train")
    logger.setLevel(log_level)
    logger.propagate = False  # 중복 출력 방지

    # 기존 핸들러 제거
    if logger.handlers:
        logger.handlers = []

    # 콘솔 핸들러
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(ch_formatter)

    # 파일 핸들러 (회전 로그)
    fh = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(log_level)
    fh_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fh_formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info("============================================================")
    logger.info("[LOGGER] Initialized logger at %s", log_path)
    logger.info("============================================================")

    return logger


# ============================================================
# CIF 리스트 로딩
# ============================================================
def load_cif_list(data_dir: str, logger: logging.Logger) -> List[str]:
    patterns = [
        os.path.join(data_dir, "*.cif"),
        os.path.join(data_dir, "*", "*.cif"),
    ]
    cif_paths = []
    for p in patterns:
        cif_paths.extend(glob.glob(p))
    cif_paths = sorted(set(cif_paths))

    if not cif_paths:
        logger.error("[DATA] No CIF files found in %s", data_dir)
        raise FileNotFoundError(f"No CIF files in {data_dir}")

    logger.info("[DATA] Found %d CIF files under %s", len(cif_paths), data_dir)
    return cif_paths


# ============================================================
# Atoms loader 생성 (랜덤 CIF 샘플링)
# ============================================================
def build_atoms_loader(
    cif_paths: List[str],
    mace_calc: MACECalculator,
    logger: logging.Logger,
):
    """
    MOFEnv에 넘겨줄 atoms_loader 클로저를 만든다.
    매 호출마다 랜덤 CIF 하나를 골라 ASE Atoms를 리턴.

    atoms.info["cif_path"] 에 경로를 넣어두면
    env.reset() 이후 main에서 현재 CIF를 알 수 있다.
    """

    def atoms_loader():
        cif_path = random.choice(cif_paths)
        atoms = read(cif_path)
        atoms.info["cif_path"] = cif_path
        atoms.calc = mace_calc
        return atoms

    logger.info("[INIT] atoms_loader with %d CIFs ready.", len(cif_paths))
    return atoms_loader


# ============================================================
# metrics.csv 기록용 헬퍼
# ============================================================
def append_metrics(
    metrics_path: str,
    ep: int,
    total_steps: int,
    ep_return: float,
    ep_steps: int,
    done_reason: str,
    fmax_last: float,
    buffer_size: int,
):
    header = (
        "episode,total_steps,return,steps,done_reason,"
        "fmax_last,buffer_size\n"
    )
    line = "{:d},{:d},{:.6f},{:d},{:s},{:.6f},{:d}\n".format(
        ep, total_steps, ep_return, ep_steps, done_reason, fmax_last, buffer_size
    )

    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        f.write(line)


# ============================================================
# 메인 학습 루프
# ============================================================
def train(args):
    logger = setup_logger(args.log_dir, logging.INFO)
    set_seed(args.seed)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("[INIT] Using device: %s", device)

    # ----------------------------
    # CIF 리스트
    # ----------------------------
    cif_paths = load_cif_list(args.data_dir, logger)

    # ----------------------------
    # MACE Calculator
    # ----------------------------
    if not os.path.exists(args.mace_model):
        logger.error("[INIT] MACE model not found at %s", args.mace_model)
        raise FileNotFoundError(args.mace_model)

    logger.info("[INIT] Loading MACE model from %s", args.mace_model)
    mace_calc = MACECalculator(
        model_path=args.mace_model,
        device=device,
        default_dtype="float32",
    )

    # ----------------------------
    # Atoms loader & Env & 초기 obs_dim
    # ----------------------------
    atoms_loader = build_atoms_loader(cif_paths, mace_calc, logger)

    env = MOFEnv(
        atoms_loader=atoms_loader,
        k_neighbors=args.k_neighbors,
        cmax=args.cmax,
        max_steps=args.max_steps,
        fmax_threshold=args.fmax_threshold,
        bond_break_ratio=args.bond_break_ratio,
        k_bond=args.k_bond,
        max_penalty=args.max_penalty,
        debug_bond=args.debug_bond,
    )

    # 첫 reset으로 obs_dim, global_dim 파악
    obs_atom, obs_global = env.reset()
    N0, feat_dim = obs_atom.shape
    obs_dim = feat_dim
    act_dim = 3  # per-atom 3D displacement

    global_dim = 0
    if obs_global is not None:
        obs_global = np.asarray(obs_global, dtype=np.float32)
        global_dim = obs_global.shape[-1]

    # atom_type_id가 env에 있다면 MultiAgentSAC에 atom-type 정보 전달
    n_atom_types = getattr(env, "n_atom_types", None)

    logger.info(
        "[INIT] First env reset: N_atoms=%d, obs_dim(per-atom)=%d, "
        "global_dim=%d, act_dim=%d",
        N0,
        obs_dim,
        global_dim,
        act_dim,
    )

    # ----------------------------
    # Replay Buffer
    # ----------------------------
    replay_buffer = MultiAgentReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        max_size=args.replay_size,
        alpha=args.per_alpha,
        beta=args.per_beta,
        n_step=args.n_step,
        gamma=args.gamma,
    )
    logger.info(
        "[INIT] ReplayBuffer: size=%d, n_step=%d, gamma=%.4f, PER(alpha=%.3f, beta=%.3f)",
        args.replay_size,
        args.n_step,
        args.gamma,
        args.per_alpha,
        args.per_beta,
    )

    # ----------------------------
    # Multi-Agent SAC Agent
    # ----------------------------
    agent = MultiAgentSAC(
        obs_dim=obs_dim,
        act_dim=act_dim,
        replay_buffer=replay_buffer,
        n_atom_types=n_atom_types,
        global_dim=global_dim,
        actor_hidden=(args.actor_hidden, args.actor_hidden),
        critic_hidden=(args.critic_hidden, args.critic_hidden),
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.init_alpha,
        target_entropy_scale=args.target_entropy_scale,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=args.lr_alpha,
        device=device,
        auto_alpha=True,
        use_v_net=False,
        max_grad_norm=args.max_grad_norm,
        per_use_weights=True,
    )

    logger.info(
        "[INIT] MultiAgentSAC ready: actor_hidden=%s, critic_hidden=%s",
        str((args.actor_hidden, args.actor_hidden)),
        str((args.critic_hidden, args.critic_hidden)),
    )

    # ----------------------------
    # 체크포인트 / 메트릭 디렉토리
    # ----------------------------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    metrics_path = os.path.join(args.log_dir, "metrics.csv")

    total_steps = 0
    start_time = time.time()

    # ----------------------------
    # 에피소드 루프
    # ----------------------------
    for ep in range(args.num_episodes):
        # Curriculum: max_steps 조정 가능 (원하면 수정)
        env.max_steps = args.max_steps

        obs_atom, obs_global = env.reset()
        obs_atom = np.asarray(obs_atom, dtype=np.float32)
        if obs_global is not None:
            obs_global = np.asarray(obs_global, dtype=np.float32)

        # atom type id (있으면 사용)
        atom_type = getattr(env, "atom_type_id", None)
        if atom_type is not None:
            atom_type = np.asarray(atom_type, dtype=np.int64)

        N_atoms = obs_atom.shape[0]
        done = False
        ep_return = 0.0
        ep_steps = 0
        last_fmax = np.nan
        done_reason = "none"

        cif_path = getattr(env, "current_cif_path", "unknown")
        logger.info("================================================================")
        logger.info("[EP %d] START", ep)
        logger.info("[EP %d] CIF = %s", ep, cif_path)
        logger.info("[EP %d] N_atoms = %d, max_steps = %d", ep, N_atoms, env.max_steps)
        logger.info("[EP %d] obs_dim(per-atom) = %d, global_dim = %d", ep, obs_dim, global_dim)

        step_iter = tqdm(
            range(env.max_steps),
            desc=f"[EP {ep}]",
            leave=False,
            ncols=100,
        )

        for step_idx in step_iter:
            total_steps += 1
            ep_steps += 1

            # 1) 행동 선택
            if total_steps < args.warmup_steps:
                # Random exploration
                actions = np.random.uniform(
                    low=-1.0, high=1.0, size=(N_atoms, act_dim)
                ).astype(np.float32)
            else:
                actions_tensor = agent.act(
                    obs_atom,
                    atom_type_id=atom_type,
                    global_feat=obs_global,
                    deterministic=False,
                )
                actions = actions_tensor.detach().cpu().numpy().astype(np.float32)

            # 2) env.step
            next_obs_atom, next_obs_global, reward, done, info = env.step(actions)

            next_obs_atom = np.asarray(next_obs_atom, dtype=np.float32)
            if next_obs_global is not None:
                next_obs_global = np.asarray(next_obs_global, dtype=np.float32)

            # next atom types (구조 바뀌어도 타입은 그대로라고 가정, 필요시 env에서 갱신)
            next_atom_type = getattr(env, "atom_type_id", None)
            if next_atom_type is not None:
                next_atom_type = np.asarray(next_atom_type, dtype=np.int64)

            ep_return += float(reward)
            last_fmax = float(info.get("Fmax", np.nan))
            done_reason = info.get("done_reason", "unknown")

            # 3) Replay buffer에 per-atom transition 저장
            replay_buffer.store(
                obs=obs_atom,
                acts=actions,
                rews=reward,
                next_obs=next_obs_atom,
                done=done,
                atom_type=atom_type,
                next_atom_type=next_atom_type,
                global_feat=obs_global,
                next_global=next_obs_global,
            )

            # 4) 학습 업데이트 (warmup 이후)
            metrics = {}
            if total_steps >= args.warmup_steps:
                for _ in range(args.updates_per_step):
                    metrics = agent.update(args.batch_size)

            # 5) tqdm 표시
            postfix = {
                "Fmax": f"{last_fmax: .2e}" if not np.isnan(last_fmax) else "nan",
                "reward": f"{reward: .4f}",
                "alpha": f"{agent.alpha: .4f}",
                "buffer": len(replay_buffer),
            }
            step_iter.set_postfix(postfix)

            # logging (간헐적으로)
            if step_idx % args.log_interval_steps == 0 or done:
                logger.info(
                    "[EP %d][STEP %d] N=%d | Fmax=%s | r=%.6f | return=%.6f | "
                    "alpha=%.4f | buffer=%d",
                    ep,
                    step_idx,
                    N_atoms,
                    f"{last_fmax: .3e}" if not np.isnan(last_fmax) else "nan",
                    float(reward),
                    ep_return,
                    agent.alpha,
                    len(replay_buffer),
                )

            # 상태 업데이트
            obs_atom = next_obs_atom
            obs_global = next_obs_global
            atom_type = next_atom_type
            N_atoms = obs_atom.shape[0]

            if done:
                logger.info(
                    "[EP %d] DONE at step %d, reason=%s",
                    ep,
                    step_idx,
                    done_reason,
                )
                break

        # 에피소드 종료 후 메트릭 기록
        elapsed = time.time() - start_time
        steps_per_sec = total_steps / max(elapsed, 1e-6)
        logger.info(
            "[EP %d] END | steps=%d | return=%.6f | last_Fmax=%s | done_reason=%s | "
            "total_steps=%d (%.2f steps/s)",
            ep,
            ep_steps,
            ep_return,
            f"{last_fmax: .3e}" if not np.isnan(last_fmax) else "nan",
            done_reason,
            total_steps,
            steps_per_sec,
        )

        append_metrics(
            metrics_path=metrics_path,
            ep=ep,
            total_steps=total_steps,
            ep_return=ep_return,
            ep_steps=ep_steps,
            done_reason=done_reason,
            fmax_last=last_fmax if not np.isnan(last_fmax) else 0.0,
            buffer_size=len(replay_buffer),
        )

        # 체크포인트 저장
        if (ep + 1) % args.ckpt_interval == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_ep{ep+1:05d}.pt")
            agent.save(ckpt_path)
            logger.info("[CKPT] Saved checkpoint: %s", ckpt_path)

    logger.info("============================================================")
    logger.info("[TRAIN] Finished all episodes. total_steps=%d", total_steps)
    logger.info("============================================================")


# ============================================================
# Argparse
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="MOF Multi-Agent SAC (per-atom) Training"
    )

    # 데이터 / 모델
    parser.add_argument(
        "--data_dir",
        type=str,
        default="mofs/train_pool_valid",
        help="Directory containing MOF CIF files.",
    )
    parser.add_argument(
        "--mace_model",
        type=str,
        default="mofs_v2.model",
        help="Path to MACE model file.",
    )

    # 환경 설정
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=12,
        help="Number of nearest neighbors (k-NN) for local features.",
    )
    parser.add_argument(
        "--cmax",
        type=float,
        default=0.4,
        help="Force-based displacement scaling cap.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Max steps per episode (can be curriculum-controlled).",
    )
    parser.add_argument(
        "--fmax_threshold",
        type=float,
        default=0.12,
        help="Force convergence threshold (success criterion).",
    )
    parser.add_argument(
        "--bond_break_ratio",
        type=float,
        default=2.4,
        help="Bond length / covalent_radius ratio for soft bond penalty.",
    )
    parser.add_argument(
        "--k_bond",
        type=float,
        default=3.0,
        help="Bond penalty stiffness.",
    )
    parser.add_argument(
        "--max_penalty",
        type=float,
        default=10.0,
        help="Max bond penalty per step.",
    )
    parser.add_argument(
        "--debug_bond",
        action="store_true",
        help="Enable detailed bond penalty debugging logs.",
    )

    # SAC / Replay 설정
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--replay_size",
        type=int,
        default=200_000,
        help="Replay buffer max size (per-atom transitions).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for SAC updates (per-atom samples).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Soft-update coefficient for target networks.",
    )
    parser.add_argument(
        "--init_alpha",
        type=float,
        default=0.2,
        help="Initial entropy temperature alpha.",
    )
    parser.add_argument(
        "--target_entropy_scale",
        type=float,
        default=1.0,
        help="Target entropy = -act_dim * scale.",
    )
    parser.add_argument(
        "--actor_hidden",
        type=int,
        default=256,
        help="Hidden layer size for actor MLP (2 layers).",
    )
    parser.add_argument(
        "--critic_hidden",
        type=int,
        default=256,
        help="Hidden layer size for critic MLP (2 layers).",
    )
    parser.add_argument(
        "--lr_actor",
        type=float,
        default=3e-4,
        help="Actor learning rate.",
    )
    parser.add_argument(
        "--lr_critic",
        type=float,
        default=3e-4,
        help="Critic learning rate.",
    )
    parser.add_argument(
        "--lr_alpha",
        type=float,
        default=3e-4,
        help="Alpha optimizer learning rate.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=10.0,
        help="Gradient clipping max norm (0 or negative to disable).",
    )

    # PER / n-step
    parser.add_argument(
        "--per_alpha",
        type=float,
        default=0.6,
        help="PER alpha (priority exponent).",
    )
    parser.add_argument(
        "--per_beta",
        type=float,
        default=0.4,
        help="PER beta (IS weight exponent).",
    )
    parser.add_argument(
        "--n_step",
        type=int,
        default=3,
        help="n-step return for replay buffer.",
    )

    # Warmup / updates
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10_000,
        help="Number of steps with random policy before SAC updates.",
    )
    parser.add_argument(
        "--updates_per_step",
        type=int,
        default=1,
        help="Number of SAC updates per env step after warmup.",
    )

    # Logging / Checkpoint
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to store logs and metrics.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints",
        help="Directory to store model checkpoints.",
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=50,
        help="Save checkpoint every N episodes.",
    )
    parser.add_argument(
        "--log_interval_steps",
        type=int,
        default=50,
        help="Log step metrics every N steps within an episode.",
    )

    # 기타
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu' (default: auto).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # max_grad_norm <= 0이면 사용 안 함
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        args.max_grad_norm = None
    train(args)
