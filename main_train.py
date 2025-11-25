import os
import sys
import time
import csv
import argparse
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
import torch
from tqdm import tqdm
from mace.calculators import MACECalculator

from env.mof_env import MOFEnv, AtomsLoader
from sac.agent import SACAgent
from utils.replay_buffer import ReplayBuffer


# ------------------------------------------------------------
# 로깅 세팅
# ------------------------------------------------------------

def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        # File handler (rotating)
        fh = RotatingFileHandler(
            os.path.join(log_dir, "train.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Stream handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-atom Multi-agent SAC for MOF fast relaxation (MACE)"
    )

    parser.add_argument("--mof-dir", type=str, default="mofs/train_pool_valid",
                        help="Directory containing training MOF CIFs")
    parser.add_argument("--model-path", type=str, default="mofs_v2.model",
                        help="Path to MACE model (.model)")
    parser.add_argument("--log-dir", type=str, default="logs_mof_sac",
                        help="Logging directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=42)

    # RL hyperparams
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=20_000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)

    parser.add_argument("--fixed-alpha", action="store_true",
                        help="If set, fix alpha instead of auto entropy tuning")
    parser.add_argument("--alpha", type=float, default=0.2)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logger(args.log_dir)
    logger.info(f"Arguments: {vars(args)}")

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # MACE Calculator
    logger.info(f"Loading MACE model from {args.model_path}")
    calculator = MACECalculator(model_path=args.model_path, device=device_str)

    # Atoms loader
    atoms_loader = AtomsLoader(args.mof-dir if hasattr(args, "mof-dir") else args.mof_dir)

    # Env
    env = MOFEnv(
        atoms_loader=atoms_loader,
        calculator=calculator,
        max_steps=args.max_steps,
        fmax_threshold=0.12,
        bond_break_ratio=2.4,
        k_bond=3.0,
        max_penalty=10.0,
        alpha=0.04,
        w_f=1.0,
        w_bond=1.0,
        w_com=0.1,
        debug_bond=False,
        cmax=0.05
    )

    # 첫 reset으로 obs_dim 확인
    obs0 = env.reset()
    obs_dim = obs0.shape[1]
    act_dim = 3  # 3D displacement

    logger.info(f"Initial obs_dim={obs_dim}, act_dim={act_dim}, N_atoms={obs0.shape[0]}")

    # Agent & Buffer
    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        gamma=args.gamma,
        tau=args.tau,
        lr=args.lr,
        alpha=args.alpha,
        automatic_entropy_tuning=not args.fixed_alpha,
    )

    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        size=args.buffer_size,
        device=device,
    )

    metrics_path = os.path.join(args.log_dir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)

    total_steps = 0
    updates = 0

    with open(metrics_path, "a", newline="") as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow([
                "episode", "total_steps", "ep_steps", "return",
                "Fmax_last", "Fmean_last",
                "bond_penalty_last", "com_shift_last",
                "buffer_size",
            ])

        for ep in range(args.num_episodes):
            obs = env.reset()
            done = False
            ep_return = 0.0
            ep_steps = 0

            last_info = {}

            pbar = tqdm(
                range(env.max_steps),
                desc=f"[EP {ep}]",
                leave=False,
            )

            for _ in pbar:
                actions = agent.select_action(obs, deterministic=False)

                next_obs, reward, done, info = env.step(actions)

                # Replay buffer에 per-atom으로 flatten 저장
                replay_buffer.store_batch(
                    obs_batch=obs,
                    act_batch=actions,
                    rew=reward,
                    next_obs_batch=next_obs,
                    done=done,
                )

                ep_return += reward
                ep_steps += 1
                total_steps += 1
                last_info = info
                obs = next_obs

                # 학습 업데이트
                if (
                    total_steps >= args.warmup_steps
                    and len(replay_buffer) >= args.batch_size
                    and total_steps % args.update_every == 0
                ):
                    for _ in range(args.gradient_steps):
                        loss_info = agent.update_parameters(
                            replay_buffer,
                            batch_size=args.batch_size,
                        )
                        updates += 1

                    logger.info(
                        "[UPDATE] step=%d | updates=%d | "
                        "critic=%.4f (q1=%.4f, q2=%.4f) | actor=%.4f | alpha=%.4f",
                        total_steps,
                        updates,
                        loss_info["critic_loss"],
                        loss_info["q1_loss"],
                        loss_info["q2_loss"],
                        loss_info["actor_loss"],
                        loss_info["alpha"],
                    )

                # 로그 & 진행바
                pbar.set_postfix({
                    "R": f"{ep_return:.2f}",
                    "Fmax": f"{info['Fmax']:.2e}",
                    "buf": len(replay_buffer),
                })

                logger.info(
                    "[EP %d][STEP %d] N=%d | Fmax=%.3e, Fmean=%.3e | "
                    "E=%.6f, E/atom=%.6f | COM=%.3e | "
                    "Disp_mean=%.3e, Disp_max=%.3e | "
                    "bond_pen=%.3f | rew=%.6f | total_steps=%d | buffer=%d",
                    ep,
                    ep_steps,
                    info["n_atoms"],
                    info["Fmax"],
                    info["Fmean"],
                    info["energy"],
                    info["energy_per_atom"],
                    info["com_shift"],
                    info["disp_mean"],
                    info["disp_max"],
                    info["bond_penalty"],
                    reward,
                    total_steps,
                    len(replay_buffer),
                )

                if done:
                    break

            # episode summary
            if last_info:
                writer.writerow([
                    ep,
                    total_steps,
                    ep_steps,
                    ep_return,
                    last_info.get("Fmax", 0.0),
                    last_info.get("Fmean", 0.0),
                    last_info.get("bond_penalty", 0.0),
                    last_info.get("com_shift", 0.0),
                    len(replay_buffer),
                ])
                f_csv.flush()

            logger.info(
                "[EP %d] done | steps=%d | return=%.6f | total_steps=%d",
                ep,
                ep_steps,
                ep_return,
                total_steps,
            )

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
