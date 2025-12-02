#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RL vs BFGS Comparative Engine (with Perturb Support)
====================================================
기능:
- Phase2 SAC checkpoint 로드
- 약 100개 MOF 샘플링
- RL relaxation & BFGS relaxation 모두 실행
- perturb 적용지원 (RL + BFGS 둘 다)
- step-by-step energy/fmax/time CSV 저장
- RL vs BFGS energy plot 저장
- summary.csv 저장
- 상세 logging 출력
"""

import os
import time
import logging
from logging.handlers import RotatingFileHandler
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ase.io import read
from ase.optimize import BFGS

import torch
from mace.calculators import MACECalculator

from env.mof_env import MOFEnv
from sac.agent import SACAgent
from utils.replay_buffer import ReplayBuffer


# ===============================================================
# LOGGING SETUP
# ===============================================================
os.makedirs("compare_logs", exist_ok=True)

handler = RotatingFileHandler(
    "compare_logs/compare_engine.log",
    maxBytes=20_000_000,
    backupCount=5,
)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
))

logger = logging.getLogger("compare")
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# ===============================================================
# CONFIG
# ===============================================================
PHASE2_CKPT = "/home/yongsang/20251123_MOF_SAC/phase03_qmof_branch02_cmax_control_20251130/checkpoints_phase2/ckpt_ep1500_final.pt"
POOL_DIR    = "../mofs/train_pool_valid"

N_TEST = 10
FMAX_THRESH = 0.05
RL_MAX_STEPS = 1000
C_MAX = 0.4

# Perturb 설정
USE_PERTURB    = True
PERTURB_SIGMA  = 0.04  # Å
MAX_PERTURB    = 0.25  # Å

SAVE_DIR = "compare_results"
os.makedirs(SAVE_DIR, exist_ok=True)


# ===============================================================
# Surrogate model
# ===============================================================
calc = MACECalculator(
    model_paths=["../mofs_v2.model"],
    head="pbe_d3",
    device="cuda",
    default_dtype="float32"
)


# ===============================================================
# Perturb 함수 (BFGS에서도 사용)
# ===============================================================
def apply_perturb(atoms, sigma, max_perturb):
    pos = atoms.get_positions()

    disp = np.random.normal(0, sigma, pos.shape)
    disp = np.clip(disp, -max_perturb, max_perturb)

    new_pos = pos + disp
    atoms.set_positions(new_pos)

    logger.info(f"[PERTURB] Applied: sigma={sigma:.3f}, max={max_perturb:.3f}")


# ===============================================================
# Load Phase2 policy
# ===============================================================
def load_policy(checkpoint_path, obs_dim):
    logger.info(f"[LOAD] SAC checkpoint: {checkpoint_path}")

    rb = ReplayBuffer(obs_dim=obs_dim, max_size=1)
    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=3,
        replay_buffer=rb,
        device="cuda",
        lr=3e-4,
        gamma=0.995,
        tau=5e-3,
        batch_size=256,
    )

    ckpt = torch.load(checkpoint_path, map_location=agent.device)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.q1.load_state_dict(ckpt["q1"])
    agent.q2.load_state_dict(ckpt["q2"])
    agent.v.load_state_dict(ckpt["v"])
    agent.v_tgt.load_state_dict(ckpt["v_tgt"])
    agent.log_alpha.data.fill_(ckpt["log_alpha"])

    logger.info("[LOAD] Policy loaded successfully.")
    return agent


# ===============================================================
# RL RELAXATION
# ===============================================================
def rl_relax(atoms, agent, save_dir, max_steps=RL_MAX_STEPS):
    os.makedirs(save_dir, exist_ok=True)

    env = MOFEnv(
        atoms_loader=lambda: atoms,
        k_neighbors=12,
        fmax_threshold=FMAX_THRESH,
        max_steps=max_steps,
        cmax=C_MAX,
        random_perturb=USE_PERTURB,
        perturb_sigma=PERTURB_SIGMA,
        max_perturb=MAX_PERTURB,
    )

    obs = env.reset()
    N = env.N

    step_log = []
    total_t0 = time.time()

    for step in range(max_steps):
        t0 = time.time()

        actions = []
        for i in range(N):
            actions.append(agent.act(obs[i]))
        actions = np.stack(actions, axis=0)

        actions -= actions.mean(axis=0, keepdims=True)

        next_obs, reward, done = env.step(actions)

        Etot = env.atoms.get_potential_energy()
        fmax = np.max(np.linalg.norm(env.forces, axis=1))

        step_log.append({
            "step": step,
            "energy": Etot,
            "fmax": fmax,
            "time": time.time() - t0,
        })

        logger.info(
            f"[RL][STEP {step}] E={Etot:.6f} fmax={fmax:.6f} done={done}"
        )

        obs = next_obs
        if done:
            break

    total_time = time.time() - total_t0

    df = pd.DataFrame(step_log)
    df.to_csv(os.path.join(save_dir, "rl_steps.csv"), index=False)

    return df["energy"].tolist(), df["fmax"].tolist(), step, total_time


# ===============================================================
# BFGS RELAXATION
# ===============================================================
def bfgs_relax(atoms, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    atoms.calc = calc

    step_log = []

    def logB():
        Etot = atoms.get_potential_energy()
        fmax = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
        step_log.append({
            "step": len(step_log),
            "energy": Etot,
            "fmax": fmax,
            "time": 0.0,
        })
        logger.info(f"[BFGS][STEP {len(step_log)}] E={Etot:.6f} fmax={fmax:.6f}")

    total_t0 = time.time()

    opt = BFGS(atoms, logfile=None)
    opt.attach(logB, interval=1)
    opt.run(fmax=FMAX_THRESH, steps=RL_MAX_STEPS)

    total_time = time.time() - total_t0

    df = pd.DataFrame(step_log)
    df.to_csv(os.path.join(save_dir, "bfgs_steps.csv"), index=False)

    return df["energy"].tolist(), df["fmax"].tolist(), len(df), total_time


# ===============================================================
# MAIN
# ===============================================================
def main():

    logger.info("[START] RL vs BFGS Comparative Engine")

    cifs = sorted(glob(os.path.join(POOL_DIR, "*.cif")))
    selected = np.random.choice(cifs, min(N_TEST, len(cifs)), replace=False)
    logger.info(f"[DATA] Selected {len(selected)} MOFs")

    probe_atoms = read(selected[0])
    probe_atoms.calc = calc
    env_probe = MOFEnv(
        atoms_loader=lambda: probe_atoms,
        k_neighbors=12,
        fmax_threshold=FMAX_THRESH,
        max_steps=10,
        cmax=C_MAX
    )
    obs_probe = env_probe.reset()
    obs_dim = obs_probe.shape[1]

    agent = load_policy(PHASE2_CKPT, obs_dim)

    summary_path = os.path.join(SAVE_DIR, "summary.csv")
    with open(summary_path, "w") as f:
        f.write("name,rl_steps,bfgs_steps,rl_time,bfgs_time,rl_finalE,bfgs_finalE,perturb_sigma\n")

    # MAIN LOOP
    for cif in tqdm(selected, desc="Comparing", ncols=120):
        name = os.path.basename(cif).replace(".cif", "")
        logger.info(f"\n===== Testing {name} =====")
        logger.info(f"[CONFIG] perturb={USE_PERTURB}, sigma={PERTURB_SIGMA}, max={MAX_PERTURB}")

        out_dir  = os.path.join(SAVE_DIR, name)
        rl_dir   = os.path.join(out_dir, "rl")
        bfgs_dir = os.path.join(out_dir, "bfgs")
        os.makedirs(out_dir, exist_ok=True)

        atoms_rl = read(cif)
        atoms_rl.calc = calc

        atoms_bfgs = read(cif)
        atoms_bfgs.calc = calc

        # ---------------------------
        # 공정성: BFGS도 perturb 적용
        # ---------------------------
        if USE_PERTURB:
            apply_perturb(atoms_bfgs, PERTURB_SIGMA, MAX_PERTURB)

        # RL 실행
        rl_E, rl_F, rl_steps, rl_time = rl_relax(atoms_rl, agent, rl_dir)

        # BFGS 실행
        bfgs_E, bfgs_F, bfgs_steps, bfgs_time = bfgs_relax(atoms_bfgs, bfgs_dir)

        # Plot
        plt.figure(figsize=(8,6))
        plt.plot(rl_E, label="RL", lw=2)
        plt.plot(bfgs_E, label="BFGS", lw=2)
        plt.xlabel("Step")
        plt.ylabel("Energy (eV)")
        plt.title(f"{name}: RL vs BFGS")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "energy_compare.png"))
        plt.close()

        # Summary
        with open(summary_path, "a") as f:
            f.write(
                f"{name},{rl_steps},{bfgs_steps},{rl_time:.4f},{bfgs_time:.4f},{rl_E[-1]},{bfgs_E[-1]},{PERTURB_SIGMA}\n"
            )

        logger.info(
            f"[DONE] {name} | RL={rl_steps}, BFGS={bfgs_steps} | "
            f"RL time={rl_time:.2f}s, BFGS time={bfgs_time:.2f}s"
        )

    logger.info("[END] Comparison finished.")


if __name__ == "__main__":
    main()
