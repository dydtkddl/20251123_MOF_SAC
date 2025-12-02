#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gen_bfgs_trajectories.py
========================

MACE surrogate + ASE BFGS로 QMOF 구조들을 relax하여
BFGS trajectory 기반 expert 데이터를 생성하는 스크립트.

- 입력:
    - QMOF CIF pool 디렉토리 (기본: ../mofs/train_pool_valid)
    - MACE 모델 경로 (기본: ../mofs_v2.model)

- 출력:
    - output_dir 아래에 CIF별 npz 파일 생성
      파일명 예) EXP_QMOF_XXXX.npz

  npz 내용 (per-atom, per-step pair 기준):
    - obs          : (N_samples, obs_dim)
    - disp         : (N_samples, 3)    # BFGS에서 t→t+1 실제 변위
    - act4         : (N_samples, 4)    # [gate=1, dx_raw, dy_raw, dz_raw]
    - traj_id      : (N_samples,)      # trajectory index
    - step_idx     : (N_samples,)      # BFGS step index t (중간 step만, 1..L-2)
    - atom_idx     : (N_samples,)      # 원자 index (0..N_atom-1)
    - natoms       : scalar (int)      # 각 trajectory 당 원자 수 (모든 sample 동일)
    - obs_dim      : scalar (int)
    - cif_path     : string

  여기서 act4는 Env의 4D action 스케일링을 반대로 추정한 값으로,
  대략적으로 Env에서 동일한 disp를 내도록 설계됨:
      disp_target ≈ 0.003 * gate * disp_raw * (scale / cmax)
  (gate=1 가정, per-atom scale = min(||F||, cmax))

사용법:
    cd 프로젝트_루트
    python scripts/gen_bfgs_trajectories.py \
        --pool-dir ../mofs/train_pool_valid \
        --output-dir ./expert_data_bfgs \
        --max-cifs 100 \
        --fmax 0.05 \
        --max-steps 200
"""

import os
import sys
import time
import argparse
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
from tqdm import tqdm

from ase.io import read
from ase.optimize import BFGS

from mace.calculators import MACECalculator

# 프로젝트 루트 기준으로 env/ 모듈이 import 되도록 경로 추가
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.mof_env import MOFEnv  # noqa: E402


# ============================================================
# Logging setup
# ============================================================

def setup_logger(log_path: str, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("gen_bfgs")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 기존 핸들러 중복 방지
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 콘솔
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # 파일(회전)
        fh = RotatingFileHandler(
            log_path,
            maxBytes=10_000_000,
            backupCount=5,
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ============================================================
# BFGS Relaxation for single CIF
# ============================================================

def relax_single_cif(
    cif_path: str,
    calc: MACECalculator,
    fmax: float,
    max_steps: int,
    maxstep: float,
    logger: logging.Logger,
):
    """
    단일 CIF 파일에 대해 ASE BFGS를 수행하여 snapshot 리스트 반환.

    Returns
    -------
    snapshots : list[ase.Atoms]
        step 0 (초기 구조)부터 마지막 step까지의 Atoms 복사본 리스트.
    energies : list[float]
        각 snapshot의 potential energy.
    forces : list[np.ndarray]
        각 snapshot의 forces, shape (N_atoms, 3).
    """
    logger.info(f"[BFGS] Start CIF: {cif_path}")
    atoms = read(cif_path)
    atoms.calc = calc

    snapshots = []
    energies = []
    forces = []

    # snapshot 0 (초기 구조)
    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    snapshots.append(atoms.copy())
    energies.append(float(e0))
    forces.append(f0.copy())

    natoms = len(atoms)
    logger.info(
        f"[BFGS] Initial energy={e0:.6f} eV, "
        f"N_atoms={natoms}"
    )

    opt = BFGS(atoms, maxstep=maxstep, logfile=None)

    # tqdm progress bar for BFGS steps
    desc = f"BFGS[{os.path.basename(cif_path)}]"
    with tqdm(total=max_steps, desc=desc, ncols=120) as pbar:

        def callback():
            # BFGS 1 step 후 snapshot 저장
            e = atoms.get_potential_energy()
            f = atoms.get_forces()

            snapshots.append(atoms.copy())
            energies.append(float(e))
            forces.append(f.copy())

            pbar.update(1)

        opt.attach(callback, interval=1)
        start_time = time.time()
        opt.run(fmax=fmax, steps=max_steps)
        elapsed = time.time() - start_time

    logger.info(
        f"[BFGS] Done CIF={cif_path}, "
        f"steps={len(snapshots)-1}, "
        f"final_energy={energies[-1]:.6f} eV, "
        f"elapsed={elapsed:.2f} s"
    )

    return snapshots, energies, forces


# ============================================================
# Build Expert Dataset from BFGS snapshots
# ============================================================

def build_expert_from_snapshots(
    snapshots,
    forces_list,
    calc: MACECalculator,
    cif_path: str,
    traj_id: int,
    logger: logging.Logger,
    cmax: float = 0.4,
):
    """
    BFGS snapshot 리스트로부터 per-atom expert 데이터를 생성.

    - 각 t에 대해:
        prev = t-1, curr = t, next = t+1
        obs_t      : MOFEnv의 _obs() (prev_force, prev_disp 세팅 포함)
        disp_t     : curr→next 실제 변위
        act4_t     : Env 공식 역산하여 [gate=1, dx_raw, dy_raw, dz_raw] 추정

    Parameters
    ----------
    snapshots : list[ase.Atoms]
    forces_list : list[np.ndarray]
    calc : MACECalculator
    cif_path : str
    traj_id : int
    logger : logging.Logger
    cmax : float
        MOFEnv의 cmax와 동일하게 맞춰줌.

    Returns
    -------
    data : dict or None
        npz로 바로 저장 가능한 dict 구조.
        snapshot 수가 너무 적으면(None) 반환.
    """
    L = len(snapshots)
    if L < 3:
        logger.warning(
            f"[EXPERT] CIF={cif_path} has too few snapshots (L={L}), skip."
        )
        return None

    natoms = len(snapshots[0])
    logger.info(
        f"[EXPERT] Building expert data for CIF={cif_path}, "
        f"traj_id={traj_id}, L={L}, N_atoms={natoms}"
    )

    all_obs = []
    all_disp = []
    all_act4 = []
    all_step_idx = []
    all_atom_idx = []

    # t=1..L-2 까지만 사용 (prev/next 모두 존재하는 구간)
    for t in range(1, L - 1):
        atoms_prev = snapshots[t - 1].copy()
        atoms_curr = snapshots[t].copy()
        atoms_next = snapshots[t + 1].copy()

        # calc 보장
        atoms_prev.calc = calc
        atoms_curr.calc = calc

        # Env 구성용 loader (현재 snapshot 기준)
        def loader(at=atoms_curr):
            return at.copy()

        env = MOFEnv(
            atoms_loader=loader,
            k_neighbors=12,
            cmax=cmax,
            max_steps=300,
            fmax_threshold=0.05,
            bond_break_ratio=2.4,
            k_bond=3.0,
            max_penalty=10.0,
            debug_bond=False,
            random_perturb=False,
            perturb_sigma=0.0,
            max_perturb=0.0,
            terminal_bonus_base=0.0,
            time_penalty=0.0,
            fail_penalty=0.0,
        )

        # reset()으로 bonds/adj 등 초기화 + forces(curr) 계산
        obs0 = env.reset()  # (N_atoms, obs_dim)
        obs_dim = obs0.shape[1]

        # prev forces/disp/COM을 BFGS trajectory 기반으로 채워 줌
        f_prev = forces_list[t - 1].astype(np.float32)
        disp_prev = (
            atoms_curr.positions - atoms_prev.positions
        ).astype(np.float32)
        com_prev = atoms_prev.positions.mean(axis=0).astype(np.float32)

        if f_prev.shape != env.forces.shape:
            logger.warning(
                "[EXPERT] Force shape mismatch at t=%d: %s vs %s",
                t, f_prev.shape, env.forces.shape
            )

        env.prev_forces = f_prev
        env.prev_disp = disp_prev
        env.COM_prev = com_prev

        # obs_t 다시 계산 (prev_* 반영)
        obs_t = env._obs()  # (N_atoms, obs_dim)

        # target disp: curr→next
        disp_t = (
            atoms_next.positions - atoms_curr.positions
        ).astype(np.float32)  # (N_atoms, 3)

        # Env 공식 역산으로 approximate expert action (4D) 계산
        # Env step에서:
        #   scale = min(||F||, cmax)
        #   disp_env = 0.003 * gate * disp_raw * (scale / cmax)
        # 여기서는 gate=1 가정, BFGS disp ≈ disp_env로 보고,
        #   disp_raw ≈ disp / (0.003 * scale / cmax)
        forces_curr = env.forces  # (N_atoms, 3)
        gnorm = np.linalg.norm(forces_curr, axis=1)
        scale = np.minimum(gnorm, cmax).reshape(-1, 1)  # (N_atoms, 1)
        denom = 0.003 * (scale / cmax) + 1e-8           # (N_atoms, 1)

        disp_raw = disp_t / denom                       # (N_atoms, 3)
        # [-1, 1]로 클리핑 (Env에서도 tanh 후 [-1,1] 사용)
        disp_raw = np.clip(disp_raw, -1.0, 1.0)

        gate_raw = np.ones((natoms, 1), dtype=np.float32)  # gate≈1
        act4_t = np.concatenate([gate_raw, disp_raw], axis=1)  # (N, 4)

        all_obs.append(obs_t.astype(np.float32))
        all_disp.append(disp_t.astype(np.float32))
        all_act4.append(act4_t.astype(np.float32))

        all_step_idx.append(
            np.full(natoms, t, dtype=np.int32)
        )
        all_atom_idx.append(
            np.arange(natoms, dtype=np.int32)
        )

    obs_arr = np.concatenate(all_obs, axis=0)       # (K * N_atoms, obs_dim)
    disp_arr = np.concatenate(all_disp, axis=0)     # (K * N_atoms, 3)
    act4_arr = np.concatenate(all_act4, axis=0)     # (K * N_atoms, 4)
    step_idx = np.concatenate(all_step_idx, axis=0) # (K * N_atoms,)
    atom_idx = np.concatenate(all_atom_idx, axis=0) # (K * N_atoms,)

    n_samples = obs_arr.shape[0]

    logger.info(
        f"[EXPERT] CIF={cif_path}, traj_id={traj_id} → "
        f"snapshots_used={L-2}, samples={n_samples}, "
        f"obs_dim={obs_dim}, natoms={natoms}"
    )

    data = dict(
        obs=obs_arr,
        disp=disp_arr,
        act4=act4_arr,
        traj_id=np.full(n_samples, traj_id, dtype=np.int32),
        step_idx=step_idx,
        atom_idx=atom_idx,
        natoms=np.int32(natoms),
        obs_dim=np.int32(obs_dim),
        cif_path=np.array([cif_path]),   # string은 1D 배열로 저장
    )

    return data


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate BFGS-based expert data from QMOF CIFs."
    )
    p.add_argument(
        "--pool-dir",
        type=str,
        default="../mofs/train_pool_valid",
        help="Directory containing QMOF .cif files.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./expert_data_bfgs",
        help="Directory to save expert npz files.",
    )
    p.add_argument(
        "--mace-model",
        type=str,
        default="../mofs_v2.model",
        help="Path to MACE model file.",
    )
    p.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="BFGS convergence threshold on force (eV/Å).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum BFGS steps per structure.",
    )
    p.add_argument(
        "--maxstep",
        type=float,
        default=0.04,
        help="Maximum step size for BFGS (Å).",
    )
    p.add_argument(
        "--max-cifs",
        type=int,
        default=None,
        help="Maximum number of CIF files to process (for quick tests).",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default="gen_bfgs.log",
        help="Log file name (will be created in output-dir).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CIF shuffling.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_file)
    logger = setup_logger(log_path, args.log_level)

    logger.info("========== BFGS Expert Generation Start ==========")
    logger.info(f"POOL_DIR      = {args.pool_dir}")
    logger.info(f"OUTPUT_DIR    = {args.output_dir}")
    logger.info(f"MACE_MODEL    = {args.mace_model}")
    logger.info(f"FMAX          = {args.fmax}")
    logger.info(f"MAX_STEPS     = {args.max_steps}")
    logger.info(f"MAXSTEP       = {args.maxstep}")
    logger.info(f"MAX_CIFS      = {args.max_cifs}")
    logger.info(f"SEED          = {args.seed}")

    # MACE surrogate
    logger.info("[MACE] Initializing surrogate calculator...")
    calc = MACECalculator(
        model_paths=[args.mace_model],
        head="pbe_d3",
        device="cuda",
        default_dtype="float32",
    )
    logger.info("[MACE] Loaded model.")

    # CIF 리스트
    all_cifs = [
        os.path.join(args.pool_dir, f)
        for f in os.listdir(args.pool_dir)
        if f.endswith(".cif")
    ]
    if not all_cifs:
        logger.error(f"No .cif files found in {args.pool_dir}")
        return

    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_cifs)

    if args.max_cifs is not None:
        all_cifs = all_cifs[: args.max_cifs]

    logger.info(f"Total CIFs to process: {len(all_cifs)}")

    global_start = time.time()
    n_success = 0
    n_failed = 0

    for traj_id, cif_path in enumerate(tqdm(all_cifs, desc="CIFs", ncols=120)):
        cif_name = os.path.basename(cif_path)
        out_name = f"EXP_{os.path.splitext(cif_name)[0]}.npz"
        out_path = os.path.join(args.output_dir, out_name)

        if os.path.exists(out_path):
            logger.info(f"[SKIP] {out_path} already exists, skip CIF={cif_path}")
            continue

        try:
            snapshots, energies, forces = relax_single_cif(
                cif_path=cif_path,
                calc=calc,
                fmax=args.fmax,
                max_steps=args.max_steps,
                maxstep=args.maxstep,
                logger=logger,
            )

            data = build_expert_from_snapshots(
                snapshots=snapshots,
                forces_list=forces,
                calc=calc,
                cif_path=cif_path,
                traj_id=traj_id,
                logger=logger,
                cmax=0.4,  # 현재 Env와 동일한 값
            )

            if data is None:
                logger.warning(
                    f"[SKIP] CIF={cif_path} produced no expert data."
                )
                n_failed += 1
                continue

            np.savez_compressed(out_path, **data)
            n_success += 1
            logger.info(f"[SAVE] Expert data saved → {out_path}")

        except Exception as e:
            logger.exception(f"[ERROR] Failed CIF={cif_path}: {e}")
            n_failed += 1

    elapsed = time.time() - global_start
    logger.info("========== BFGS Expert Generation Done ==========")
    logger.info(f"Success: {n_success}, Failed: {n_failed}, Elapsed={elapsed/3600:.3f} h")

    print(
        f"[DONE] BFGS expert generation complete: "
        f"success={n_success}, failed={n_failed}, elapsed={elapsed/3600:.3f} h"
    )


if __name__ == "__main__":
    main()
