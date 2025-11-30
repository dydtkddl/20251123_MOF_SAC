#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QMOF → (optional perturb) → MACECalculator → BFGS Optimization
Fully SAC-matching random perturb mechanism included.
"""

import os
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

from ase.io import read, write
from ase.optimize import BFGS
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii

from mace.calculators import MACECalculator


############################################################
# Logging
############################################################
os.makedirs("logs", exist_ok=True)
log_handler = RotatingFileHandler("logs/bfgs_mace.log", maxBytes=10_000_000, backupCount=3)
log_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

logger = logging.getLogger("BFGS")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
logger.propagate = False


############################################################
# SAC-style perturb: Gaussian + max_perturb + bond-safe clamp
############################################################
def apply_sac_style_perturb(atoms, sigma=0.05, max_perturb=0.20):
    """
    Atoms positions ← atoms.positions + perturb
    - sigma: Gaussian noise std
    - max_perturb: per-atom max displacement (Å)
    - bond-safe clamp: ensures d/d0 ∈ [0.8, 1.2]
    """

    logger.info(f"[PERTURB] sigma={sigma:.3f}, max_perturb={max_perturb:.3f}")

    pos = atoms.positions.copy()
    N = len(pos)

    # ---------------------
    # 1. detect true bonds (like SAC env)
    # ---------------------
    i, j, offsets = neighbor_list("ijS", atoms, cutoff=4.0)
    cell = atoms.cell.array

    bond_pairs = []
    bond_d0 = []

    for a, b, off in zip(i, j, offsets):
        rel = pos[b] + off @ cell - pos[a]
        d = np.linalg.norm(rel)

        rc = covalent_radii[atoms[a].number] + covalent_radii[atoms[b].number]

        if d <= rc + 0.4:
            bond_pairs.append((a, b))
            bond_d0.append(d)

    bond_pairs = np.array(bond_pairs, int)
    bond_d0 = np.array(bond_d0, float)

    # ---------------------
    # 2. Gaussian noise
    # ---------------------
    delta = np.random.normal(0.0, sigma, size=pos.shape)

    # per-atom max clip
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_perturb / np.maximum(norms, 1e-12))
    delta *= scale

    # ---------------------
    # 3. bond-safe clamp
    # ---------------------
    max_ratio = 1.20
    min_ratio = 0.80

    for _ in range(5):
        new_pos = pos + delta
        bad = False

        for idx, (a, b) in enumerate(bond_pairs):
            rel = new_pos[b] - new_pos[a]
            d = np.linalg.norm(rel)
            ratio = d / bond_d0[idx]

            if ratio > max_ratio or ratio < min_ratio:
                bad = True
                break

        if not bad:
            pos = new_pos
            atoms.positions = pos
            logger.info("[PERTURB] accepted (bond-safe)")
            return

        delta *= 0.5

    atoms.positions = pos
    logger.info("[PERTURB] forced accept after max clamp loops")


############################################################
# Custom BFGS optimizer with tqdm
############################################################
class BFGSProgress(BFGS):
    def __init__(self, atoms, max_steps=300, **kwargs):
        super().__init__(atoms, **kwargs)
        self.max_steps = max_steps
        self.step_count = 0
        self.pbar = tqdm(total=max_steps, ncols=100, desc="[BFGS]")

        if os.path.exists("traj.xyz"):
            os.remove("traj.xyz")

    def step(self, forces=None):
        super().step(forces)
        self.step_count += 1
        self.pbar.update(1)

        e = self.atoms.get_potential_energy()
        forces = self.atoms.get_forces()
        fmax = np.abs(forces).max()

        logger.info(f"[STEP {self.step_count}] E={e:.8f} | Fmax={fmax:.6f}")
        self.atoms.write("traj.xyz", append=True)

        if self.step_count >= self.max_steps:
            raise RuntimeError("Max steps reached. Stopping optimization.")

    def run(self, fmax=0.05):
        try:
            super().run(fmax=fmax)
        finally:
            self.pbar.close()


############################################################
# Main optimization function
############################################################
def optimize_with_perturb(
    cif_path,
    mace_model="mofs_v2.model",
    device="cuda",
    fmax=0.05,
    max_steps=300,
    sigma=0.05,
    max_perturb=0.30,
):
    logger.info("="*80)
    logger.info(f"[LOAD CIF] {cif_path}")

    # Load CIF
    atoms = read(cif_path)

    # Apply perturb (optional)
    if sigma > 0:
        apply_sac_style_perturb(atoms, sigma=sigma, max_perturb=max_perturb)

    # Set up MACE calculator
    calc = MACECalculator(
        model_paths=[mace_model],
        head="pbe_d3",
        device=device,
        default_dtype="float32",
    )
    atoms.calc = calc

    # Log initial
    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    logger.info(f"[INIT] E0={e0:.8f} | Fmax0={np.abs(f0).max():.6f}")

    # BFGS optimize
    opt = BFGSProgress(atoms, max_steps=max_steps)
    opt.run(fmax=fmax)

    # Save final result
    write("optimized.cif", atoms)
    logger.info("[DONE] saved optimized.cif")



############################################################
# CLI
############################################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--cif", type=str, required=True)
    parser.add_argument("--model", type=str, default="mofs_v2.model")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=300)

    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--max_perturb", type=float, default=0.30)

    args = parser.parse_args()

    optimize_with_perturb(
        cif_path=args.cif,
        mace_model=args.model,
        device=args.device,
        fmax=args.fmax,
        max_steps=args.steps,
        sigma=args.sigma,
        max_perturb=args.max_perturb,
    )


