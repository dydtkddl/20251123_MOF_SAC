#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from logging.handlers import RotatingFileHandler
from tqdm import tqdm


# ============================================
# Logging
# ============================================
logger = logging.getLogger("energy_plot")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler("plot_energy.log", maxBytes=5_000_000, backupCount=3)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# ============================================
# argparse
# ============================================
def parse_args():
    parser = argparse.ArgumentParser(description="Plot energy convergence (twin y-axis).")
    parser.add_argument("--file", type=str, required=True,
                        help="energy file (e.g., energies.txt)")
    parser.add_argument("--save", type=str, default="energy_convergence_twin.png",
                        help="output plot filename")
    return parser.parse_args()


# ============================================
# Main
# ============================================
def main():
    args = parse_args()
    logger.info(f"Loading energy file: {args.file}")

    # ------------------------
    # Load file
    # ------------------------
    raw = []
    with open(args.file, "r") as f:
        for line in tqdm(f, desc="Reading file"):
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            raw.append([float(x) for x in parts])

    raw = np.array(raw)
    steps = raw[:, 0]
    total_e = raw[:, 1]
    per_atom_e = raw[:, 2]

    logger.info(f"Loaded {len(steps)} steps.")

    # ------------------------
    # Plot (Twin Y Axis)
    # ------------------------
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    ax1 = plt.gca()
    ax2 = ax1.twinx()   # shared x-axis, new y-axis

    # Left axis (Total Energy)
    p1, = ax1.plot(
        steps, total_e,
        color="#4C72B0", linewidth=2, label="Total Energy (E_total)"
    )

    # Right axis (Energy per atom)
    p2, = ax2.plot(
        steps, per_atom_e,
        color="#C44E52", linewidth=2, linestyle="--", label="Energy per atom (E/N)"
    )

    # Y-limit auto padding
    def auto_ylim(data):
        ymin, ymax = data.min(), data.max()
        d = (ymax - ymin) * 0.1
        return ymin - d, ymax + d

    ax1.set_ylim(*auto_ylim(total_e))
    ax2.set_ylim(*auto_ylim(per_atom_e))

    # Labels
    ax1.set_xlabel("Step", fontsize=13)
    ax1.set_ylabel("Total Energy (eV)", fontsize=13, color=p1.get_color())
    ax2.set_ylabel("Energy per atom (eV)", fontsize=13, color=p2.get_color())

    # Title
    plt.title("Energy Convergence (Twin Y Axis)", fontsize=16, pad=15)

    # Legend combine
    lines = [p1, p2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=12, loc="upper right")

    plt.tight_layout()
    plt.savefig(args.save, dpi=300)
    logger.info(f"Saved plot to {args.save}")

    plt.show()


if __name__ == "__main__":
    main()


