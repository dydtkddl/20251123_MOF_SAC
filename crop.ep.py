#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Episode Length (Steps) vs Episode Index
=============================================
- 자동으로 EP 디렉토리를 감지
- energies.txt 읽어서 step count 계산
- raw curve 연하게
- moving average curve 진하게
- PNG 저장
- (optional) CSV 저장

사용법:
$ python plot_steps_vs_episode.py
(snapshots_phase2 폴더 내부 또는 상위 폴더 어디에서든)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

# optional smoothing
try:
    from scipy.ndimage import gaussian_filter1d
    USE_SMOOTH = True
except ImportError:
    USE_SMOOTH = False


# ---------------------------------------------------------
# auto-detect EP folders
# ---------------------------------------------------------
def detect_base_dir():

    # Case 1: current dir
    eps = glob("./EP*")
    if len(eps) > 0:
        print(f"[AUTO] Using current dir: {os.getcwd()}")
        return "."

    # Case 2: parent dir
    parent = os.path.dirname(os.getcwd())
    eps = glob(os.path.join(parent, "EP*"))
    if len(eps) > 0:
        print(f"[AUTO] Using parent dir: {parent}")
        return parent

    # no EP found
    raise RuntimeError(
        "EP* folders not found in current or parent directory.\n"
        "Run this script in or above snapshots_phase2."
    )


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main():

    BASE_DIR = detect_base_dir()

    ep_dirs = sorted(glob(os.path.join(BASE_DIR, "EP*")))
    print(f"[INFO] Found {len(ep_dirs)} episodes")

    if len(ep_dirs) == 0:
        print("[ERROR] No episodes found. exit.")
        return

    ep_ids = []
    step_counts = []

    for ep in tqdm(ep_dirs, desc="Parsing", ncols=100):

        name = os.path.basename(ep)
        ep_id = int(name.replace("EP", ""))

        energy_file = os.path.join(ep, "energies.txt")

        if not os.path.exists(energy_file):
            print(f"[WARN] Missing: {energy_file}")
            continue

        # robust load
        try:
            data = np.loadtxt(energy_file)
        except Exception as e:
            print(f"[WARN] Failed to parse {energy_file}: {e}")
            continue

        # data handling
        if data.ndim == 1:
            # one row only
            steps = [data[0]]
        else:
            steps = data[:, 0]

        # convert (step index max +1)
        step_count = int(np.max(steps)) + 1

        ep_ids.append(ep_id)
        step_counts.append(step_count)

    # ---------------------------------------------------------
    # convert to arrays
    # ---------------------------------------------------------
    ep_ids = np.array(ep_ids, dtype=int)
    step_counts = np.array(step_counts, dtype=float)

    # sort by EP index just in case
    sort_idx = np.argsort(ep_ids)
    ep_ids = ep_ids[sort_idx]
    step_counts = step_counts[sort_idx]

    # ---------------------------------------------------------
    # smoothing
    # ---------------------------------------------------------
    if USE_SMOOTH:
        smooth = gaussian_filter1d(step_counts, sigma=3)
    else:
        # fallback: rolling mean
        w = 10
        smooth = np.convolve(step_counts, np.ones(w)/w, mode="same")

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(11,6))

    # raw curve (연하게)
    plt.plot(
        ep_ids,
        step_counts,
        color="steelblue",
        alpha=0.3,
        linewidth=1.0,
        label="raw steps"
    )

    # smooth curve (짙게)
    plt.plot(
        ep_ids,
        smooth,
        color="steelblue",
        alpha=1.0,
        linewidth=3.0,
        label="smoothed (moving average)"
    )

    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Steps per Episode", fontsize=14)
    plt.title("Episode Length vs RL Progression", fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # ---------------------------------------------------------
    # save figure
    # ---------------------------------------------------------
    out_png = "steps_vs_episode.png"
    plt.savefig(out_png, dpi=200)
    print(f"[SAVE] {out_png}")

    # no plt.show() (non-interactive)

    # ---------------------------------------------------------
    # save csv (optional)
    # ---------------------------------------------------------
    out_csv = "steps_vs_episode.csv"
    with open(out_csv, "w") as f:
        f.write("episode,steps\n")
        for ep, s in zip(ep_ids, step_counts):
            f.write(f"{ep},{s}\n")

    print(f"[SAVE] {out_csv}")
    print("[DONE]")


# ---------------------------------------------------------
# entry
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
