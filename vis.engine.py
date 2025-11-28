#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Super Robust Parser for MOF SAC training logs
- Detect EP, step, return even if all appear on one line
- Detect even if EP/step order varies
- No dependency on line ordering
- Dual-axis plot
"""

import re
import os
import logging
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# --------------------------------------------
# Moving average
# --------------------------------------------
def moving_average(x, w=50):
    if len(x) == 0:
        return np.array([])
    if len(x) < w:
        w = len(x)
    return np.convolve(x, np.ones(w)/w, mode='same')


# --------------------------------------------
# SUPER ROBUST PARSER
# --------------------------------------------
def parse_logs(log_files):

    EP, STEP, RET = [], [], []

    p_ep = re.compile(r"\[EP\s+(\d+)\]")
    p_step = re.compile(r"step=(\d+)")
    p_ret = re.compile(r"return=([-+]?\d*\.\d+|\d+)")

    # 캐쉬: 가장 최근 발견한 EP, STEP (return 나올 때 flush)
    last_ep = None
    last_step = None

    for log_file in tqdm(log_files, desc="Parsing logs"):
        with open(log_file, "r") as f:
            lines = f.readlines()

        for line in lines:

            # 1) EP 감지
            m_ep = p_ep.search(line)
            if m_ep:
                last_ep = int(m_ep.group(1))

            # 2) step 감지
            m_step = p_step.search(line)
            if m_step:
                last_step = int(m_step.group(1))

            # 3) return 감지 (trigger point)
            m_ret = p_ret.search(line)
            if m_ret:
                if last_ep is not None and last_step is not None:
                    EP.append(last_ep)
                    STEP.append(last_step)
                    RET.append(float(m_ret.group(1)))

                # reset for next episode
                last_ep = None
                last_step = None

    df = pd.DataFrame({
        "episode": EP,
        "step": STEP,
        "return": RET
    }).sort_values("episode")

    return df


# --------------------------------------------
# Plotting
# --------------------------------------------
def plot_dual_axis(df, window=50, out_png="ep_plot.png"):

    if len(df) == 0:
        logging.warning("No episode data → skip plot")
        return

    ep = df["episode"].values
    st = df["step"].values
    rt = df["return"].values

    st_ma = moving_average(st, w=window)
    rt_ma = moving_average(rt, w=window)

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()

    # Return
    ax1.plot(ep, rt, alpha=0.3, color="red", label="Return (raw)")
    ax1.plot(ep, rt_ma, color="darkred", linewidth=2, label=f"Return (MA{window})")
    ax1.set_ylabel("Return", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    # Step
    ax2 = ax1.twinx()
    ax2.plot(ep, st, alpha=0.3, color="blue", label="Step (raw)")
    ax2.plot(ep, st_ma, color="navy", linewidth=2, label=f"Step (MA{window})")
    ax2.set_ylabel("Terminated Step", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    plt.title(f"Episode Return & Terminated Step (Moving Avg={window})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    logging.info(f"Saved → {out_png}")


# --------------------------------------------
# CLI
# --------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--save", type=str, default="ep_plot.png")
    return parser


# --------------------------------------------
# MAIN
# --------------------------------------------
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    logs = sorted(glob("train.log*"))
    logging.info(f"Found {len(logs)} log files")

    df = parse_logs(logs)
    logging.info(f"Parsed episodes: {len(df)}")

    df.to_csv("episode_step_return.csv", index=False)
    logging.info("Saved CSV → episode_step_return.csv")

    plot_dual_axis(df, window=args.window, out_png=args.save)
