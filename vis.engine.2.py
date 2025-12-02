#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Super Robust Parser for MOF SAC training logs (pattern-based log loading)
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

    last_ep = None
    last_step = None

    for log_file in tqdm(log_files, desc="Parsing logs"):
        with open(log_file, "r") as f:
            lines = f.readlines()

        for line in lines:

            m_ep = p_ep.search(line)
            if m_ep:
                last_ep = int(m_ep.group(1))

            m_step = p_step.search(line)
            if m_step:
                last_step = int(m_step.group(1))

            m_ret = p_ret.search(line)
            if m_ret:
                if last_ep is not None and last_step is not None:
                    EP.append(last_ep)
                    STEP.append(last_step)
                    RET.append(float(m_ret.group(1)))

                last_ep = None
                last_step = None

    df = pd.DataFrame({
        "episode": EP,
        "step": STEP,
        "return": RET
    }).sort_values("episode")

    return df


# --------------------------------------------
# Plot (with optional ylim)
# --------------------------------------------
def plot_dual_axis(df, window=50, out_png="ep_plot.png",
                   ylim_ret=None, ylim_step=None):

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

    # Return curve
    ax1.plot(ep, rt, alpha=0.3, color="red", label="Return (raw)")
    ax1.plot(ep, rt_ma, color="darkred", linewidth=2, label=f"Return (MA{window})")
    ax1.set_ylabel("Return", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    # Apply return axis limits
    if ylim_ret is not None:
        ax1.set_ylim(ylim_ret[0], ylim_ret[1])

    # Step axis
    ax2 = ax1.twinx()
    ax2.plot(ep, st, alpha=0.3, color="blue", label="Step (raw)")
    ax2.plot(ep, st_ma, color="navy", linewidth=2, label=f"Step (MA{window})")
    ax2.set_ylabel("Terminated Step", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Apply step axis limits
    if ylim_step is not None:
        ax2.set_ylim(ylim_step[0], ylim_step[1])

    plt.title(f"Episode Return & Terminated Step (MA={window})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    logging.info(f"Saved → {out_png}")


# --------------------------------------------
# CLI
# --------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log", type=str, default="train.log*",
                        help="Log file pattern (e.g., 'train.log*', '*.log').")

    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--save", type=str, default="ep_plot.png")

    # -------- NEW: ylim options --------
    parser.add_argument("--ylim_ret", type=float, nargs=2,
                        default=[-20, 55],
                        help="Return y-axis limits: min max (default: -20 55)")

    parser.add_argument("--ylim_step", type=float, nargs=2,
                        default=[0, 1000],
                        help="Step y-axis limits: min max (default: 0 1000)")

    return parser


# --------------------------------------------
# MAIN
# --------------------------------------------
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    logs = sorted(glob(args.log))
    logging.info(f"Found {len(logs)} log files for pattern: {args.log}")

    if len(logs) == 0:
        logging.error("No logs found. Check the pattern.")
        exit(1)

    df = parse_logs(logs)
    logging.info(f"Parsed episodes: {len(df)}")

    df.to_csv("episode_step_return.csv", index=False)
    logging.info("Saved CSV → episode_step_return.csv")

    plot_dual_axis(
        df,
        window=args.window,
        out_png=args.save,
        ylim_ret=args.ylim_ret,
        ylim_step=args.ylim_step
    )

