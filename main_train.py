###############################################################
# main_train.py — Structure-Level MACS-SAC (Option 1 FINAL)
# Author : ChatGPT for Yongsang
#
# 핵심:
#   - ReplayBuffer에는 obs_atom만 저장
#   - actor는 flatten된 obs_global만 사용
#   - update()에서 flatten 처리
#   - env.reset(), env.step() = (obs_atom, obs_global)
###############################################################

import os
import time
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


################################################################
# LOGGING SETUP
################################################################
log_handler = RotatingFileHandler(
    "train.log", maxBytes=40_000_000, backupCount=10
)
log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)


################################################################
# SAVE CHECKPOINT
################################################################
def save_checkpoint(ep, agent, tag="auto"):
    os.makedirs("checkpoints", exist_ok=True)

    ckpt = {
        "epoch": ep,
        "actor": agent.actor.state_dict(),
        "Q1": agent.critic.Q1.state_dict(),
        "Q2": agent.critic.Q2.state_dict(),
        "v": agent.v.state_dict(),
        "v_target": agent.v_target.state_dict(),
        "log_alpha": float(agent.log_alpha.detach().cpu())
    }

    path = f"checkpoints/ckpt_{tag}_ep{ep:04d}.pt"
    torch.save(ckpt, path)
    logger.info(f"[CKPT] saved: {path}")


################################################################
# CIF SAMPLER
################################################################
POOL_DIR = "mofs/train_pool_valid"

def sample_cif():
    files = [
        os.path.join(POOL_DIR, f)
        for f in os.listdir(POOL_DIR)
        if f.endswith(".cif")
    ]
    if len(files) == 0:
        raise RuntimeError("No CIF in pool.")
    return np.random.choice(files)


################################################################
# MACE CALCULATOR
################################################################
calc = MACECalculator(
    model_paths=["mofs_v2.model"],
    head="pbe_d3",
    device="cuda",
    default_dtype="float32"
)


################################################################
# TRAIN CONFIG
################################################################
EPOCHS = 1500
BASE_STEPS = 200
FINAL_STEPS = 900
HORIZON_SCH = 500

FMAX_THRESH = 0.05
BUFFER_SIZE = 300_000

#BUFFER_SIZE = 4_000_000
BATCH_SIZE = 256
CKPT_INTERVAL = 10


logger.info("====== Structure-Level MACS-SAC Training Start ======")

global_start = time.time()

agent = None
replay = None

OBS_GLOBAL_DIM = None
ACT_GLOBAL_DIM = None



################################################################
# TRAIN LOOP
################################################################
for ep in range(EPOCHS):

    logger.info("\n" + "="*120)
    logger.info(f"[EP {ep}] START")

    ################################################################
    # Curriculum schedule
    ################################################################
    r = min(ep / HORIZON_SCH, 1.0)
    max_steps = int(BASE_STEPS + (FINAL_STEPS - BASE_STEPS) * r)
    logger.info(f"[EP {ep}] max_steps = {max_steps}")

    ################################################################
    # CIF load
    ################################################################
    cif = sample_cif()
    atoms = read(cif)
    atoms.calc = calc

    logger.info(f"[EP {ep}] CIF = {cif}")
    logger.info(f"[EP {ep}] N_atoms = {len(atoms)}")

    def loader():
        a = atoms.copy()
        a.calc = calc
        return a

    ################################################################
    # Env
    ################################################################
    env = MOFEnv(
        atoms_loader=loader,
        max_steps=max_steps,
        fmax_threshold=FMAX_THRESH,
        k_neighbors=12,
        disp_scale=0.03
    )

    ################################################################
    # Reset → (obs_atom, obs_global)
    ################################################################
    obs_atom, obs_global = env.reset()

    N_atom = obs_atom.shape[0]
    FEAT = obs_atom.shape[1]
    OBS_DIM = obs_global.shape[0]

    logger.info(f"[EP {ep}] FEAT={FEAT}, OBS_GLOBAL_DIM={OBS_DIM}")

    ################################################################
    # INIT (first episode)
    ################################################################
    if agent is None:

        OBS_GLOBAL_DIM = OBS_DIM
        ACT_GLOBAL_DIM = 3 * N_atom

        replay = ReplayBuffer(
            n_atoms=N_atom,
            feat_dim=FEAT,
            act_global_dim=ACT_GLOBAL_DIM,   # = 3*N_atom
            max_size=BUFFER_SIZE,
            n_step=1,
            gamma=0.995,
            alpha=0.6,
            beta=0.4
        )


        agent = SACAgent(
            obs_global_dim=OBS_GLOBAL_DIM,
            act_global_dim=ACT_GLOBAL_DIM,
            replay_buffer=replay,
            gamma=0.995,
            tau=5e-3,
            batch_size=BATCH_SIZE,
            lr=3e-4,
            device="cuda"
        )

        logger.info("[INIT] SACAgent + ReplayBuffer initialized.")

    else:
        # dimension mismatch — skip episode
        if OBS_DIM != OBS_GLOBAL_DIM:
            logger.warning(
                f"[EP {ep}] OBS_GLOBAL_DIM mismatch: expected={OBS_GLOBAL_DIM}, got={OBS_DIM} => SKIP"
            )
            continue

    ################################################################
    # EPISODE START
    ################################################################
    replay.new_episode()

    ep_ret = 0.0
    done = False
    done_reason = None

    tqdm_bar = tqdm(range(max_steps), desc=f"[EP {ep}]", ncols=140)

    for step in tqdm_bar:

        ################################################################
        # 1) STRUCTURE-LEVEL ACTION
        ################################################################
        action_global = agent.act(obs_global)     # (3N,)
        action_arr = action_global.reshape(N_atom, 3)

        ################################################################
        # 2) ENV STEP
        ################################################################
        next_atom, next_global, reward_scalar, done, done_reason, Etot, Fmax = env.step(action_arr)

        ################################################################
        # 3) STORE — OPTION1: obs_atom/no flatten
        ################################################################
        replay.store(
            obs_atom,
            action_global,
            reward_scalar,
            next_atom,
            done
        )


        ################################################################
        # 4) LOGGING
        ################################################################
        alpha_val = float(agent.alpha.detach().cpu())
        tqdm_bar.set_postfix({
            "Fmax": f"{Fmax:.2e}",
            "reward": f"{reward_scalar:.4f}",
            "alpha": f"{alpha_val:.4f}"
        })

        ep_ret += reward_scalar
        obs_atom = next_atom
        obs_global = next_global  # flatten for actor

        if done:
            logger.info(f"[EP {ep}] DONE at step {step}, reason={done_reason}")
            break


    ################################################################
    # EPISODE END
    ################################################################
    logger.info(f"[EP {ep}] return = {ep_ret:.6f}")

    if done_reason in ["com", "bond"]:
        replay.end_episode(keep=False)
        logger.info(f"[EP {ep}] BAD EPISODE → removed")
    else:
        replay.end_episode(keep=True)


    ################################################################
    # SAC UPDATE
    ################################################################
    if len(replay) > agent.batch_size:
        info = agent.update()
        logger.info(
            "[UPDATE] "
            f"policy={info['policy_loss']} | "
            f"q1={info['q1_loss']:.6f} | "
            f"q2={info['q2_loss']:.6f} | "
            f"v={info['v_loss']:.6f} | "
            f"alpha={info['alpha']:.5f}"
        )


    ################################################################
    # PERIODIC CHECKPOINT
    ################################################################
    if ep % CKPT_INTERVAL == 0 and ep > 0:
        save_checkpoint(ep, agent, tag="interval")


################################################################
# FINAL SAVE
################################################################
save_checkpoint(EPOCHS, agent, tag="final")

logger.info(
    f"==== TRAIN COMPLETE (elapsed={(time.time() - global_start)/3600:.3f} hr) ===="
)
print("== TRAIN DONE ==")
