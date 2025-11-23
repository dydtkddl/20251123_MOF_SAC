##############################
# train_mof.py  (ULTRA LOG + ROTATING LOG + CHECKPOINT + PER-STEP LOG)
##############################

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


############################################################
# ROTATING LOGGING
############################################################
log_handler = RotatingFileHandler(
    "train.log",
    maxBytes=10_000_000,
    backupCount=10,
)
log_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
))
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)


############################################################
# CHECKPOINT SAVE FUNCTION
############################################################
def save_checkpoint(ep, agent, tag="auto"):
    os.makedirs("checkpoints", exist_ok=True)

    ckpt = {
        "epoch": ep,
        "actor": agent.actor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "v": agent.v.state_dict(),
        "v_tgt": agent.v_tgt.state_dict(),
        "log_alpha": float(agent.log_alpha.detach().cpu()),
    }
    p = f"checkpoints/ckpt_ep{ep:04d}_{tag}.pt"
    torch.save(ckpt, p)

    logger.info(f"[CHECKPOINT] saved => {p}")


############################################################
# pool DIR
############################################################
POOL_DIR = "mofs/train_pool_valid"


def sample_cif():
    cifs = [
        os.path.join(POOL_DIR, f)
        for f in os.listdir(POOL_DIR)
        if f.endswith(".cif")
    ]
    return np.random.choice(cifs)


############################################################
# perturb
############################################################
def perturb(a, sigma=0.05):
    p = a.get_positions()
    p += np.random.normal(0, sigma, p.shape)
    a.set_positions(p)
    return a


############################################################
# MACE surrogate
############################################################
calc = MACECalculator(
    model_paths=["mofs_v2.model"],
    head="pbe_d3",
    device="cuda",
    default_dtype="float32"
)


############################################################
# config
############################################################
EPOCHS      = 200
MAX_STEPS   = 1000
FMAX_THRESH = 0.05

BUFFER_SIZE = 3_000_000
BATCH_SIZE  = 256

OBS_DIM     = 204
ACT_DIM     = 3

CHECKPOINT_INTERVAL = 5


############################################################
# replay + agent
############################################################
replay = ReplayBuffer(obs_dim=OBS_DIM, act_dim=ACT_DIM, max_size=BUFFER_SIZE)

agent = SACAgent(
    obs_dim=OBS_DIM,
    act_dim=ACT_DIM,
    replay_buffer=replay,
    device="cuda",
    lr=3e-4,
    gamma=0.995,
    tau=5e-3,
    batch_size=BATCH_SIZE,
)


############################################################
# TRAIN
############################################################
logger.info(f"[MACS-MOF] start training EPOCHS={EPOCHS}")

global_start = time.time()


for ep in range(EPOCHS):

    cif = sample_cif()
    atoms = read(cif)
    atoms = perturb(atoms)
    atoms.calc = calc

    logger.info("")
    logger.info("="*80)
    logger.info(f"[EP {ep}] CIF={cif}")

    env = MOFEnv(
        atoms_loader=lambda: atoms,
        k_neighbors=12,
        fmax_threshold=FMAX_THRESH,
        max_steps=MAX_STEPS,
    )

    obs = env.reset()
    ep_ret = 0.0
    step_times = []


    for step in tqdm(range(MAX_STEPS), desc=f"[EP {ep}]", ncols=120):

        t0 = time.time()

        act = agent.act(obs)
        next_obs, rew, done = env.step(act)

        for i in range(env.N):
            replay.store(obs[i], act[i], rew[i], next_obs[i], done)

        if len(replay) > agent.batch_size:
            agent.update()

        obs = next_obs
        ep_ret += np.mean(rew)

        f = np.linalg.norm(env.forces,axis=1)
        f_avg = float(np.mean(f))
        f_max = float(np.max(f))
        f_min = float(np.min(f))

        step_times.append( (time.time()-t0)*1000 )


        ###########################
        # PER-STEP LOG 復元!
        ###########################
        logger.info(
            f"[EP {ep}][STEP {step}] "
            f"Natom={env.N} | "
            f"Favg={f_avg:.6f} Fmax={f_max:.6f} Fmin={f_min:.6f} | "
            f"rew={np.mean(rew):.6f} | "
            f"replay={len(replay):,} | "
            f"alpha={float(agent.alpha):.6f}"
        )


        if done:
            break


    logger.info(f"[EP {ep}] return={ep_ret:.6f}")
    logger.info(f"[EP {ep}] replay={len(replay):,}")


    # periodic checkpoint
    if ep % CHECKPOINT_INTERVAL == 0 and ep>0:
        save_checkpoint(ep, agent, tag="interval")



####################
# final checkpoint
####################
save_checkpoint(EPOCHS, agent, tag="final")


logger.info("[TRAIN DONE]")
logger.info(f"wallclock={(time.time()-global_start)/3600:.3f} hr")

print("== training finished ==")
