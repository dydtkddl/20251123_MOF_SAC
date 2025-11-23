##############################
#  train_mof.py (FINAL)
##############################

import os
import time
import numpy as np
import logging
from tqdm import tqdm

from ase.io import read
from mace.calculators import MACECalculator

from env.mof_env import MOFEnv
from sac.agent import SACAgent
from utils.replay_buffer import ReplayBuffer


##############################
# Logging
##############################
logging.basicConfig(
    filename="train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train")


##############################
# MOF dataset pool
##############################
POOL_DIR = "mofs/train_pool"


def sample_cif():
    cifs = [
        os.path.join(POOL_DIR, f)
        for f in os.listdir(POOL_DIR)
        if f.endswith(".cif")
    ]
    assert len(cifs) > 0
    return np.random.choice(cifs)


##############################
# Random perturbation
##############################
def perturb(atoms, sigma=0.05):
    pos = atoms.get_positions()
    pos += np.random.normal(0, sigma, pos.shape)
    atoms.set_positions(pos)
    return atoms


##############################
# MACE surrogate
##############################
calc = MACECalculator(
    model_paths=["mofs_v2.model"],
    head="pbe_d3",
    device="cuda",
    default_dtype="float64"
)


##############################
# Config
##############################
EPOCHS      = 200
MAX_STEPS   = 1000
FMAX_THRESH = 0.05

BUFFER_SIZE = 3_000_000
BATCH_SIZE  = 256

OBS_DIM     = 204
ACT_DIM     = 3


##############################
# Replay + Agent
##############################
replay = ReplayBuffer(
    obs_dim=OBS_DIM,
    act_dim=ACT_DIM,
    max_size=BUFFER_SIZE
)

agent = SACAgent(
    obs_dim=OBS_DIM,
    act_dim=ACT_DIM,
    replay_buffer=replay,
    lr=3e-4,
    gamma=0.995,
    tau=5e-3,
    batch_size=BATCH_SIZE,
    device="cuda"
)


##############################
# Allowed element set
##############################
try:
    allowed = set(calc.model.atomic_numbers)
except:
    from mace.tools.z_table import atomic_numbers
    allowed = set(atomic_numbers)

##############################
# TRAIN
##############################
logger.info(f"[START] epochs={EPOCHS}")
train_start = time.time()


for ep in range(EPOCHS):

    cif = sample_cif()
    atoms = read(cif)

    # filter unsupported elements
    zs = atoms.get_atomic_numbers()

    if any(z not in allowed for z in zs):
        logger.warning(f"[EP {ep}] SKIPPED unsupported elements CIF={cif}")
        continue

    # perturb
    atoms = perturb(atoms)
    atoms.calc = calc

    env = MOFEnv(
        atoms_loader=lambda: atoms,
        k_neighbors=12,
        fmax_threshold=FMAX_THRESH,
        max_steps=MAX_STEPS,
    )

    obs = env.reset()
    ep_ret = 0.0
    step_times = []

    logger.info("")
    logger.info(f"========== EP {ep} ==========")
    logger.info(f"CIF = {cif}")
    logger.info(f"N atoms = {env.N}")


    ep_start = time.time()

    for step in tqdm(range(MAX_STEPS), desc=f"[EP {ep}]"):

        s = time.time()

        act = agent.act(obs)
        next_obs, rew, done = env.step(act)

        for i in range(env.N):
            replay.store(obs[i], act[i], rew[i], next_obs[i], done)

        if len(replay) > agent.batch_size:
            agent.update()

        obs = next_obs
        ep_ret += np.mean(rew)

        step_times.append((time.time() - s) * 1000)

        if done:
            break


    logger.info(f"[EP {ep}] return={ep_ret:.6f}")
    logger.info(f"[EP {ep}] replay size={len(replay):,}")
    logger.info(f"[EP {ep}] mean_step_ms={np.mean(step_times):.3f}")
    logger.info(f"[EP {ep}] duration_s={time.time()-ep_start:.2f}")


logger.info("== TRAINING DONE ==")
logger.info(f"total time = {(time.time()-train_start)/3600:.3f} hr")

print("== training finished ==")
