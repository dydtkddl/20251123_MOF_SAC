##############################
# train_mof.py  (FINAL CLEAN + STABLE + EXTENDED LOG)
##############################

import os
import time
import numpy as np
import logging
from tqdm import tqdm

from ase.io import read
from ase.data import covalent_radii
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
# Pool
##############################
POOL_DIR = "mofs/train_pool"


def sample_cif():
    cifs = [os.path.join(POOL_DIR, f)
            for f in os.listdir(POOL_DIR)
            if f.endswith(".cif")]

    assert len(cifs) > 0
    return np.random.choice(cifs)


##############################
# random perturb
##############################
def perturb(atoms, sigma=0.05):
    pos = atoms.get_positions()
    pos += np.random.normal(0, sigma, pos.shape)
    atoms.set_positions(pos)
    return atoms


##############################
# Surrogate
##############################
calc = MACECalculator(
    model_paths=["mofs_v2.model"],
    head="pbe_d3",
    device="cuda",
    default_dtype="float64"
)


##############################
# config
##############################
EPOCHS      = 200
MAX_STEPS   = 1000
FMAX_THRESH = 0.05

BUFFER_SIZE = 3_000_000
BATCH_SIZE  = 256

OBS_DIM     = 204
ACT_DIM     = 3


##############################
# replay + agent
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
# element filter
##############################
allowed = set(calc.atomic_numbers)


##############################
# TRAIN
##############################
logger.info(f"[MACS-MOF] start epochs={EPOCHS}")
global_start = time.time()

for ep in range(EPOCHS):

    cif = sample_cif()

    atoms = read(cif)

    zs = atoms.get_atomic_numbers()
    if any(z not in allowed for z in zs):
        logger.warning(f"[SKIP] unsupported element in CIF: {cif}")
        continue

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

    logger.info("")
    logger.info(f"========== EP {ep} ==========")
    logger.info(f"CIF = {cif}")
    logger.info(f"N atoms = {env.N}")

    step_times = []
    t0 = time.time()

    for step in tqdm(range(MAX_STEPS), desc=f"[EP {ep}]"):

        step_t = time.time()

        act = agent.act(obs)

        next_obs, rew, done = env.step(act)

        for i in range(env.N):
            replay.store(obs[i], act[i], rew[i], next_obs[i], done)

        if len(replay) > agent.batch_size:
            agent.update()

        obs = next_obs
        ep_ret += np.mean(rew)

        step_times.append((time.time() - step_t) * 1000)

        if done:
            break

    mean_step_t = np.mean(step_times)
    max_step_t = np.max(step_times)

    logger.info(f"[EP {ep}] return = {ep_ret:.6f}")
    logger.info(f"[EP {ep}] replay size = {len(replay):,}")
    logger.info(f"[EP {ep}] mean_step_ms = {mean_step_t:.3f}")
    logger.info(f"[EP {ep}] max_step_ms = {max_step_t:.3f}")
    logger.info(f"[EP {ep}] duration_s = {time.time() - t0:.2f}")


logger.info("training completed.")
logger.info(f"Total walltime = {(time.time()-global_start)/3600:.3f} h")

print("== training finished ==")
