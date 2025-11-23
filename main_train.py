##############################
# train_mof.py (ENHANCED LOGGING)
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
# Logging config
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
POOL_DIR = "mofs/train_pool_valid"


def sample_cif():
    cifs = [
        os.path.join(POOL_DIR, f)
        for f in os.listdir(POOL_DIR)
        if f.endswith(".cif")
    ]

    assert len(cifs) > 0
    return np.random.choice(cifs)


##############################
# perturb
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
# init replay + agent
##############################
replay = ReplayBuffer(
    obs_dim=OBS_DIM,
    act_dim=ACT_DIM,
    max_size=BUFFER_SIZE,
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
# Training
##############################
logger.info(f"[MACS-MOF] start epochs={EPOCHS}")

global_start = time.time()

for ep in range(EPOCHS):

    cif = sample_cif()
    t0 = time.time()
    print(cif)
    atoms = read(cif)
    atoms = perturb(atoms, sigma=0.05)
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
    logger.info(f"================ EP {ep} ================")
    logger.info(f"CIF sample: {cif}")
    logger.info(f"N atoms: {env.N}")

    step_times = []

    for step in tqdm(range(MAX_STEPS), desc=f"[EP {ep}]"):

        step_start = time.time()

        act = agent.act(obs)

        next_obs, rew, done = env.step(act)

        # build replay
        for i in range(env.N):
            replay.store(obs[i], act[i], rew[i], next_obs[i], done)

        # RL update
        if len(replay) > agent.batch_size:
            agent.update()

        obs = next_obs
        ep_ret += np.mean(rew)

        dt = (time.time() - step_start) * 1000
        step_times.append(float(dt))

        if done:
            if np.mean(np.linalg.norm(env.forces,axis=1)) < FMAX_THRESH:
                termination = "EOS: converged"
            else:
                termination = "EOS: max step"
            break
    else:
        termination = "EOS: forced stop"


    ########################
    # EP Logging
    ########################

    mean_step_t = np.mean(step_times)
    max_step_t = np.max(step_times)

    logger.info(f"[EP {ep}] return = {ep_ret:.6f}")
    logger.info(f"[EP {ep}] replay size = {len(replay):,}")
    logger.info(f"[EP {ep}] avg reward = {ep_ret / (step+1):.6f}")
    logger.info(f"[EP {ep}] mean step ms = {mean_step_t:.3f}")
    logger.info(f"[EP {ep}] max  step ms = {max_step_t:.3f}")
    logger.info(f"[EP {ep}] termination = {termination}")


logger.info("training completed.")
logger.info(f"total walltime = {(time.time()-global_start)/3600:.3f} hr")
print("== training finished ==")
