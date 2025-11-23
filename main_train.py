# train_mof.py

import os
import numpy as np
import logging
from tqdm import tqdm

from ase.io import read
from mace.calculators import MACECalculator

from env.mof_env import MOFEnv
from sac.agent import SACAgent


###############################################
# Logging
###############################################
logging.basicConfig(
    filename="train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train")


###############################################
# MOF pool loader
###############################################
POOL_DIR = "mofs/train_pool"


def get_random_cif_path():
    cifs = [os.path.join(POOL_DIR, f) for f in os.listdir(POOL_DIR) if f.endswith(".cif")]
    assert len(cifs) > 0
    return np.random.choice(cifs)


###############################################
# perturb
###############################################
def perturb_atoms(atoms, sigma=0.05):
    pos = atoms.get_positions()
    noise = np.random.normal(0.0, sigma, pos.shape)
    atoms.set_positions(pos + noise)
    return atoms


###############################################
# create MACE surrogate loader
###############################################
calc = MACECalculator(
    model_paths=["mofs_v2.model"],
    head="pbe_d3",
    device="cuda",
    default_dtype="float64"
)


###############################################
# Training Loop
###############################################
EPOCHS = 200
MAX_STEPS = 300

logger.info(f"Training start over MOF pool, epochs={EPOCHS}")

for ep in range(EPOCHS):

    # 1) select cif
    cif_path = get_random_cif_path()
    atoms = read(cif_path)

    # 2) perturb
    atoms = perturb_atoms(atoms, sigma=0.05)

    # 3) assign surrogate
    atoms.calc = calc

    # 4) create env
    env = MOFEnv(
        atoms_loader=lambda: atoms,
        max_steps=MAX_STEPS,
        fmax_threshold=0.05
    )

    # SAC agent
    agent = SACAgent(
        n_atoms=env.N,
        obs_dim=6,
        action_dim=3,
        lr=3e-4,
        gamma=0.995,
        tau=5e-3,
        buffer_size=200000,
        batch_size=128,
        device="cuda"
    )

    logger.info(f"[EP{ep}] CIF={os.path.basename(cif_path)} N={env.N}")

    obs = env.reset()
    ep_reward = 0.0

    for _ in tqdm(range(MAX_STEPS), desc=f"[EP {ep}]"):

        action = agent.select_action(obs)

        next_obs, reward, done = env.step(action)

        agent.add_transition(obs, action, reward, next_obs, done)
        agent.learn()

        obs = next_obs
        ep_reward += np.mean(reward)

        if done:
            break

    logger.info(f"[EP{ep}] reward={ep_reward:.4f}")


logger.info("training done")
print("Training finished.")

