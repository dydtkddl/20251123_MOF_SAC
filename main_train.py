from sac.agent import SACAgent
from env.mof_env import MOFEnv
from utils.replay_buffer import ReplayBuffer
from logging_conf import init_logging
from tqdm import tqdm


def main():

    init_logging()

    # load MOFs
    base = load_base_structures()       # from QMOF subset
    surrogate = load_surrogate_model()  # MACE-MP-MOF0

    env = MOFEnv(base_structures=base, surrogate=surrogate)

    agent = SACAgent(
        obs_dim=env.compute_observation().shape[1],
        action_dim=3
    )

    agent.train(env, total_steps=200_000)


if __name__=="__main__":
    main()
