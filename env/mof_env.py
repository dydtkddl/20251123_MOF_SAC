# env/mof_env.py

import numpy as np
from copy import deepcopy


class MOFEnv:
    """
    MACS-MOF reinforcement environment
    with ASE + MACE surrogate force provider
    """

    def __init__(
        self,
        atoms_loader,        # returns ASE Atoms
        max_steps=300,
        fmax_threshold=0.05, # force convergence
    ):
        
        self.atoms_loader = atoms_loader
        self.max_steps = max_steps
        self.fmax_threshold = fmax_threshold
        
        self.reset()


    # ==================================================
    # reset
    # ==================================================
    def reset(self):

        self.atoms = self.atoms_loader()
        self.N = len(self.atoms)

        # evaluate first force
        self.forces = self.atoms.get_forces()

        self.prev_actions = np.zeros_like(self.forces)

        self.step_count = 0

        return self._obs()


    # ==================================================
    # make observation per-atom
    # ==================================================
    def _obs(self):

        # you can expand later to full MACS-style
        obs = np.concatenate([
            self.forces,            # (N,3)
            self.prev_actions,      # (N,3)
        ], axis=1)

        # (N,6)
        return obs


    # ==================================================
    # step
    # ==================================================
    def step(self, action):
        """
        action: (N,3)
        """

        self.step_count += 1

        # apply displacement
        self.atoms.positions += action

        # evaluate surrogate forces
        new_forces = self.atoms.get_forces()

        # reward = force magnitude reduction
        reward = -np.linalg.norm(new_forces,axis=1)

        done = False

        if np.mean(np.linalg.norm(new_forces,axis=1)) < self.fmax_threshold:
            done = True

        if self.step_count >= self.max_steps:
            done = True

        # update state
        self.prev_actions = action.copy()
        self.forces = new_forces.copy()

        obs = self._obs()

        return obs, reward, done


