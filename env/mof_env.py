# env/mof_env.py

import numpy as np
from copy import deepcopy
from ase.neighborlist import neighbor_list


class MOFEnv:
    """
    MACS-MOF reinforcement environment
    Following MACS methodology:

    Observation per-atom:
        oᵢᵗ = concat(
            fᵢᵗ,
            f(nei1), ... f(nei_k),
            |r1| ... |rk|,
            r1 ... rk
        )

    Reward:
        R = log(|gᵢᵗ|) - log(|gᵢᵗ⁺¹|)
    """

    def __init__(
        self,
        atoms_loader,
        k_neighbors=12,
        cmax=0.4,
        max_steps=300,
        fmax_threshold=0.05,
    ):
        self.atoms_loader = atoms_loader
        self.k = k_neighbors
        self.cmax = cmax
        self.max_steps = max_steps
        self.fmax_threshold = fmax_threshold

        self.reset()


    ############################################################
    # reset
    ############################################################
    def reset(self):

        self.atoms = self.atoms_loader()
        self.N = len(self.atoms)

        self.forces = self.atoms.get_forces()
        self.prev_forces = np.zeros_like(self.forces)
        self.prev_disp = np.zeros_like(self.forces)

        self.covalent_radii = np.array([atm.covalent_radius for atm in self.atoms])

        self.step_count = 0

        return self._obs()


    ############################################################
    # compute neighbors
    ############################################################
    def _compute_neighbors(self):

        i, j, offsets = neighbor_list(
            "ijS",
            self.atoms,
            cutoff=6.0
        )

        rel_pos = (
            self.atoms.positions[j]
            + offsets @ self.atoms.cell
            - self.atoms.positions[i]
        )

        nei_dict = {idx: [] for idx in range(self.N)}
        for a, b, r in zip(i, j, rel_pos):
            nei_dict[a].append((b, r))

        for idx in range(self.N):
            nei_dict[idx] = sorted(nei_dict[idx], key=lambda x: np.linalg.norm(x[1]))[: self.k]

        return nei_dict


    ############################################################
    # feature fᵢᵗ
    ############################################################
    def _make_feature(self, idx):

        ri = self.covalent_radii[idx]

        gi = self.forces[idx]
        gprev = self.prev_forces[idx]

        gnorm = np.linalg.norm(gi)
        gnorm = max(gnorm, 1e-12)

        loggn = np.log(gnorm)

        cti = min(gnorm, self.cmax)

        di = self.prev_disp[idx]

        dgi = gi - gprev

        fi = np.concatenate([
            np.array([ri, cti, loggn]),
            gi,
            di,
            dgi
        ])

        return fi


    ############################################################
    # obs
    ############################################################
    def _obs(self):

        neighbors = self._compute_neighbors()

        obs_list = []

        for i in range(self.N):

            fi = self._make_feature(i)

            block = [fi]

            for (j, rel) in neighbors[i]:
                block.append(self._make_feature(j))

            for _ in range(self.k - len(neighbors[i])):
                block.append(np.zeros_like(fi))

            dists = []
            vecs = []

            for (_, rel) in neighbors[i]:
                dists.append(np.linalg.norm(rel))
                vecs.append(rel)

            for _ in range(self.k - len(neighbors[i])):
                dists.append(0.0)
                vecs.append(np.zeros(3))


            dists = np.array(dists)
            vecs = np.array(vecs)

            block.append(dists)
            block.append(vecs.reshape(-1))

            oi = np.concatenate(block)

            obs_list.append(oi)

        return np.array(obs_list)


    ############################################################
    # step
    ############################################################
    ############################################################
    def step(self, action):

        self.step_count += 1

        # ---- scaling factor cᵢᵗ ----
        gnorm = np.linalg.norm(self.forces, axis=1)
        gnorm = np.where(gnorm > 1e-12, gnorm, 1e-12)

        c = np.minimum(gnorm, self.cmax).reshape(-1,1)

        # ---- scaled displacement ----
        disp = c * action

        self.atoms.positions += disp

        # ---- compute new forces ----
        new_forces = self.atoms.get_forces()

        # ---- reward ----
        old_norm = np.linalg.norm(self.forces, axis=1)
        new_norm = np.linalg.norm(new_forces, axis=1)

        old_norm = np.where(old_norm > 1e-12, old_norm, 1e-12)
        new_norm = np.where(new_norm > 1e-12, new_norm, 1e-12)

        reward = np.log(old_norm) - np.log(new_norm)

        # ---- done ----
        done = False

        if np.mean(new_norm) < self.fmax_threshold:
            done = True

        if self.step_count >= self.max_steps:
            done = True

        # ---- update state ----
        self.prev_disp = disp.copy()
        self.prev_forces = self.forces.copy()
        self.forces = new_forces.copy()

        obs = self._obs()

        return obs, reward, done
