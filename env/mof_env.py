import numpy as np
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii


class MOFEnv:
    """
    MACS-MOF environment

    obs per atom = concat(
        fᵢᵗ,
        f(nei1)...f(nei_k),
        |r1|...|rk|,
        r1...rk
    )

    reward_i = log(|gᵢᵗ|) - log(|gᵢᵗ⁺¹|)
    """


    def __init__(
        self,
        atoms_loader,
        k_neighbors=12,
        cmax=0.4,
        max_steps=300,
        fmax_threshold=0.05,
        bond_break_ratio=1.8,    # 🔥 bond break threshold
        bond_penalty=100.0       # 🔥 reward penalty
    ):
        self.atoms_loader = atoms_loader
        self.k = k_neighbors
        self.cmax = cmax
        self.max_steps = max_steps
        self.fmax_threshold = fmax_threshold

        # bond parameters
        self.bond_break_ratio = bond_break_ratio
        self.bond_penalty = bond_penalty

        self.reset()


    ############################################################
    def reset(self):

        self.atoms = self.atoms_loader()
        self.N = len(self.atoms)

        # initial forces
        self.forces = self.atoms.get_forces()

        # history
        self.prev_forces = np.zeros_like(self.forces)
        self.prev_disp = np.zeros_like(self.forces)

        # covalent radii
        self.covalent_radii = np.array(
            [covalent_radii[a.number] for a in self.atoms]
        )

        # ================================================
        # 🔥 save initial bond distances to prevent breakage
        # ================================================
        i, j, offsets = neighbor_list("ijS", self.atoms, cutoff=3.0)
        pos = self.atoms.positions
        rel = pos[j] + offsets @ self.atoms.cell - pos[i]
        d0 = np.linalg.norm(rel, axis=1)

        self.bond_pairs = np.stack([i, j], axis=1)
        self.bond_d0 = d0

        self.step_count = 0

        return self._obs()


    ############################################################
    def _compute_neighbors(self):

        i, j, offsets = neighbor_list("ijS", self.atoms, cutoff=6.0)

        rel = (
            self.atoms.positions[j]
            + offsets @ self.atoms.cell
            - self.atoms.positions[i]
        )

        nd = {idx: [] for idx in range(self.N)}
        for a, b, r in zip(i, j, rel):
            nd[a].append((b, r))

        for idx in range(self.N):
            nd[idx] = sorted(
                nd[idx], key=lambda x: np.linalg.norm(x[1])
            )[: self.k]

        return nd


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
            dgi,
        ])

        return fi


    ############################################################
    def _obs(self):

        neighbors = self._compute_neighbors()
        obs_list = []

        for i in range(self.N):

            fi = self._make_feature(i)
            block = [fi]

            # fk neighbors
            for (j, rel) in neighbors[i]:
                block.append(self._make_feature(j))

            # padding neighbors (missing)
            for _ in range(self.k - len(neighbors[i])):
                block.append(np.zeros_like(fi))

            # distances + vectors
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

        return np.array(obs_list, dtype=np.float32)


    ############################################################
    def step(self, action):

        self.step_count += 1

        # compute cᵢᵗ
        gnorm = np.linalg.norm(self.forces, axis=1)
        gnorm = np.where(gnorm > 1e-12, gnorm, 1e-12)
        c = np.minimum(gnorm, self.cmax).reshape(-1, 1)

        # scaled displacement
        disp = c * action

        self.atoms.positions += disp

        # new forces
        new_forces = self.atoms.get_forces()

        old_norm = np.linalg.norm(self.forces, axis=1)
        new_norm = np.linalg.norm(new_forces, axis=1)

        old_norm = np.where(old_norm > 1e-12, old_norm, 1e-12)
        new_norm = np.where(new_norm > 1e-12, new_norm, 1e-12)

        reward = np.log(old_norm) - np.log(new_norm)

        done = False

        # =====================================================
        # 🔥 bond length check → penalty + done
        # =====================================================
        pos = self.atoms.positions
        broken = False

        for idx, (a, b) in enumerate(self.bond_pairs):
            d0 = self.bond_d0[idx]
            d = np.linalg.norm(pos[a] - pos[b])

            if d > self.bond_break_ratio * d0:
                reward -= self.bond_penalty
                broken = True
            if d < 0.6 * d0:
                reward -= self.bond_penalty
                broken = True
        if broken:
            done = True


        # termination by force threshold
        if np.mean(new_norm) < self.fmax_threshold:
            done = True

        if self.step_count >= self.max_steps:
            done = True

        # update histories
        self.prev_disp = disp.copy()
        self.prev_forces = self.forces.copy()
        self.forces = new_forces.copy()

        obs = self._obs()

        return obs, reward, done
