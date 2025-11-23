import numpy as np
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii
from ase.geometry import get_distances


class MOFEnv:

    def __init__(
        self,
        atoms_loader,
        k_neighbors=12,
        cmax=0.4,
        max_steps=300,
        fmax_threshold=0.05,
        bond_break_ratio=1.8,
        k_bond=50.0,              # Soft bond penalty strength
        w_f=0.7,                  # Force reward weight
        w_e=0.3,                  # Energy reward weight
        debug_bond=False
    ):
        self.atoms_loader = atoms_loader
        self.k = k_neighbors
        self.cmax = cmax
        self.max_steps = max_steps
        self.fmax_threshold = fmax_threshold

        # Bond parameters
        self.bond_break_ratio = bond_break_ratio
        self.k_bond = k_bond
        self.debug_bond = debug_bond

        # Reward weights
        self.w_f = w_f
        self.w_e = w_e

        self.reset()


    # ============================================================
    # TRUE BOND DETECTION (PBC-SAFE)
    # ============================================================
    def _detect_true_bonds(self, atoms):

        i, j, offsets = neighbor_list("ijS", atoms, cutoff=4.0)
        pos = atoms.positions
        cell = atoms.cell

        bond_pairs = []
        bond_d0 = []

        for a, b, off in zip(i, j, offsets):
            rel = pos[b] + off @ cell - pos[a]
            d = np.linalg.norm(rel)
            rc = covalent_radii[atoms[a].number] + covalent_radii[atoms[b].number]

            if d <= rc + 0.4:
                bond_pairs.append((a, b))
                bond_d0.append(d)

        return np.array(bond_pairs, dtype=int), np.array(bond_d0, dtype=float)


    # ============================================================
    def reset(self):

        self.atoms = self.atoms_loader()
        self.N = len(self.atoms)

        self.forces = self.atoms.get_forces()

        # History
        self.prev_forces = np.zeros_like(self.forces)
        self.prev_disp = np.zeros_like(self.forces)

        # Covalent radii
        self.covalent_radii = np.array(
            [covalent_radii[a.number] for a in self.atoms]
        )

        # Bond detection
        self.bond_pairs, self.bond_d0 = self._detect_true_bonds(self.atoms)
        print(f"[INIT] Detected true bonds = {len(self.bond_pairs)}")

        # Initial energy
        self.E_prev = self.atoms.get_potential_energy()

        self.step_count = 0

        return self._obs()


    # ============================================================
    def _compute_neighbors(self):

        i, j, offsets = neighbor_list("ijS", self.atoms, cutoff=6.0)
        cell = self.atoms.cell
        pos = self.atoms.positions

        rel = pos[j] + offsets @ cell - pos[i]

        nd = {idx: [] for idx in range(self.N)}
        for a, b, r in zip(i, j, rel):
            nd[a].append((b, r))

        for idx in range(self.N):
            nd[idx] = sorted(
                nd[idx], key=lambda x: np.linalg.norm(x[1])
            )[: self.k]

        return nd


    # ============================================================
    def _make_feature(self, idx):

        ri = self.covalent_radii[idx]
        gi = self.forces[idx]
        gprev = self.prev_forces[idx]

        gnorm = max(np.linalg.norm(gi), 1e-12)

        return np.concatenate([
            np.array([ri, min(gnorm, self.cmax), np.log(gnorm)]),
            gi,
            self.prev_disp[idx],
            gi - gprev,
        ])


    # ============================================================
    def _obs(self):

        neighbors = self._compute_neighbors()
        obs_list = []

        for i in range(self.N):

            fi = self._make_feature(i)
            block = [fi]

            # neighbor features
            for (j, _) in neighbors[i]:
                block.append(self._make_feature(j))

            # zero padding
            for _ in range(self.k - len(neighbors[i])):
                block.append(np.zeros_like(fi))

            # distance + vector
            dists = []
            vecs = []

            for (_, rel) in neighbors[i]:
                dists.append(np.linalg.norm(rel))
                vecs.append(rel)

            for _ in range(self.k - len(neighbors[i])):
                dists.append(0.0)
                vecs.append(np.zeros(3))

            block.append(np.array(dists))
            block.append(np.array(vecs).reshape(-1))

            obs_list.append(np.concatenate(block))

        return np.array(obs_list, dtype=np.float32)


    # ============================================================
    #                      🔥 STEP (SOFT BONDS) 🔥
    # ============================================================
    def step(self, action):

        self.step_count += 1

        # displacement scaling
        gnorm = np.linalg.norm(self.forces, axis=1)
        gnorm = np.where(gnorm > 1e-12, gnorm, 1e-12)

        c = np.minimum(gnorm, self.cmax).reshape(-1, 1)
        disp = c * action
        self.atoms.positions += disp

        # new forces
        new_forces = self.atoms.get_forces()

        old_norm = np.maximum(np.linalg.norm(self.forces, axis=1), 1e-12)
        new_norm = np.maximum(np.linalg.norm(new_forces, axis=1), 1e-12)

        # --------------------------------------------
        # 1) Force reward
        # --------------------------------------------
        r_f = np.log(old_norm) - np.log(new_norm)

        # --------------------------------------------
        # 2) Energy reward
        # --------------------------------------------
        E_new = self.atoms.get_potential_energy()
        r_e = (self.E_prev - E_new)

        # mixed reward
        reward = self.w_f * r_f + self.w_e * r_e

        # update stored energy
        self.E_prev = E_new

        # ==============================================================  
        #       SOFT BOND PENALTY  (NO DONE)
        # ==============================================================
        cell = self.atoms.cell
        pbc = self.atoms.pbc

        for idx, (a, b) in enumerate(self.bond_pairs):

            d = get_distances(
                self.atoms.positions[a][None],
                self.atoms.positions[b][None],
                cell=cell,
                pbc=pbc
            )[1][0][0]

            d0 = self.bond_d0[idx]
            ratio = d / d0

            # stretch
            stretch = max(0.0, ratio - self.bond_break_ratio)

            # compress
            compress = max(0.0, 0.6 - ratio)

            soft_penalty = self.k_bond * (stretch**2 + compress**2)
            reward -= soft_penalty

            if self.debug_bond:
                print(f"[Bond {idx}] d0={d0:.3f}, d={d:.3f}, ratio={ratio:.3f}, penalty={soft_penalty:.3f}")

        # =============================================================
        # Termination conditions (but NOT bond-related)
        # =============================================================
        done = False

        if np.mean(new_norm) < self.fmax_threshold:
            done = True

        if self.step_count >= self.max_steps:
            done = True

        # update histories
        self.prev_disp = disp.copy()
        self.prev_forces = self.forces.copy()
        self.forces = new_forces.copy()

        return self._obs(), reward, done
