import numpy as np
from ase.data import covalent_radii
from ase.geometry import find_mic


###############################################################
# Enhanced MOFEnv (Action = scale × (-normalized_force))
# Fully compatible with main_train.py (6-return version)
###############################################################
class MOFEnv:

    def __init__(
        self,
        atoms_loader,
        max_steps=300,
        disp_scale=0.03,
        fmax_threshold=0.05,
        com_threshold=0.30,
        com_lambda=10.0,
        bond_break_ratio=2.4,
        bond_lambda=5.0,
        w_force=1.0,
        w_energy=0.1,
    ):
        self.loader = atoms_loader
        self.max_steps = max_steps
        self.disp_scale = disp_scale
        self.fmax_threshold = fmax_threshold

        self.com_threshold = com_threshold
        self.com_lambda = com_lambda

        self.bond_break_ratio = bond_break_ratio
        self.bond_lambda = bond_lambda

        self.w_force = w_force
        self.w_energy = w_energy

        self.reset()


    ##################################################################
    # Utility: PBC-aware vector
    ##################################################################
    def _rel_vec(self, i, j):
        pos = self.atoms.positions
        cell = self.atoms.cell.array
        raw = pos[j] - pos[i]
        frac = np.linalg.solve(cell.T, raw)
        frac -= np.round(frac)
        return frac @ cell


    ##################################################################
    # Detect true bonds using covalent radii
    ##################################################################
    def _detect_bonds(self):
        Z = self.atomic_numbers
        N = self.N

        bonds = []
        d0 = []

        for i in range(N):
            for j in range(i + 1, N):
                rc = covalent_radii[Z[i]] + covalent_radii[Z[j]] + 0.3
                d = np.linalg.norm(self._rel_vec(i, j))
                if d <= rc:
                    bonds.append((i, j))
                    d0.append(d)

        return np.array(bonds), np.array(d0)


    ##################################################################
    # Coordination number
    ##################################################################
    def _coordination_numbers(self):
        CN = np.zeros(self.N)
        for (i, j) in self.bond_pairs:
            CN[i] += 1
            CN[j] += 1
        return CN


    ##################################################################
    # Local planarity proxy
    ##################################################################
    def _local_planarity(self):
        pos = self.atoms.positions
        planar = np.zeros(self.N)

        for i in range(self.N):
            neigh = np.where(self.bond_pairs[:,0] == i)[0].tolist() + \
                    np.where(self.bond_pairs[:,1] == i)[0].tolist()

            neigh_atoms = set()
            for idx in neigh:
                a, b = self.bond_pairs[idx]
                neigh_atoms.add(a)
                neigh_atoms.add(b)
            neigh_atoms.discard(i)
            neigh_atoms = list(neigh_atoms)

            if len(neigh_atoms) < 3:
                continue

            a, b, c = neigh_atoms[:3]
            v1 = pos[a] - pos[i]
            v2 = pos[b] - pos[i]
            v3 = pos[c] - pos[i]

            n1 = np.cross(v1, v2)
            n2 = np.cross(v1, v3)

            if np.linalg.norm(n1) < 1e-8 or np.linalg.norm(n2) < 1e-8:
                continue

            cosang = np.dot(n1, n2) / (np.linalg.norm(n1)*np.linalg.norm(n2))
            planar[i] = abs(cosang)

        return planar


    ##################################################################
    # Graph radius proxy
    ##################################################################
    def _local_graph_radius(self):
        R = np.zeros(self.N)
        for (i, j) in self.bond_pairs:
            d = np.linalg.norm(self._rel_vec(i, j))
            R[i] += d
            R[j] += d
        return R


    ##################################################################
    # Stiffness proxy: |F| / |disp_last|
    ##################################################################
    def _local_stiffness(self):
        if self.disp_last is None:
            return np.zeros(self.N)

        disp_mag = np.linalg.norm(self.disp_last, axis=1) + 1e-12
        force_mag = np.linalg.norm(self.forces, axis=1)

        return force_mag / disp_mag


    ##################################################################
    # Torsion angle approx
    ##################################################################
    def _torsion_feature(self):
        pos = self.atoms.positions
        tors = np.zeros(self.N)

        for i in range(self.N):
            neigh = np.where(self.bond_pairs[:,0] == i)[0].tolist() + \
                    np.where(self.bond_pairs[:,1] == i)[0].tolist()

            neigh_atoms = set()
            for idx in neigh:
                a, b = self.bond_pairs[idx]
                neigh_atoms.add(a)
                neigh_atoms.add(b)
            neigh_atoms.discard(i)
            neigh_atoms = list(neigh_atoms)

            if len(neigh_atoms) < 3:
                continue

            a, b, c = neigh_atoms[:3]
            p0, p1, p2, p3 = pos[a], pos[i], pos[b], pos[c]

            b0 = -(p1 - p0)
            b1 = (p2 - p1)
            b2 = (p3 - p2)

            n1 = np.cross(b0, b1)
            n2 = np.cross(b1, b2)

            if np.linalg.norm(n1) < 1e-8 or np.linalg.norm(n2) < 1e-8:
                continue

            x = np.dot(n1, n2)
            y = np.dot(np.cross(n1, n2), b1 / (np.linalg.norm(b1)+1e-12))
            tors[i] = np.arctan2(y, x)

        return tors


    ##################################################################
    # Local stress proxy
    ##################################################################
    def _local_stress(self, CN):
        force_mag = np.linalg.norm(self.forces, axis=1)
        return force_mag * CN


    ##################################################################
    # SBU group ID
    ##################################################################
    def _sbu_id(self):
        Z = self.atomic_numbers
        metals = set([20,22,23,24,25,26,27,28,29,40,42,44])
        sbu = np.zeros(self.N)

        cid = 1
        for i in range(self.N):
            if Z[i] in metals:
                sbu[i] = cid
                cid += 1

        return sbu


    ##################################################################
    # Observation vector
    ##################################################################
    def _obs(self):
        F = self.forces
        f_norm = np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
        f_unit = F / f_norm

        CN = self._coordination_numbers().reshape(-1,1)
        planar = self._local_planarity().reshape(-1,1)
        graphR = self._local_graph_radius().reshape(-1,1)
        stiff = self._local_stiffness().reshape(-1,1)
        tors = self._torsion_feature().reshape(-1,1)
        stress = self._local_stress(CN.flatten()).reshape(-1,1)
        sbu = self._sbu_id().reshape(-1,1)

        global_force_mean = np.mean(f_norm)
        global_force_std = np.std(f_norm)

        global_f = np.full((self.N, 2), 
                           [global_force_mean, global_force_std],
                           dtype=np.float32)

        energy_norm = np.full((self.N,1), self.energy, np.float32)

        obs = np.concatenate([
            f_unit,        # 3
            f_norm,        # 1
            CN,            # 1
            planar,        # 1
            graphR,        # 1
            stiff,         # 1
            tors,          # 1
            stress,        # 1
            sbu,           # 1
            energy_norm,   # 1
            global_f       # 2
        ], axis=1)

        return obs.astype(np.float32)


    ##################################################################
    # Reset
    ##################################################################
    def reset(self):
        self.atoms = self.loader()
        self.N = len(self.atoms)
        self.atomic_numbers = np.array([a.number for a in self.atoms])

        self.forces = self.atoms.get_forces().astype(np.float32)
        self.energy = float(self.atoms.get_potential_energy())

        self.bond_pairs, self.bond_d0 = self._detect_bonds()

        self.com_prev = self.atoms.positions.mean(axis=0)
        self.disp_last = None
        self.step_count = 0

        return self._obs()


    ##################################################################
    # Step  — returns 6 outputs required by main_train.py
    ##################################################################
    def step(self, action_scale):

        self.step_count += 1

        scale = np.clip(action_scale, 0.0, 1.0)

        F = self.forces
        Fnorm = np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
        Funit = F / Fnorm

        disp = -self.disp_scale * scale * Funit
        self.disp_last = disp.copy()

        self.atoms.positions += disp

        # Re evaluate
        new_forces = self.atoms.get_forces().astype(np.float32)
        new_energy = float(self.atoms.get_potential_energy())

        # Reward
        old_norm = np.linalg.norm(F, axis=1)
        new_norm = np.linalg.norm(new_forces, axis=1)

        r_force = (np.log(old_norm+1e-12) - np.log(new_norm+1e-12))
        r_force_mean = float(np.mean(r_force))

        r_energy = (self.energy - new_energy)

        # reward is scalar → expand to per-atom vector
        reward_scalar = self.w_force * r_force_mean + self.w_energy * r_energy
        reward_vec = np.full(self.N, reward_scalar, dtype=np.float32)

        # COM drift
        com_new = self.atoms.positions.mean(axis=0)
        delta_com = np.linalg.norm(com_new - self.com_prev)
        reward_vec -= self.com_lambda * delta_com

        self.com_prev = com_new.copy()

        if delta_com > self.com_threshold:
            return self._obs(), reward_vec, True, "com", new_energy, float(np.max(new_norm))

        # Bond break check
        for k, (i, j) in enumerate(self.bond_pairs):
            r = np.linalg.norm(self._rel_vec(i,j))
            r0 = self.bond_d0[k]
            ratio = r/r0
            if ratio > self.bond_break_ratio:
                over = ratio - self.bond_break_ratio
                reward_vec -= self.bond_lambda * over
                return self._obs(), reward_vec, True, "bond", new_energy, float(np.max(new_norm))

        # Convergence
        Fmax = float(np.max(new_norm))
        if Fmax < self.fmax_threshold:
            self.energy = new_energy
            self.forces = new_forces
            return self._obs(), reward_vec, True, "fmax", new_energy, Fmax

        # Max steps
        if self.step_count >= self.max_steps:
            self.energy = new_energy
            self.forces = new_forces
            return self._obs(), reward_vec, True, "max_steps", new_energy, Fmax

        # Normal update
        self.energy = new_energy
        self.forces = new_forces

        return self._obs(), reward_vec, False, None, new_energy, Fmax
