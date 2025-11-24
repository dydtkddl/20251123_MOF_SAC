###############################################################
# env/mof_env.py — HYBRID-MACS + FULL PBC GEOMETRY VERSION
# All geometric / chemical / structural features PBC-corrected
# Full hybrid feature: f_i = f_j identical
# Reward = Δlog|F| (per atom)
# Exact MACS-style kNN (PBC-aware, fast)
###############################################################

import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list


class MOFEnv:

    ###################################################################
    # Constructor
    ###################################################################
    def __init__(
        self,
        atoms_loader,
        max_steps=300,
        disp_scale=0.03,
        fmax_threshold=0.05,

        com_threshold=0.25,
        com_lambda=4.0,
        bond_break_ratio=2.4,
        bond_lambda=2.0,

        w_force=1.0,

        k_neighbors=12,
        cutoff_factor=0.8
    ):
        self.loader = atoms_loader
        self.max_steps = max_steps

        self.base_disp_scale = disp_scale
        self.fmax_threshold = fmax_threshold

        self.com_threshold = com_threshold
        self.com_lambda = com_lambda
        self.bond_break_ratio = bond_break_ratio
        self.bond_lambda = bond_lambda

        self.w_force = w_force
        self.k = k_neighbors
        self.cutoff_factor = cutoff_factor

        self.reset()


    ###################################################################
    # PBC minimum-image helpers
    ###################################################################
    def _pbc_vec(self, i, j):
        """Return minimum image vector from atom i → j."""
        pos = self.atoms.positions
        cell = self.atoms.cell.array

        diff = pos[j] - pos[i]
        frac = np.linalg.solve(cell.T, diff)
        frac -= np.round(frac)
        return frac @ cell

    def _pbc_vec_pos(self, p_i, p_j):
        """Minimum-image for two raw positions."""
        cell = self.atoms.cell.array
        diff = p_j - p_i
        frac = np.linalg.solve(cell.T, diff)
        frac -= np.round(frac)
        return frac @ cell


    ###################################################################
    # MACS-style kNN (fast, PBC-corrected)
    ###################################################################
    def _kNN(self):

        cell_len = self.atoms.cell.lengths()
        cutoff = self.cutoff_factor * np.min(cell_len)

        i_list, j_list, S_list = neighbor_list(
            "ijS", self.atoms, cutoff
        )

        N = self.N
        k = self.k

        candidates = [[] for _ in range(N)]

        for ii, jj, S in zip(i_list, j_list, S_list):
            # compute PBC vector using S
            cell = self.atoms.cell.array
            v = (S @ cell)
            d = np.linalg.norm(v)

            candidates[ii].append((d, jj, v))
            candidates[jj].append((d, ii, -v))

        nbr_idx = np.zeros((N, k), dtype=int)
        relpos  = np.zeros((N, k, 3), dtype=np.float32)
        dist    = np.zeros((N, k), dtype=np.float32)

        for i in range(N):

            cands = candidates[i]
            if len(cands) < k:
                need = k - len(cands)
                cands += [(1e9, i, np.zeros(3))] * need

            cands.sort(key=lambda x: x[0])
            topk = cands[:k]

            for t, (d, j, v) in enumerate(topk):
                nbr_idx[i, t] = j
                relpos[i, t] = v
                dist[i, t] = d

        return nbr_idx, relpos, dist


    ###################################################################
    # Bond detection (PBC-aware) — IMPORTANT
    ###################################################################
    def _detect_bonds(self):
        Z = self.atomic_numbers
        N = self.N

        bonds = []
        d0 = []

        for i in range(N):
            Zi = Z[i]
            for j in range(i+1, N):

                rc = covalent_radii[Zi] + covalent_radii[Z[j]] + 0.25

                v = self._pbc_vec(i, j)
                d = np.linalg.norm(v)

                if d <= rc:
                    bonds.append((i, j))
                    d0.append(d)

        return np.array(bonds), np.array(d0)
    ###################################################################
    # Coordination number — based on PBC-aware bond detection
    ###################################################################
    def _coordination_numbers(self):
        CN = np.zeros(self.N)
        for i, j in self.bond_pairs:
            CN[i] += 1
            CN[j] += 1
        return CN


    ###################################################################
    # Local planarity (using PBC-corrected vectors)
    ###################################################################
    def _local_planarity(self):
        pos = self.atoms.positions
        planar = np.zeros(self.N)

        for i in range(self.N):

            # neighbors connected by bonds
            idxs = np.where(self.bond_pairs[:, 0] == i)[0].tolist() \
                 + np.where(self.bond_pairs[:, 1] == i)[0].tolist()

            neigh = set()
            for idx in idxs:
                a, b = self.bond_pairs[idx]
                neigh.add(a)
                neigh.add(b)

            neigh.discard(i)
            neigh = list(neigh)

            if len(neigh) < 3:
                continue

            # take first 3 neighbors
            a, b, c = neigh[:3]

            # PBC vectors
            v1 = self._pbc_vec(i, a)
            v2 = self._pbc_vec(i, b)
            v3 = self._pbc_vec(i, c)

            n1 = np.cross(v1, v2)
            n2 = np.cross(v1, v3)

            if np.linalg.norm(n1) < 1e-9 or np.linalg.norm(n2) < 1e-9:
                continue

            cosang = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
            planar[i] = abs(cosang)

        return planar


    ###################################################################
    # Local graph radius (sum of PBC distances to bond neighbors)
    ###################################################################
    def _local_graph_radius(self):
        R = np.zeros(self.N)

        for i, j in self.bond_pairs:
            v = self._pbc_vec(i, j)
            d = np.linalg.norm(v)
            R[i] += d
            R[j] += d

        return R


    ###################################################################
    # Stiffness = ||F|| / ||disp_prev||  (disp_prev is already cartesian)
    ###################################################################
    def _local_stiffness(self):
        if self.disp_last is None:
            return np.zeros(self.N)
        disp_mag = np.linalg.norm(self.disp_last, axis=1) + 1e-12
        fmag = np.linalg.norm(self.forces, axis=1)
        return fmag / disp_mag


    ###################################################################
    # Torsion angle (PBC-based dihedral)
    ###################################################################
    def _torsion(self):
        tors = np.zeros(self.N)
        pos = self.atoms.positions

        for i in range(self.N):

            idxs = np.where(self.bond_pairs[:, 0] == i)[0].tolist() \
                 + np.where(self.bond_pairs[:, 1] == i)[0].tolist()

            neigh = set()
            for idx in idxs:
                a, b = self.bond_pairs[idx]
                neigh.add(a)
                neigh.add(b)

            neigh.discard(i)
            neigh = list(neigh)

            if len(neigh) < 3:
                continue

            a, b, c = neigh[:3]

            # positions with PBC vectors
            p0 = pos[a]
            p1 = pos[i]
            p2 = pos[b]
            p3 = pos[c]

            b0 = -self._pbc_vec_pos(p0, p1)
            b1 =  self._pbc_vec_pos(p1, p2)
            b2 =  self._pbc_vec_pos(p2, p3)

            n1 = np.cross(b0, b1)
            n2 = np.cross(b1, b2)

            if np.linalg.norm(n1) < 1e-9 or np.linalg.norm(n2) < 1e-9:
                continue

            x = np.dot(n1, n2)
            y = np.dot(np.cross(n1, n2), b1 / (np.linalg.norm(b1) + 1e-12))

            tors[i] = np.arctan2(y, x)

        return tors


    ###################################################################
    # Local stress = ||F|| * CN
    ###################################################################
    def _local_stress(self, CN):
        fmag = np.linalg.norm(self.forces, axis=1)
        return fmag * CN


    ###################################################################
    # SBU Tag (metal index labeling)
    ###################################################################
    def _sbu_id(self):
        Z = self.atomic_numbers
        metals = {20,22,23,24,25,26,27,28,29,40,42,44}
        arr = np.zeros(self.N)
        cid = 1
        for i in range(self.N):
            if Z[i] in metals:
                arr[i] = cid
                cid += 1
        return arr


    ###################################################################
    # Build full hybrid feature f_i = f_j
    ###################################################################
    def _build_f(self):

        F = self.forces
        f_norm = np.linalg.norm(F, axis=1) + 1e-12
        f_unit = F / f_norm[:, None]

        if self.F_prev is None:
            dF = np.zeros_like(F)
        else:
            dF = F - self.F_prev

        if self.disp_last is None:
            disp_prev = np.zeros_like(F)
        else:
            disp_prev = self.disp_last

        # All PBC-based features
        CN      = self._coordination_numbers()
        planar  = self._local_planarity()
        graphR  = self._local_graph_radius()
        stiff   = self._local_stiffness()
        tors    = self._torsion()
        stress  = self._local_stress(CN)
        sbu     = self._sbu_id()

        gF_mean = float(np.mean(f_norm))
        gF_std  = float(np.std(f_norm) + 1e-12)

        logF = np.log(f_norm)

        # final hybrid feature
        f = np.concatenate([
            f_unit,                     # 3
            logF[:,None],               # 1
            F,                          # 3
            dF,                         # 3
            disp_prev,                  # 3
            CN[:,None], planar[:,None], graphR[:,None],
            stiff[:,None], tors[:,None], stress[:,None], sbu[:,None],
            np.full((self.N,1), gF_mean),
            np.full((self.N,1), gF_std),
        ], axis=1).astype(np.float32)

        return f


    ###################################################################
    # Build observation (Hybrid-MACS)
    ###################################################################
    def _obs(self):

        f = self._build_f()

        nbr_idx, relpos, dist = self._kNN()

        N, k = self.N, self.k
        Fdim = f.shape[1]

        obs = np.zeros((N, Fdim + k*Fdim + k*3 + k), dtype=np.float32)

        for i in range(N):
            neigh_f = f[nbr_idx[i]].reshape(-1)

            obs[i] = np.concatenate([
                f[i],
                neigh_f,
                relpos[i].reshape(-1),
                dist[i].reshape(-1)
            ])

        return obs
    ###################################################################
    # Reset
    ###################################################################
    def reset(self):

        # Load structure fresh
        self.atoms = self.loader()
        self.N = len(self.atoms)
        self.atomic_numbers = np.array([a.number for a in self.atoms])

        # initial forces
        self.forces = self.atoms.get_forces().astype(np.float32)

        # initial bond list (PBC-aware)
        self.bond_pairs, self.bond_d0 = self._detect_bonds()

        # history
        self.F_prev = None
        self.disp_last = None

        # COM tracking
        self.com_prev = self.atoms.positions.mean(axis=0)

        # step counter
        self.step_count = 0

        return self._obs()


    ###################################################################
    # STEP (fully PBC-aware Hybrid-MACS)
    ###################################################################
    def step(self, action_scale):

        self.step_count += 1

        # -----------------------------------------------------------
        # Previous force
        # -----------------------------------------------------------
        F_old = self.forces
        fnorm = np.linalg.norm(F_old, axis=1, keepdims=True)

        # Adaptive displacement (MACS style)
        adaptive_scale = np.tanh(fnorm / 6.0)
        disp_scale = self.base_disp_scale * adaptive_scale

        # Clip action to [0,1]
        scale = np.clip(action_scale, 0.0, 1.0)

        # Displacement along negative force direction
        F_unit = F_old / (fnorm + 1e-12)
        disp = -disp_scale * scale * F_unit
        self.disp_last = disp.copy()

        self.atoms.positions += disp


        # -----------------------------------------------------------
        # New forces (PBC-aware via ASE)
        # -----------------------------------------------------------
        new_F = self.atoms.get_forces().astype(np.float32)
        new_norm = np.linalg.norm(new_F, axis=1)


        # -----------------------------------------------------------
        # Reward per atom (force-only, MACS style)
        # -----------------------------------------------------------
        old_norm = np.linalg.norm(F_old, axis=1)
        r = np.log(old_norm + 1e-12) - np.log(new_norm + 1e-12)
        reward_vec = np.clip(r, -5, 5).astype(np.float32)


        # -----------------------------------------------------------
        # COM drift penalty (PBC DOES NOT wrap COM — absolute drift)
        # -----------------------------------------------------------
        com_new = self.atoms.positions.mean(axis=0)
        dcom = np.linalg.norm(com_new - self.com_prev)

        reward_vec -= self.com_lambda * np.tanh(4.0 * dcom)
        self.com_prev = com_new.copy()

        if dcom > self.com_threshold:
            # fatal termination
            self.F_prev = new_F.copy()
            self.forces = new_F
            return self._obs(), reward_vec, True, "com", 0.0, float(np.max(new_norm))


        # -----------------------------------------------------------
        # Bond break termination (FULL PBC)
        # -----------------------------------------------------------
        for idx, (i, j) in enumerate(self.bond_pairs):

            v = self._pbc_vec(i, j)
            d = np.linalg.norm(v)

            ratio = d / self.bond_d0[idx]

            if ratio > self.bond_break_ratio:
                # penalty
                over = ratio - self.bond_break_ratio
                reward_vec += -self.bond_lambda * np.log1p(over)

                self.F_prev = new_F.copy()
                self.forces = new_F
                return self._obs(), reward_vec, True, "bond", 0.0, float(np.max(new_norm))


        # -----------------------------------------------------------
        # Fmax threshold convergence
        # -----------------------------------------------------------
        Fmax = float(np.max(new_norm))
        if Fmax < self.fmax_threshold:
            self.F_prev = new_F.copy()
            self.forces = new_F
            return self._obs(), reward_vec, True, "fmax", 0.0, Fmax


        # -----------------------------------------------------------
        # Max step termination
        # -----------------------------------------------------------
        if self.step_count >= self.max_steps:
            self.F_prev = new_F.copy()
            self.forces = new_F
            return self._obs(), reward_vec, True, "max_steps", 0.0, Fmax


        # -----------------------------------------------------------
        # Normal transition
        # -----------------------------------------------------------
        self.F_prev = new_F.copy()
        self.forces = new_F

        return self._obs(), reward_vec, False, None, 0.0, Fmax
