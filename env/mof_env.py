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
        k_bond=50.0,
        w_f=0.7,
        w_e=0.3,
        debug_bond=False
    ):
        self.atoms_loader = atoms_loader
        self.k = k_neighbors
        self.cmax = cmax
        self.max_steps = max_steps
        self.fmax_threshold = fmax_threshold

        # Soft bond penalty
        self.bond_break_ratio = bond_break_ratio
        self.k_bond = k_bond
        self.debug_bond = debug_bond

        # Reward weights
        self.w_f = w_f
        self.w_e = w_e

        self.reset()


    # ============================================================
    # TRUE BOND DETECTION (PBC-safe)
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
    # CYCLE DETECTION: Aromatic-like 6-ring detection
    # ============================================================
    def _detect_aromatic_nodes(self, adj):
        N = self.N
        aromatic = set()

        for start in range(N):
            queue = [(start, [start])]
            while queue:
                node, path = queue.pop()
                if len(path) > 6:
                    continue

                for nxt in adj[node]:
                    if nxt == start and len(path) == 6:
                        aromatic |= set(path)
                    elif nxt not in path:
                        queue.append((nxt, path + [nxt]))

        return aromatic


    # ============================================================
    def reset(self):

        self.atoms = self.atoms_loader()
        self.N = len(self.atoms)

        self.forces = self.atoms.get_forces()

        # History
        self.prev_forces = np.zeros_like(self.forces)
        self.prev_disp   = np.zeros_like(self.forces)

        # Covalent radii
        self.covalent_radii = np.array(
            [covalent_radii[a.number] for a in self.atoms]
        )

        # True bonds
        self.bond_pairs, self.bond_d0 = self._detect_true_bonds(self.atoms)
        print(f"[INIT] Detected true bonds = {len(self.bond_pairs)}")

        # ---------- GRAPH adjacency ----------
        self.adj = {i: [] for i in range(self.N)}
        for a, b in self.bond_pairs:
            self.adj[a].append(b)
            self.adj[b].append(a)

        # ---------- ATOMIC NUMBER ----------
        Z = np.array([a.number for a in self.atoms])

        # ============================================================
        # 1) Aromatic 6-cycle detection
        # ============================================================
        aromatic_nodes = self._detect_aromatic_nodes(self.adj)
        self.is_aromatic = np.zeros(self.N, dtype=np.float32)
        self.is_aromatic[list(aromatic_nodes)] = 1.0

        # ============================================================
        # 2) Atomic roles
        # ============================================================
        self.is_metal          = np.zeros(self.N, dtype=np.float32)
        self.is_carboxylate_O  = np.zeros(self.N, dtype=np.float32)
        self.is_mu2O           = np.zeros(self.N, dtype=np.float32)
        self.is_mu3O           = np.zeros(self.N, dtype=np.float32)
        self.is_linker         = np.zeros(self.N, dtype=np.float32)
        self.is_aromatic_C     = np.zeros(self.N, dtype=np.float32)

        # metal classification (simple rule)
        self.is_metal = (Z > 20).astype(np.float32)

        # carboxylate O (O–C–O)
        for i in range(self.N):
            if Z[i] == 8:
                for c in self.adj[i]:
                    if Z[c] == 6:
                        O2 = [x for x in self.adj[c] if Z[x] == 8 and x != i]
                        if len(O2) > 0:
                            self.is_carboxylate_O[i] = 1.0

        # bridging oxygens
        for i in range(self.N):
            if Z[i] == 8:
                metal_neighbors = sum(self.is_metal[j] for j in self.adj[i])

                if metal_neighbors == 2:
                    self.is_mu2O[i] = 1.0
                if metal_neighbors >= 3:
                    self.is_mu3O[i] = 1.0

        # aromatic carbon
        for i in range(self.N):
            if Z[i] == 6 and self.is_aromatic[i] == 1.0:
                self.is_aromatic_C[i] = 1.0

        # linker
        for i in range(self.N):
            if (not self.is_metal[i]) and (not self.is_carboxylate_O[i]) and (not self.is_aromatic_C[i]):
                if Z[i] in [6,7]:
                    self.is_linker[i] = 1.0

        # ============================================================
        # 3) Bond-class counts
        # ============================================================
        self.bond_types = np.zeros((self.N, 6), dtype=np.float32)

        for a, b in self.bond_pairs:

            # METAL–O
            if self.is_metal[a] and Z[b] == 8:
                self.bond_types[a][0] += 1
                self.bond_types[b][0] += 1

            # METAL–N
            if self.is_metal[a] and Z[b] == 7:
                self.bond_types[a][1] += 1
                self.bond_types[b][1] += 1

            # CARBOXYLATE
            if self.is_carboxylate_O[a] or self.is_carboxylate_O[b]:
                self.bond_types[a][2] += 1
                self.bond_types[b][2] += 1

            # AROMATIC C–C
            if (self.is_aromatic[a] and self.is_aromatic[b] 
                and Z[a] == 6 and Z[b] == 6):
                self.bond_types[a][3] += 1
                self.bond_types[b][3] += 1

            # MU2O
            if self.is_mu2O[a] or self.is_mu2O[b]:
                self.bond_types[a][4] += 1
                self.bond_types[b][4] += 1

            # MU3O
            if self.is_mu3O[a] or self.is_mu3O[b]:
                self.bond_types[a][5] += 1
                self.bond_types[b][5] += 1

        # Initial energy
        self.E_prev = self.atoms.get_potential_energy()
        self.step_count = 0

        return self._obs()


    # ============================================================
    # Compute k-nearest neighbors (MACS-style)
    # ============================================================
    def _compute_neighbors(self):

        i, j, offsets = neighbor_list("ijS", self.atoms, cutoff=6.0)
        cell = self.atoms.cell
        pos  = self.atoms.positions

        rel = pos[j] + offsets @ cell - pos[i]

        nd = {idx: [] for idx in range(self.N)}
        for a, b, r in zip(i, j, rel):
            nd[a].append((b, r))

        for idx in range(self.N):
            nd[idx] = sorted(nd[idx], key=lambda x: np.linalg.norm(x[1]))[: self.k]

        return nd


    # ============================================================
    # MAKE FEATURE (MACS + new chemistry features)
    # ============================================================
    def _make_feature(self, idx):

        ri    = self.covalent_radii[idx]
        gi    = self.forces[idx]
        gprev = self.prev_forces[idx]

        gnorm = max(np.linalg.norm(gi), 1e-12)

        # core MACS features
        core = np.concatenate([
            np.array([ri, min(gnorm, self.cmax), np.log(gnorm)]),
            gi,
            self.prev_disp[idx],
            gi - gprev,
        ])

        # atomic roles (6)
        roles = np.array([
            self.is_aromatic[idx],
            self.is_metal[idx],
            self.is_linker[idx],
            self.is_carboxylate_O[idx],
            self.is_mu2O[idx],
            self.is_mu3O[idx],
        ])

        # bond-class counts (6)
        btypes = self.bond_types[idx]

        return np.concatenate([core, roles, btypes])


    # ============================================================
    def _obs(self):

        neighbors = self._compute_neighbors()
        obs_list  = []

        for i in range(self.N):

            fi = self._make_feature(i)
            block = [fi]

            # neighbors
            for (j, _) in neighbors[i]:
                block.append(self._make_feature(j))

            # zero-padding
            for _ in range(self.k - len(neighbors[i])):
                block.append(np.zeros_like(fi))

            # distances
            dists = []
            vecs  = []

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
    # STEP
    # ============================================================
    def step(self, action):

        self.step_count += 1

        gnorm = np.linalg.norm(self.forces, axis=1)
        gnorm = np.where(gnorm > 1e-12, gnorm, 1e-12)

        c    = np.minimum(gnorm, self.cmax).reshape(-1, 1)
        disp = c * action
        self.atoms.positions += disp

        new_forces = self.atoms.get_forces()

        old_norm = np.maximum(np.linalg.norm(self.forces, axis=1), 1e-12)
        new_norm = np.maximum(np.linalg.norm(new_forces, axis=1), 1e-12)

        # force reward
        r_f = np.log(old_norm) - np.log(new_norm)

        # energy reward
        E_new = self.atoms.get_potential_energy()
        r_e   = (self.E_prev - E_new)

        reward = self.w_f * r_f + self.w_e * r_e
        self.E_prev = E_new

        # soft bond penalties
        cell = self.atoms.cell
        pbc  = self.atoms.pbc

        for idx, (a,b) in enumerate(self.bond_pairs):

            d = get_distances(
                self.atoms.positions[a][None],
                self.atoms.positions[b][None],
                cell=cell,
                pbc=pbc
            )[1][0][0]

            d0 = self.bond_d0[idx]
            ratio = d / d0

            stretch  = max(0.0, ratio - self.bond_break_ratio)
            compress = max(0.0, 0.6 - ratio)

            soft_penalty = self.k_bond * (stretch**2 + compress**2)
            reward -= soft_penalty

        # termination
        done = False

        if np.mean(new_norm) < self.fmax_threshold:
            done = True

        if self.step_count >= self.max_steps:
            done = True

        # update histories
        self.prev_disp   = disp.copy()
        self.prev_forces = self.forces.copy()
        self.forces      = new_forces.copy()

        return self._obs(), reward, done
