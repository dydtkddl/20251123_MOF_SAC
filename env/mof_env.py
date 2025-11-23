import numpy as np
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii
from ase.geometry import get_distances
from ase.geometry import find_mic


class MOFEnv:

    # ============================================================
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

        # Soft bond parameters
        self.bond_break_ratio = bond_break_ratio
        self.k_bond = k_bond
        self.debug_bond = debug_bond

        # Reward weights
        self.w_f = w_f
        self.w_e = w_e

        self.feature_dim = None

        self.reset()

    # ============================================================
    # TRUE BOND DETECTION
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
    # AROMATIC (6-cycle) DETECTION (IMPROVED)
    # ============================================================
    def _detect_aromatic_nodes(self, adj, Z):
        N = len(Z)
        aromatic = set()
        visited_cycles = set()

        def canonical_cycle(cycle):
            L = len(cycle)
            seqs = []
            for r in range(L):
                seqs.append(tuple(cycle[r:] + cycle[:r]))
            rev = list(reversed(cycle))
            for r in range(L):
                seqs.append(tuple(rev[r:] + rev[:r]))
            return min(seqs)

        def dfs(start, current, depth):
            if depth > 6:
                return
            last = current[-1]
            for nxt in adj[last]:
                if nxt == start and depth == 6:
                    cyc = canonical_cycle(current.copy())
                    if cyc not in visited_cycles:
                        # carbon-only, degree constraint
                        if all(Z[x] == 6 and len(adj[x]) <= 3 for x in cyc):
                            aromatic.update(cyc)
                        visited_cycles.add(cyc)
                elif nxt > start and nxt not in current:
                    dfs(start, current + [nxt], depth + 1)

        for s in range(N):
            if Z[s] == 6 and len(adj[s]) <= 3:
                dfs(s, [s], 1)

        return aromatic

    # ============================================================
    # METAL FLAG (IMPROVED)
    # ============================================================
    def _assign_metal_flags(self, Z):
        # Typical MOF metals
        MOF_METALS = {
            12, 13, 20,        # Mg, Al, Ca
            22,23,24,25,26,27,28,29,  # Ti → Cu
            30,                # Zn
            40,                # Zr
            72                 # Hf
        }
        return np.array([1.0 if z in MOF_METALS else 0.0 for z in Z], dtype=np.float32)

    # ============================================================
    # CARBOXYLATE O DETECTION (IMPROVED)
    # ============================================================
    def _detect_carboxylate_O(self, Z, adj, is_metal):
        N = len(Z)
        is_carboxylate_O = np.zeros(N, dtype=np.float32)

        for O in range(N):
            if Z[O] != 8:
                continue

            for C in adj[O]:
                if Z[C] != 6:
                    continue

                # C must have two O neighbors
                O_list = [x for x in adj[C] if Z[x] == 8]
                if len(O_list) != 2:
                    continue

                # C must bind at least one metal
                metal_neighbors = sum(is_metal[n] for n in adj[C])
                if metal_neighbors >= 1:
                    is_carboxylate_O[O] = 1.0
                    break

        return is_carboxylate_O

    # ============================================================
    # μ2-O / μ3-O DETECTION (IMPROVED)
    # ============================================================
    def _detect_mu_oxygens(self, Z, adj, is_metal):
        N = len(Z)
        is_mu2O = np.zeros(N, dtype=np.float32)
        is_mu3O = np.zeros(N, dtype=np.float32)

        for O in range(N):
            if Z[O] != 8:
                continue

            metal_count = sum(is_metal[n] for n in adj[O])

            if metal_count == 2:
                is_mu2O[O] = 1.0
            elif metal_count >= 3:
                is_mu3O[O] = 1.0

        return is_mu2O, is_mu3O

    # ============================================================
    # RESET
    # ============================================================
    def reset(self):

        self.atoms = self.atoms_loader()
        self.N = len(self.atoms)

        # Forces
        self.forces = self.atoms.get_forces()
        self.prev_forces = np.zeros_like(self.forces)
        self.prev_disp = np.zeros_like(self.forces)

        # Radius
        Z = np.array([a.number for a in self.atoms])
        self.covalent_radii = np.array([covalent_radii[z] for z in Z])

        # True bonds
        self.bond_pairs, self.bond_d0 = self._detect_true_bonds(self.atoms)
        print(f"[INIT] Detected true bonds = {len(self.bond_pairs)}")

        # Build adjacency
        self.adj = {i: [] for i in range(self.N)}
        for a, b in self.bond_pairs:
            self.adj[a].append(b)
            self.adj[b].append(a)

        # Aromatic detection
        aromatic_nodes = self._detect_aromatic_nodes(self.adj, Z)
        self.is_aromatic = np.zeros(self.N, dtype=np.float32)
        self.is_aromatic[list(aromatic_nodes)] = 1.0

        # Role flags
        self.is_metal = self._assign_metal_flags(Z)
        self.is_carboxylate_O = self._detect_carboxylate_O(Z, self.adj, self.is_metal)
        self.is_mu2O, self.is_mu3O = self._detect_mu_oxygens(Z, self.adj, self.is_metal)

        # Aromatic carbon
        self.is_aromatic_C = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            if Z[i] == 6 and self.is_aromatic[i] == 1.0:
                self.is_aromatic_C[i] = 1.0

        # Linker atoms
        self.is_linker = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            if (
                not self.is_metal[i]
                and not self.is_carboxylate_O[i]
                and not self.is_aromatic_C[i]
            ):
                if Z[i] in [6, 7]:
                    self.is_linker[i] = 1.0

        # ============================================================
        # BOND-TYPE COUNTS (IMPROVED)
        # ============================================================
        self.bond_types = np.zeros((self.N, 6), dtype=np.float32)

        for a, b in self.bond_pairs:

            # Metal–O
            if self.is_metal[a] and Z[b] == 8:
                self.bond_types[a][0] += 1
            if self.is_metal[b] and Z[a] == 8:
                self.bond_types[b][0] += 1

            # Metal–N
            if self.is_metal[a] and Z[b] == 7:
                self.bond_types[a][1] += 1
            if self.is_metal[b] and Z[a] == 7:
                self.bond_types[b][1] += 1

            # Carboxylate O
            if self.is_carboxylate_O[a]:
                self.bond_types[b][2] += 1
            if self.is_carboxylate_O[b]:
                self.bond_types[a][2] += 1

            # Aromatic C–C
            if self.is_aromatic_C[a] and self.is_aromatic_C[b]:
                self.bond_types[a][3] += 1
                self.bond_types[b][3] += 1

            # μ2-O
            if self.is_mu2O[a]:
                self.bond_types[b][4] += 1
            if self.is_mu2O[b]:
                self.bond_types[a][4] += 1

            # μ3-O
            if self.is_mu3O[a]:
                self.bond_types[b][5] += 1
            if self.is_mu3O[b]:
                self.bond_types[a][5] += 1

        # Feature dimension
        self.feature_dim = len(self._make_feature(0))

        self.E_prev = self.atoms.get_potential_energy()
        self.step_count = 0

        return self._obs()

    # ============================================================
    # Relative vector (PBC-aware MIC)
    # ============================================================
    def _rel_vec(self, i, j):
        disp = self.atoms.positions[j] - self.atoms.positions[i]
        cell = self.atoms.cell.array
        frac = np.linalg.solve(cell.T, disp)
        frac -= np.round(frac)
        return frac @ cell

    # ============================================================
    # Hop sets
    # ============================================================
    def _get_hop_sets(self, idx, max_hop=3):
        visited = set([idx])
        frontier = [idx]

        hop_map = {1: [], 2: [], 3: []}

        for hop in range(1, max_hop + 1):
            next_frontier = []
            for node in frontier:
                for nxt in self.adj[node]:
                    if nxt not in visited:
                        visited.add(nxt)
                        next_frontier.append(nxt)
                        hop_map[hop].append(nxt)

            frontier = next_frontier

        return hop_map

    # ============================================================
    # MAKE FEATURE
    # ============================================================
    def _make_feature(self, idx):

        ri = self.covalent_radii[idx]
        gi = self.forces[idx]
        gprev = self.prev_forces[idx]

        gnorm = max(np.linalg.norm(gi), 1e-12)

        core = np.concatenate([
            np.array([ri, min(gnorm, self.cmax), np.log(gnorm)]),
            gi,
            self.prev_disp[idx],
            gi - gprev,
        ])

        roles = np.array([
            self.is_aromatic[idx],
            self.is_metal[idx],
            self.is_linker[idx],
            self.is_carboxylate_O[idx],
            self.is_mu2O[idx],
            self.is_mu3O[idx],
        ])

        return np.concatenate([core, roles, self.bond_types[idx]])

    # ============================================================
    # OBSERVATION
    # ============================================================
    def _obs(self):

        obs_list = []

        for i in range(self.N):

            fi = self._make_feature(i)
            hop_sets = self._get_hop_sets(i, max_hop=3)

            selected = []

            # 1-hop
            for j in hop_sets[1]:
                if len(selected) < self.k:
                    selected.append(j)

            # 2-hop
            for j in hop_sets[2]:
                if len(selected) < self.k:
                    selected.append(j)

            # 3-hop
            remain = self.k - len(selected)
            if remain > 0 and len(hop_sets[3]) > 0:
                candidates = hop_sets[3]
                if len(candidates) <= remain:
                    selected += candidates
                else:
                    selected += list(np.random.choice(candidates, remain, replace=False))

            # Padding
            while len(selected) < self.k:
                selected.append(None)

            nbr_feats = []
            dists = []
            vecs = []

            for j in selected:
                if j is None:
                    nbr_feats.append(np.zeros_like(fi))
                    dists.append(0.0)
                    vecs.append(np.zeros(3))
                else:
                    fj = self._make_feature(j)
                    rel = self._rel_vec(i, j)

                    nbr_feats.append(fj)
                    dists.append(np.linalg.norm(rel))
                    vecs.append(rel)

            block = [fi] + nbr_feats
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

        c = np.minimum(gnorm, self.cmax).reshape(-1, 1)
        disp = c * action
        self.atoms.positions += disp

        new_forces = self.atoms.get_forces()

        old_norm = np.maximum(np.linalg.norm(self.forces, axis=1), 1e-12)
        new_norm = np.maximum(np.linalg.norm(new_forces, axis=1), 1e-12)

        # Rewards
        r_f = np.log(old_norm) - np.log(new_norm)
        E_new = self.atoms.get_potential_energy()
        r_e = (self.E_prev - E_new)
        reward = self.w_f * r_f + self.w_e * r_e
        self.E_prev = E_new

        # Soft bond penalties
        cell = self.atoms.cell
        pbc = self.atoms.pbc

        for idx, (a, b) in enumerate(self.bond_pairs):

            d = get_distances(
                self.atoms.positions[a][None],
                self.atoms.positions[b][None],
                cell=cell, pbc=pbc
            )[1][0][0]

            d0 = self.bond_d0[idx]
            ratio = d / d0

            stretch = max(0.0, ratio - self.bond_break_ratio)
            compress = max(0.0, 0.6 - ratio)

            soft_penalty = self.k_bond * (stretch**2 + compress**2)
            reward -= soft_penalty

        done = False
        if np.mean(new_norm) < self.fmax_threshold:
            done = True
        if self.step_count >= self.max_steps:
            done = True

        # Update history
        self.prev_disp = disp.copy()
        self.prev_forces = self.forces.copy()
        self.forces = new_forces.copy()

        return self._obs(), reward, done
