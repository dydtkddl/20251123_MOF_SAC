import numpy as np
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii


class MOFEnv:

    # ============================================================
    def __init__(
        self,
        atoms_loader,
        k_neighbors=12,
        cmax=0.4,
        max_steps=300,
        fmax_threshold=0.05,
        bond_break_ratio=2.4,
        k_bond=3.0,
        max_penalty=10.0,
        debug_bond=False,
    ):
        self.atoms_loader = atoms_loader

        self.k = k_neighbors
        self.cmax = cmax
        self.max_steps = max_steps
        self.fmax_threshold = fmax_threshold

        # Bond-related
        self.bond_break_ratio = bond_break_ratio
        self.k_bond = k_bond
        self.max_penalty = max_penalty
        self.debug_bond = debug_bond

        # COM control
        self.com_threshold = 0.30
        self.com_lambda = 20.0

        self.feature_dim = None
        self.reset()

    # ============================================================
    # True Bonds
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

        return np.array(bond_pairs, int), np.array(bond_d0, float)

    # ============================================================
    # Aromatic detection
    # ============================================================
    def _detect_aromatic_nodes(self, adj, Z):
        N = len(Z)
        aromatic = set()
        visited = set()

        def canonical(cycle):
            L = len(cycle)
            seqs = []
            for r in range(L):
                seqs.append(tuple(cycle[r:] + cycle[:r]))
            rev = list(reversed(cycle))
            for r in range(L):
                seqs.append(tuple(rev[r:] + rev[:r]))
            return min(seqs)

        def dfs(s, path, depth):
            if depth > 6:
                return
            last = path[-1]

            for nxt in adj[last]:
                if nxt == s and depth == 6:
                    cyc = canonical(path.copy())
                    if cyc not in visited:
                        if all(Z[x] == 6 and len(adj[x]) <= 3 for x in cyc):
                            aromatic.update(cyc)
                        visited.add(cyc)
                elif nxt > s and nxt not in path:
                    dfs(s, path + [nxt], depth + 1)

        for s in range(N):
            if Z[s] == 6 and len(adj[s]) <= 3:
                dfs(s, [s], 1)

        return aromatic

    # ============================================================
    def _assign_metal_flags(self, Z):
        MOF_METALS = {12,13,20,22,23,24,25,26,27,28,29,30,40,72}
        return np.array([1.0 if z in MOF_METALS else 0.0 for z in Z], float)

    def _detect_carboxylate_O(self, Z, adj, is_metal):
        N = len(Z)
        out = np.zeros(N, float)

        for O in range(N):
            if Z[O] != 8:
                continue
            for C in adj[O]:
                if Z[C] != 6:
                    continue

                O_list = [x for x in adj[C] if Z[x] == 8]
                if len(O_list) != 2:
                    continue

                if sum(is_metal[n] for n in adj[C]) >= 1:
                    out[O] = 1.0
                    break

        return out

    def _detect_mu_oxygens(self, Z, adj, is_metal):
        N = len(Z)
        mu2 = np.zeros(N, float)
        mu3 = np.zeros(N, float)

        for O in range(N):
            if Z[O] != 8:
                continue
            m = sum(is_metal[n] for n in adj[O])
            if m == 2:
                mu2[O] = 1.0
            elif m >= 3:
                mu3[O] = 1.0
        return mu2, mu3

    # ============================================================
    # RESET
    # ============================================================
    def reset(self):
        self.atoms = self.atoms_loader()
        self.N = len(self.atoms)

        self.forces = self.atoms.get_forces().astype(np.float32)
        self.prev_forces = np.zeros_like(self.forces)
        self.prev_disp = np.zeros_like(self.forces)

        Z = np.array([a.number for a in self.atoms])
        self.covalent_radii = np.array([covalent_radii[z] for z in Z]).astype(np.float32)

        self.bond_pairs, self.bond_d0 = self._detect_true_bonds(self.atoms)
        print(f"[INIT] Detected true bonds = {len(self.bond_pairs)}")

        # adjacency
        self.adj = {i:[] for i in range(self.N)}
        for a, b in self.bond_pairs:
            self.adj[a].append(b)
            self.adj[b].append(a)

        aromatic_nodes = self._detect_aromatic_nodes(self.adj, Z)
        self.is_aromatic = np.zeros(self.N, float)
        self.is_aromatic[list(aromatic_nodes)] = 1.0

        self.is_metal = self._assign_metal_flags(Z)
        self.is_carboxylate_O = self._detect_carboxylate_O(Z, self.adj, self.is_metal)
        self.is_mu2O, self.is_mu3O = self._detect_mu_oxygens(Z, self.adj, self.is_metal)

        self.is_aromatic_C = np.zeros(self.N, float)
        for i in range(self.N):
            if Z[i] == 6 and self.is_aromatic[i] == 1.0:
                self.is_aromatic_C[i] = 1.0

        self.is_linker = np.zeros(self.N, float)
        for i in range(self.N):
            if (
                (not self.is_metal[i])
                and (not self.is_carboxylate_O[i])
                and (not self.is_aromatic_C[i])
                and Z[i] in [6,7]
            ):
                self.is_linker[i] = 1.0

        self.bond_types = np.zeros((self.N,6), float)

        for a, b in self.bond_pairs:
            Za, Zb = Z[a], Z[b]

            if self.is_metal[a] and Zb == 8: self.bond_types[a][0] += 1
            if self.is_metal[b] and Za == 8: self.bond_types[b][0] += 1

            if self.is_metal[a] and Zb == 7: self.bond_types[a][1] += 1
            if self.is_metal[b] and Za == 7: self.bond_types[b][1] += 1

            if self.is_carboxylate_O[a]: self.bond_types[b][2] += 1
            if self.is_carboxylate_O[b]: self.bond_types[a][2] += 1

            if self.is_aromatic_C[a] and self.is_aromatic_C[b]:
                self.bond_types[a][3] += 1
                self.bond_types[b][3] += 1

            if self.is_mu2O[a]: self.bond_types[b][4] += 1
            if self.is_mu2O[b]: self.bond_types[a][4] += 1

            if self.is_mu3O[a]: self.bond_types[b][5] += 1
            if self.is_mu3O[b]: self.bond_types[a][5] += 1

        self.feature_dim = len(self._make_feature(0))
        self.step_count = 0

        self.COM_prev = self.atoms.positions.mean(axis=0).astype(np.float32)

        return self._obs()

    # ============================================================
    def _rel_vec(self, i, j):
        disp = self.atoms.positions[j] - self.atoms.positions[i]
        cell = self.atoms.cell.array
        frac = np.linalg.solve(cell.T, disp)
        frac -= np.round(frac)
        return frac @ cell

    # ============================================================
    # Hops 1–3
    # ============================================================
    def _get_hop_sets(self, idx, max_hop=3):
        visited = set([idx])
        frontier = [idx]
        hop_map = {1:[], 2:[], 3:[]}

        for hop in range(1, max_hop+1):
            nxt_frontier=[]
            for node in frontier:
                for nxt in self.adj[node]:
                    if nxt not in visited:
                        visited.add(nxt)
                        nxt_frontier.append(nxt)
                        hop_map[hop].append(nxt)
            frontier = nxt_frontier

        return hop_map

    # ============================================================
    # Feature
    # ============================================================
    def _make_feature(self, idx):

        ri = self.covalent_radii[idx]
        gi = self.forces[idx]
        gprev = self.prev_forces[idx]

        gnorm = max(np.linalg.norm(gi), 1e-12)

        core = np.concatenate([
            np.array([ri, min(gnorm, self.cmax), np.log(gnorm+1e-6)]),
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
    # Observation
    # ============================================================
    def _obs(self):

        obs_list = []

        for i in range(self.N):

            fi = self._make_feature(i)
            hop_sets = self._get_hop_sets(i)

            selected = []

            for j in hop_sets[1]:
                if len(selected) < self.k:
                    selected.append(j)

            for j in hop_sets[2]:
                if len(selected) < self.k:
                    selected.append(j)

            remain = self.k - len(selected)
            if remain > 0 and len(hop_sets[3]) > 0:
                cand = hop_sets[3]
                if len(cand) <= remain:
                    selected += cand
                else:
                    selected += list(np.random.choice(cand, remain, False))

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

        return np.array(obs_list, float)

    # ============================================================
    # STEP (Stable RL version + done_reason added)
    # ============================================================
    def step(self, action):

        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # --------------------------------------
        # 1) Force-adaptive displacement
        # --------------------------------------
        gnorm = np.linalg.norm(self.forces, axis=1)
        scale = np.minimum(gnorm, self.cmax).reshape(-1, 1)

        disp = 0.003 * action * (scale / self.cmax)
        self.atoms.positions += disp

        new_forces = self.atoms.get_forces().astype(np.float32)
        old_norm = np.maximum(np.linalg.norm(self.forces, axis=1), 1e-12)
        new_norm = np.maximum(np.linalg.norm(new_forces, axis=1), 1e-12)

        # --------------------------------------
        # 2) Force reward  (×10 scaling 적용됨)
        # --------------------------------------
        r_f = 10.0 * (np.log(old_norm+1e-6) - np.log(new_norm+1e-6))
        reward = r_f.copy()

        # --------------------------------------
        # 3) COM penalty
        # --------------------------------------
        COM_new = self.atoms.positions.mean(axis=0)
        delta_COM = np.linalg.norm(COM_new - self.COM_prev)

        reward -= self.com_lambda * delta_COM
        self.COM_prev = COM_new.copy()

        if delta_COM > self.com_threshold:
            return self._obs(), reward, True, "com"

        # --------------------------------------
        # 4) Bond penalty
        # --------------------------------------
        for idx, (a, b) in enumerate(self.bond_pairs):

            rel = self._rel_vec(a, b)
            d  = np.linalg.norm(rel)
            d0 = self.bond_d0[idx]
            ratio = d / d0

            stretch = max(0.0, ratio - self.bond_break_ratio)
            compress = max(0.0, 0.6 - ratio)

            penalty = self.k_bond * np.sqrt(stretch**2 + compress**2)
            penalty = min(penalty, self.max_penalty)

            reward -= penalty

            if ratio > 6.0 or ratio < 0.25:
                return self._obs(), reward, True, "bond"

        # --------------------------------------
        # 5) Termination conditions
        # --------------------------------------
        if np.mean(new_norm) < self.fmax_threshold:
            self._update_memory(disp, new_forces)
            return self._obs(), reward, True, "fmax"

        if self.step_count >= self.max_steps:
            self._update_memory(disp, new_forces)
            return self._obs(), reward, True, "max_steps"

        # --------------------------------------
        # Normal update
        # --------------------------------------
        self._update_memory(disp, new_forces)
        return self._obs(), reward, False, None

    # ============================================================
    def _update_memory(self, disp, new_forces):
        self.prev_disp = disp.copy()
        self.prev_forces = self.forces.copy()
        self.forces = new_forces.copy()
