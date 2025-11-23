# import numpy as np
# from ase.neighborlist import neighbor_list
# from ase.data import covalent_radii


# class MOFEnv:

#     def __init__(
#         self,
#         atoms_loader,
#         k_neighbors=12,
#         cmax=0.4,
#         max_steps=300,
#         fmax_threshold=0.05,
#         bond_break_ratio=1.8,
#         bond_penalty=100.0,
#     ):
#         self.atoms_loader = atoms_loader
#         self.k = k_neighbors
#         self.cmax = cmax
#         self.max_steps = max_steps
#         self.fmax_threshold = fmax_threshold

#         self.bond_break_ratio = bond_break_ratio
#         self.bond_penalty = bond_penalty

#         self.reset()


#     ############################################################
#     def _detect_true_bonds(self, atoms):
#         """
#         진짜 화학 결합만 정확히 탐지하는 알고리즘:
#         covalent radius sum + tolerance 방식
#         """

#         pos = atoms.positions
#         cell = atoms.cell

#         # PBC 포함 neighbor list 생성
#         i, j, offsets = neighbor_list("ijS", atoms, cutoff=4.0)

#         bond_pairs = []
#         bond_d0 = []

#         for a, b, off in zip(i, j, offsets):

#             # 거리 계산 (PBC 적용)
#             r_ab = pos[b] + off @ cell - pos[a]
#             d = np.linalg.norm(r_ab)

#             # 두 원자의 covalent radius 합
#             r_cov = covalent_radii[atoms[a].number] + covalent_radii[atoms[b].number]

#             # 허용 오차 0.4 Å
#             if d <= r_cov + 0.4:
#                 bond_pairs.append((a, b))
#                 bond_d0.append(d)

#         # numpy array 로 변환하며 순서 보존
#         return np.array(bond_pairs, dtype=int), np.array(bond_d0, dtype=float)


#     ############################################################
#     def reset(self):

#         self.atoms = self.atoms_loader()
#         self.N = len(self.atoms)

#         # Initial forces
#         self.forces = self.atoms.get_forces()

#         # History
#         self.prev_forces = np.zeros_like(self.forces)
#         self.prev_disp = np.zeros_like(self.forces)

#         # Covalent radii
#         self.covalent_radii = np.array(
#             [covalent_radii[a.number] for a in self.atoms]
#         )

#         # =======================================
#         # 🔥 진짜 bond detection 수행
#         # =======================================
#         self.bond_pairs, self.bond_d0 = self._detect_true_bonds(self.atoms)

#         print(f"Detected {len(self.bond_pairs)} true bonds")

#         self.step_count = 0

#         return self._obs()


#     ############################################################
#     def _compute_neighbors(self):
#         i, j, offsets = neighbor_list("ijS", self.atoms, cutoff=6.0)
#         rel = self.atoms.positions[j] + offsets @ self.atoms.cell - self.atoms.positions[i]

#         nd = {idx: [] for idx in range(self.N)}
#         for a, b, r in zip(i, j, rel):
#             nd[a].append((b, r))

#         for idx in range(self.N):
#             nd[idx] = sorted(nd[idx], key=lambda x: np.linalg.norm(x[1]))[: self.k]

#         return nd


#     ############################################################
#     def _make_feature(self, idx):
#         ri = self.covalent_radii[idx]

#         gi = self.forces[idx]
#         gprev = self.prev_forces[idx]

#         gnorm = np.linalg.norm(gi)
#         gnorm = max(gnorm, 1e-12)

#         loggn = np.log(gnorm)
#         cti = min(gnorm, self.cmax)

#         di = self.prev_disp[idx]
#         dgi = gi - gprev

#         return np.concatenate([
#             np.array([ri, cti, loggn]),
#             gi,
#             di,
#             dgi,
#         ])


#     ############################################################
#     def _obs(self):
#         neighbors = self._compute_neighbors()
#         obs_list = []

#         for i in range(self.N):

#             fi = self._make_feature(i)
#             block = [fi]

#             # neighbors
#             for (j, rel) in neighbors[i]:
#                 block.append(self._make_feature(j))

#             # padding
#             for _ in range(self.k - len(neighbors[i])):
#                 block.append(np.zeros_like(fi))

#             # distance + vector
#             dists = []
#             vecs = []

#             for (_, rel) in neighbors[i]:
#                 dists.append(np.linalg.norm(rel))
#                 vecs.append(rel)

#             for _ in range(self.k - len(neighbors[i])):
#                 dists.append(0.0)
#                 vecs.append(np.zeros(3))

#             block.append(np.array(dists))
#             block.append(np.array(vecs).reshape(-1))

#             obs_list.append(np.concatenate(block))

#         return np.array(obs_list, dtype=np.float32)
#     ############################################################
#     def step(self, action):

#         self.step_count += 1

#         # displacement scale
#         gnorm = np.linalg.norm(self.forces, axis=1)
#         gnorm = np.where(gnorm > 1e-12, gnorm, 1e-12)
#         c = np.minimum(gnorm, self.cmax).reshape(-1, 1)

#         # displacement applied this step
#         disp = c * action
#         self.atoms.positions += disp

#         # compute new forces
#         new_forces = self.atoms.get_forces()

#         old_norm = np.linalg.norm(self.forces, axis=1)
#         new_norm = np.linalg.norm(new_forces, axis=1)

#         old_norm = np.where(old_norm > 1e-12, old_norm, 1e-12)
#         new_norm = np.where(new_norm > 1e-12, new_norm, 1e-12)

#         reward = np.log(old_norm) - np.log(new_norm)

#         done = False

#         # =====================================
#         # 🔥 EXTREME DEBUG: full per-bond info
#         # =====================================
#         pos = self.atoms.positions
#         broken = False

#         for idx, (a, b) in enumerate(self.bond_pairs):

#             d0 = self.bond_d0[idx]
#             d = np.linalg.norm(pos[a] - pos[b])
#             ratio = d / d0 if d0 > 1e-12 else 999

#             # NEW: force magnitude per-atom
#             Fa = np.linalg.norm(self.forces[a])
#             Fb = np.linalg.norm(self.forces[b])

#             # NEW: displacement magnitude per-atom
#             da = np.linalg.norm(disp[a])
#             db = np.linalg.norm(disp[b])

#             # NEW: new force magnitude after step
#             Fna = np.linalg.norm(new_forces[a])
#             Fnb = np.linalg.norm(new_forces[b])

#             print("\n" + "="*80)
#             print(f"[BOND CHECK #{idx}] atoms ({a},{b}) [{self.atoms[a].symbol}, {self.atoms[b].symbol}]")
#             print("-"*80)

#             print(f"Initial bond length d0:        {d0:.6f} Å")
#             print(f"Current bond length d:         {d:.6f} Å")
#             print(f"Bond ratio (d/d0):             {ratio:.6f}")
#             print(f"Stretch threshold:             {self.bond_break_ratio:.2f} × d0 = {self.bond_break_ratio*d0:.6f}")
#             print(f"Compress threshold:            0.60 × d0 = {0.6*d0:.6f}")
#             print("")

#             print(f"Force  |F[a]| pre-step:        {Fa:.6f}   (atom {a})")
#             print(f"Force  |F[b]| pre-step:        {Fb:.6f}   (atom {b})")
#             print(f"Force  |F[a]| post-step:       {Fna:.6f}  (new)")
#             print(f"Force  |F[b]| post-step:       {Fnb:.6f}  (new)")
#             print("")

#             print(f"Disp   |disp[a]| this step:    {da:.6f} Å")
#             print(f"Disp   |disp[b]| this step:    {db:.6f} Å")
#             print("")

#             # -------------------------------
#             # break/stretch/compress checks
#             # -------------------------------
#             if d > self.bond_break_ratio * d0:
#                 print(">>> BOND STRETCH BREAK DETECTED! (penalty applied)")
#                 reward -= self.bond_penalty
#                 broken = True

#             elif d < 0.6 * d0:
#                 print(">>> BOND COMPRESS BREAK DETECTED! (penalty applied)")
#                 reward -= self.bond_penalty
#                 broken = True
#             else:
#                 print("Bond OK")

#             print("="*80 + "\n")

#         if broken:
#             done = True

#         # termination by force threshold
#         if np.mean(new_norm) < self.fmax_threshold:
#             done = True

#         if self.step_count >= self.max_steps:
#             done = True

#         # update histories
#         self.prev_disp = disp.copy()
#         self.prev_forces = self.forces.copy()
#         self.forces = new_forces.copy()

#         return self._obs(), reward, done
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
        bond_penalty=100.0,
        debug_bond=False
    ):
        self.atoms_loader = atoms_loader
        self.k = k_neighbors
        self.cmax = cmax
        self.max_steps = max_steps
        self.fmax_threshold = fmax_threshold

        # bond parameters
        self.bond_break_ratio = bond_break_ratio
        self.bond_penalty = bond_penalty
        self.debug_bond = debug_bond

        self.reset()


    # ============================================================
    # TRUE BOND DETECTION (PBC-SAFE)
    # ============================================================
    def _detect_true_bonds(self, atoms):

        # generate near neighbors (PBC)
        i, j, offsets = neighbor_list("ijS", atoms, cutoff=4.0)

        pos = atoms.positions
        cell = atoms.cell

        bond_pairs = []
        bond_d0 = []

        for a, b, off in zip(i, j, offsets):

            # PBC-safe relative vector
            rel = pos[b] + off @ cell - pos[a]
            d = np.linalg.norm(rel)

            # covalent radius criterion
            rc = covalent_radii[atoms[a].number] + covalent_radii[atoms[b].number]

            if d <= rc + 0.4:
                bond_pairs.append((a, b))
                bond_d0.append(d)

        return np.array(bond_pairs, dtype=int), np.array(bond_d0, dtype=float)


    # ============================================================
    def reset(self):

        self.atoms = self.atoms_loader()
        self.N = len(self.atoms)

        # forces
        self.forces = self.atoms.get_forces()

        # histories
        self.prev_forces = np.zeros_like(self.forces)
        self.prev_disp = np.zeros_like(self.forces)

        # covalent radii
        self.covalent_radii = np.array(
            [covalent_radii[a.number] for a in self.atoms]
        )

        # ---- true chemical bonds ----
        self.bond_pairs, self.bond_d0 = self._detect_true_bonds(self.atoms)
        print(f"[INIT] Detected true bonds = {len(self.bond_pairs)}")

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
            nd[idx] = sorted(nd[idx], key=lambda x: np.linalg.norm(x[1]))[: self.k]

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

            for _ in range(self.k - len(neighbors[i])):
                block.append(np.zeros_like(fi))

            # distance & vector
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
    #                    🔥   STEP (PBC-SAFE)   🔥
    # ============================================================
    def step(self, action):

        self.step_count += 1

        # --- displacement scale ---
        gnorm = np.linalg.norm(self.forces, axis=1)
        gnorm = np.where(gnorm > 1e-12, gnorm, 1e-12)

        c = np.minimum(gnorm, self.cmax).reshape(-1, 1)
        disp = c * action

        self.atoms.positions += disp

        # --- new forces ---
        new_forces = self.atoms.get_forces()

        # reward
        old_norm = np.maximum(np.linalg.norm(self.forces, axis=1), 1e-12)
        new_norm = np.maximum(np.linalg.norm(new_forces, axis=1), 1e-12)

        reward = np.log(old_norm) - np.log(new_norm)

        done = False

        # ================================================
        #         🔥 PBC-SAFE BOND CHECKING
        # ================================================
        cell = self.atoms.cell
        pbc = self.atoms.pbc
        broken = False

        for idx, (a, b) in enumerate(self.bond_pairs):

            # PBC minimum image distance
            d = get_distances(
                self.atoms.positions[a][None],
                self.atoms.positions[b][None],
                cell=cell,
                pbc=pbc,
            )[1][0][0]

            d0 = self.bond_d0[idx]

            if self.debug_bond:
                print("\n" + "="*80)
                print(f"[BOND CHECK #{idx}] atoms ({a},{b}) [{self.atoms[a].symbol}, {self.atoms[b].symbol}]")
                print("-"*80)
                print(f"d0 = {d0:.4f} Å,   d = {d:.4f} Å   ratio={d/d0:.3f}")

            # stretch
            if d > self.bond_break_ratio * d0:
                reward -= self.bond_penalty
                broken = True
                if self.debug_bond:
                    print(">>> STRETCH BREAK DETECTED")
                continue

            # compress
            if d < 0.6 * d0:
                reward -= self.bond_penalty
                broken = True
                if self.debug_bond:
                    print(">>> COMPRESS BREAK DETECTED")
                continue

            if self.debug_bond:
                print("Bond OK")

        if broken:
            done = True

        # termination conditions
        if np.mean(new_norm) < self.fmax_threshold:
            done = True

        if self.step_count >= self.max_steps:
            done = True

        # update histories
        self.prev_disp = disp.copy()
        self.prev_forces = self.forces.copy()
        self.forces = new_forces.copy()

        return self._obs(), reward, done
