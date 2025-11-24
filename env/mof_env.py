# env/mof_env.py
###############################################################
# MOFEnv — Multi-Agent RL Environment for MOF Structure Relaxation
# -----------------------------------------------------------------
# - 각 원자 = 에이전트 (per-atom observation, per-atom action)
# - _build_f():
#     * force / Δforce / disp history
#     * CN, planarity, graph radius, torsion, stress, SBU id
#     * Global force statistics (mean/std of |F|)
#     * MOF topology-aware features:
#         - _atom_roles(): metal / linker / μ2-O / μ3-O / carboxylate-O /
#                         aromatic C / terminal
#         - _bond_motif_feats(): Metal–O, Metal–N, carboxylate O–C–O,
#                               μ2/μ3-O–Metal, aromatic C–C motif
#         - _pore_side_feats(): fractional coords, center distance,
#                               boundary proximity, pore-lining flag
# - _obs(): per-atom obs_i + k-NN neighbor features → obs_atom (N, FEAT)
# - reset() → (obs_atom, obs_global_flat)
# - step(actions) → (next_obs_atom, next_obs_global_flat, reward_scalar, done,
#                    reason, Etot_stub, Fmax)
###############################################################

import logging
from typing import Tuple, Optional

import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list

logger = logging.getLogger("env.mof_env")


class MOFEnv:
    """
    Multi-Agent MOF Environment (per-atom observation & action).

    - atoms_loader: 함수를 받아서, 매 reset()마다 fresh ASE Atoms를 생성
    - 각 step에서:
        actions: shape (N, 3) in [-1, 1]
        내부적으로 MACS-style scaling으로 displacement 결정 후 구조 업데이트
        reward: log|F| 감소 + COM drift / bond stretch / fmax / max_steps 조건 반영
    """

    def __init__(
        self,
        atoms_loader,
        k_neighbors: int = 12,
        cmax: float = 0.40,
        max_steps: int = 300,
        fmax_threshold: float = 0.05,
        bond_break_ratio: float = 2.4,
        k_bond: float = 3.0,
        max_penalty: float = 10.0,
        com_threshold: float = 0.25,
        com_lambda: float = 4.0,
        w_force: float = 1.0,
        cutoff_factor: float = 0.8,
        debug_bond: bool = False,
        disp_scale: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        atoms_loader : callable
            호출 시 ASE Atoms를 반환하는 함수 (에너지/힘 calculator는 이미 붙어 있어야 함).

        k_neighbors : int
            per-atom 이웃 수 (MACS-style kNN).

        cmax : float
            MACS-style displacement 상한 (기본값). disp_scale이 따로 주어지지 않으면
            base_disp_scale = cmax 로 사용됨.

        max_steps : int
            에피소드 최대 스텝 수.

        fmax_threshold : float
            구조 최적화 성공 판정 기준 (max |F| < threshold).

        bond_break_ratio : float
            초기 bond 길이에 대한 허용 배수. d / d0 > bond_break_ratio 이면
            bond penalty 부여 + done="bond".

        k_bond : float
            bond stretch penalty 강도 (선형 스프링 계수).

        max_penalty : float
            bond penalty soft cap (penalty = min(k_bond * over, max_penalty)).

        com_threshold : float
            COM drift 기준. 초과 시 done="com".

        com_lambda : float
            COM penalty 강도.

        w_force : float
            force 최적화 reward weight (현재는 log|F| 차이를 기본 reward로 사용하므로
            대부분 1.0, 필요 시 스케일링에 사용 가능).

        cutoff_factor : float
            neighbor_list cutoff = cutoff_factor * min(cell length).

        debug_bond : bool
            True이면 bond stretch 관련 디버그 로그를 추가로 남김.

        disp_scale : Optional[float]
            명시적으로 MACS-style displacement 상한을 따로 지정하고 싶을 때 사용.
            None이면 base_disp_scale = cmax 로 설정.
        """
        self.loader = atoms_loader
        self.max_steps = max_steps

        # displacement 관련
        self.cmax = cmax
        self.base_disp_scale = disp_scale if disp_scale is not None else cmax

        # termination / penalty 설정
        self.fmax_threshold = fmax_threshold
        self.com_threshold = com_threshold
        self.com_lambda = com_lambda

        self.bond_break_ratio = bond_break_ratio
        self.k_bond = k_bond
        self.max_penalty = max_penalty
        self.w_force = w_force

        # neighbor / cutoff
        self.k = k_neighbors
        self.cutoff_factor = cutoff_factor

        # debug
        self.debug_bond = debug_bond

        # 내부 state 초기화
        self.atoms = None
        self.N = 0
        self.atomic_numbers = None
        self.forces = None
        self.F_prev = None
        self.disp_last = None
        self.bond_pairs = None
        self.bond_d0 = None
        self.com_prev = None
        self.step_count = 0

        # 첫 reset 수행
        self.reset()

    # ------------------------------------------------------------------
    # Utility: flatten observation
    # ------------------------------------------------------------------
    @staticmethod
    def flatten_obs(obs: np.ndarray) -> np.ndarray:
        """
        (N, feat) → (N*feat,) float32
        """
        return obs.reshape(-1).astype(np.float32)

    # ------------------------------------------------------------------
    # PBC minimum-image helpers
    # ------------------------------------------------------------------
    def _pbc_vec(self, i: int, j: int) -> np.ndarray:
        """
        i → j 로의 PBC 최소 이미지 벡터 (cartesian) 반환.
        """
        pos = self.atoms.positions
        cell = self.atoms.cell.array

        diff = pos[j] - pos[i]
        frac = np.linalg.solve(cell.T, diff)
        frac -= np.round(frac)
        return frac @ cell

    def _pbc_vec_pos(self, pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
        """
        두 좌표 pi, pj에 대한 PBC 최소 이미지 벡터.
        """
        cell = self.atoms.cell.array
        diff = pj - pi
        frac = np.linalg.solve(cell.T, diff)
        frac -= np.round(frac)
        return frac @ cell

    # ------------------------------------------------------------------
    # MACS-style k-Nearest Neighbors (PBC)
    # ------------------------------------------------------------------
    def _kNN(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        k-NN neighbor list with PBC-aware relative positions.

        Returns
        -------
        nbr_idx : (N, k) int
        relpos  : (N, k, 3) float32
        dist    : (N, k) float32
        """
        cell_len = self.atoms.cell.lengths()
        cutoff = self.cutoff_factor * np.min(cell_len)

        i_list, j_list, S_list = neighbor_list("ijS", self.atoms, cutoff)
        N = self.N
        k = self.k
        candidates = [[] for _ in range(N)]
        cell = self.atoms.cell.array

        for ii, jj, S in zip(i_list, j_list, S_list):
            # neighbor_list가 주는 S는 셀 변환 계수
            v = (S @ cell)
            d = np.linalg.norm(v)

            candidates[ii].append((d, jj, v))
            candidates[jj].append((d, ii, -v))

        nbr_idx = np.zeros((N, k), dtype=int)
        relpos = np.zeros((N, k, 3), dtype=np.float32)
        dist = np.zeros((N, k), dtype=np.float32)

        for i in range(N):
            cand = candidates[i]
            if len(cand) < k:
                need = k - len(cand)
                cand += [(9e9, i, np.zeros(3))] * need
            cand.sort(key=lambda x: x[0])
            topk = cand[:k]
            for t, (d, j, v) in enumerate(topk):
                nbr_idx[i, t] = j
                relpos[i, t] = v
                dist[i, t] = d

        return nbr_idx, relpos, dist

    # ------------------------------------------------------------------
    # Bond detection (PBC-aware)
    # ------------------------------------------------------------------
    def _detect_bonds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Covalent radius 기반 bond detection.

        Returns
        -------
        bonds : (Nb, 2) int
        d0    : (Nb,) float
            초기 bond 길이.
        """
        Z = self.atomic_numbers
        N = self.N

        bonds = []
        d0 = []

        for i in range(N):
            Zi = Z[i]
            for j in range(i + 1, N):
                rc = covalent_radii[Zi] + covalent_radii[Z[j]] + 0.25
                v = self._pbc_vec(i, j)
                d = np.linalg.norm(v)
                if d <= rc:
                    bonds.append((i, j))
                    d0.append(d)

        if len(bonds) == 0:
            return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)

        return np.array(bonds, dtype=int), np.array(d0, dtype=float)

    # ------------------------------------------------------------------
    # Coordination number
    # ------------------------------------------------------------------
    def _coordination_numbers(self) -> np.ndarray:
        CN = np.zeros(self.N, dtype=float)
        for i, j in self.bond_pairs:
            CN[i] += 1
            CN[j] += 1
        return CN

    # ------------------------------------------------------------------
    # Local planarity
    # ------------------------------------------------------------------
    def _local_planarity(self) -> np.ndarray:
        pos = self.atoms.positions
        planar = np.zeros(self.N, dtype=float)

        for i in range(self.N):
            idxs = np.where(self.bond_pairs[:, 0] == i)[0].tolist() + \
                   np.where(self.bond_pairs[:, 1] == i)[0].tolist()

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

    # ------------------------------------------------------------------
    # Local graph radius
    # ------------------------------------------------------------------
    def _local_graph_radius(self) -> np.ndarray:
        R = np.zeros(self.N, dtype=float)
        for i, j in self.bond_pairs:
            v = self._pbc_vec(i, j)
            d = np.linalg.norm(v)
            R[i] += d
            R[j] += d
        return R

    # ------------------------------------------------------------------
    # Local stiffness
    # ------------------------------------------------------------------
    def _local_stiffness(self) -> np.ndarray:
        if self.disp_last is None:
            return np.zeros(self.N, dtype=float)
        disp_mag = np.linalg.norm(self.disp_last, axis=1) + 1e-12
        fmag = np.linalg.norm(self.forces, axis=1)
        return fmag / disp_mag

    # ------------------------------------------------------------------
    # Torsion
    # ------------------------------------------------------------------
    def _torsion(self) -> np.ndarray:
        tors = np.zeros(self.N, dtype=float)
        pos = self.atoms.positions

        for i in range(self.N):
            idxs = np.where(self.bond_pairs[:, 0] == i)[0].tolist() + \
                   np.where(self.bond_pairs[:, 1] == i)[0].tolist()

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

            p0 = pos[a]
            p1 = pos[i]
            p2 = pos[b]
            p3 = pos[c]

            b0 = -self._pbc_vec_pos(p0, p1)
            b1 = self._pbc_vec_pos(p1, p2)
            b2 = self._pbc_vec_pos(p2, p3)

            n1 = np.cross(b0, b1)
            n2 = np.cross(b1, b2)

            if np.linalg.norm(n1) < 1e-9 or np.linalg.norm(n2) < 1e-9:
                continue

            x = np.dot(n1, n2)
            y = np.dot(np.cross(n1, n2), b1 / (np.linalg.norm(b1) + 1e-12))
            tors[i] = np.arctan2(y, x)

        return tors

    # ------------------------------------------------------------------
    # Local stress (simple proxy)
    # ------------------------------------------------------------------
    def _local_stress(self, CN: np.ndarray) -> np.ndarray:
        fmag = np.linalg.norm(self.forces, axis=1)
        return fmag * CN

    # ------------------------------------------------------------------
    # SBU identifier (very coarse metal id)
    # ------------------------------------------------------------------
    def _sbu_id(self) -> np.ndarray:
        Z = self.atomic_numbers
        metals = {20, 22, 23, 24, 25, 26, 27, 28, 29, 40, 42, 44}
        arr = np.zeros(self.N, dtype=float)
        cid = 1
        for i in range(self.N):
            if Z[i] in metals:
                arr[i] = cid
                cid += 1
        return arr

    # ------------------------------------------------------------------
    # TOPOLOGY HELPERS: roles, motifs, pore features
    # ------------------------------------------------------------------
    def _build_neighbor_dict(self):
        """
        bond_pairs 기반 neighbor list (adjacency) 생성.
        """
        neigh = [[] for _ in range(self.N)]
        for i, j in self.bond_pairs:
            neigh[i].append(j)
            neigh[j].append(i)
        return neigh

    def _atom_roles(
        self,
        CN: Optional[np.ndarray] = None,
        planar: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Atom-level role flags:
            [is_metal,
             is_linker,
             is_mu2_O,
             is_mu3_O,
             is_carboxylate_O,
             is_aromatic_C,
             is_terminal]
        shape = (N, 7)
        """
        Z = self.atomic_numbers
        if CN is None:
            CN = self._coordination_numbers()
        if planar is None:
            planar = self._local_planarity()

        neigh = self._build_neighbor_dict()
        roles = np.zeros((self.N, 7), dtype=np.float32)

        metals = {20, 22, 23, 24, 25, 26, 27, 28, 29, 40, 42, 44}

        for i in range(self.N):
            z = Z[i]
            nbs = neigh[i]
            cn_i = CN[i]

            is_metal = z in metals
            is_O = (z == 8)
            is_N = (z == 7)
            is_C = (z == 6)

            # metal flag
            if is_metal:
                roles[i, 0] = 1.0

            # terminal: non-metal & CN == 1
            if (not is_metal) and cn_i == 1:
                roles[i, 6] = 1.0

            # 기본 linker: metal이 아닌 원자 (rough)
            if not is_metal and cn_i >= 2:
                roles[i, 1] = 1.0

            # μ2-O / μ3-O : O가 metal과 2개 또는 3개 이상 결합
            if is_O:
                metal_neighbors = 0
                for j in nbs:
                    if Z[j] in metals:
                        metal_neighbors += 1
                if metal_neighbors == 2:
                    roles[i, 2] = 1.0
                elif metal_neighbors >= 3:
                    roles[i, 3] = 1.0

                # carboxylate O: C와 결합된 O이고, 해당 C가 O를 2개 이상 거느림
                for j in nbs:
                    if Z[j] == 6:  # carbon neighbor
                        O_cnt = sum(1 for k in neigh[j] if Z[k] == 8)
                        if O_cnt >= 2:
                            roles[i, 4] = 1.0
                            break

            # aromatic C (approximation):
            #  - C atom
            #  - C neighbors >= 2
            #  - local_planarity high
            if is_C:
                c_neighbors = sum(1 for j in nbs if Z[j] == 6)
                if c_neighbors >= 2 and planar[i] > 0.9:
                    roles[i, 5] = 1.0

        return roles

    def _bond_motif_feats(self) -> np.ndarray:
        """
        Bond motif participation flags per atom:
            [has_Metal_O,
             has_Metal_N,
             has_carboxylate_OCO,
             has_mu_O_Metal,
             has_aromatic_CC]
        shape = (N, 5)
        """
        Z = self.atomic_numbers
        metals = {20, 22, 23, 24, 25, 26, 27, 28, 29, 40, 42, 44}
        feat = np.zeros((self.N, 5), dtype=np.float32)

        neigh = self._build_neighbor_dict()

        # Pre-detect aromatic C (간단히 다시 한 번 정의)
        planar = self._local_planarity()
        is_aromatic_C = np.zeros(self.N, dtype=bool)
        for i in range(self.N):
            if Z[i] != 6:
                continue
            c_neighbors = sum(1 for j in neigh[i] if Z[j] == 6)
            if c_neighbors >= 2 and planar[i] > 0.9:
                is_aromatic_C[i] = True

        for i, j in self.bond_pairs:
            zi, zj = Z[i], Z[j]

            # Metal–O
            if (zi in metals and zj == 8) or (zj in metals and zi == 8):
                feat[i, 0] = 1.0
                feat[j, 0] = 1.0

            # Metal–N
            if (zi in metals and zj == 7) or (zj in metals and zi == 7):
                feat[i, 1] = 1.0
                feat[j, 1] = 1.0

            # μ2/μ3-O–Metal
            if zi == 8 and zj in metals:
                feat[i, 3] = 1.0
                feat[j, 3] = 1.0
            elif zj == 8 and zi in metals:
                feat[i, 3] = 1.0
                feat[j, 3] = 1.0

        # Carboxylate O–C–O motif: C with >=2 O neighbors + 그 O들
        for i in range(self.N):
            if Z[i] != 6:
                continue
            O_neighbors = [j for j in neigh[i] if Z[j] == 8]
            if len(O_neighbors) >= 2:
                # mark the carbon & its O neighbors
                feat[i, 2] = 1.0
                for o in O_neighbors:
                    feat[o, 2] = 1.0

        # Aromatic C–C motif: aromatic C와 그 C neighbors
        for i in range(self.N):
            if not is_aromatic_C[i]:
                continue
            feat[i, 4] = 1.0
            for j in neigh[i]:
                if Z[j] == 6:
                    feat[j, 4] = 1.0

        return feat

    def _pore_side_feats(self) -> np.ndarray:
        """
        Pore-side / fractional coordinate features:
            [frac_x, frac_y, frac_z,
             dist_center_frac,
             boundary_proximity,
             is_pore_lining_flag]
        shape = (N, 6)
        """
        cell = self.atoms.cell.array
        pos = self.atoms.positions

        # fractional coordinates in [0,1)
        frac = np.linalg.solve(cell.T, pos.T).T  # (N,3)
        frac = frac - np.floor(frac)

        center = np.array([0.5, 0.5, 0.5])
        vec_center = frac - center
        dist_center = np.linalg.norm(vec_center, axis=1)

        # distance to nearest periodic boundary (0 or 1)
        # small value → boundary 근처
        boundary_prox = np.min(
            np.stack([frac, 1.0 - frac], axis=-1), axis=-1
        )
        min_boundary = np.min(boundary_prox, axis=1)

        # 대략적인 pore-lining flag:
        #  - center에서 어느 정도 떨어져 있고
        #  - boundary에 너무 붙어있지는 않은 값
        is_pore_lining = (
            (dist_center > 0.3) & (dist_center < 0.9) &
            (min_boundary > 0.1)
        ).astype(np.float32)

        feats = np.zeros((self.N, 6), dtype=np.float32)
        feats[:, 0:3] = frac.astype(np.float32)
        feats[:, 3] = dist_center.astype(np.float32)
        feats[:, 4] = min_boundary.astype(np.float32)
        feats[:, 5] = is_pore_lining

        return feats

    # ------------------------------------------------------------------
    # Build per-atom feature f_i (core)
    # ------------------------------------------------------------------
    def _build_f(self) -> np.ndarray:
        """
        Per-atom feature f_i 생성.

        구성:
            - force 방향 (unit vector)
            - log|F_i|
            - F_i (vector)
            - ΔF_i (현재 - 이전 force)
            - 이전 step displacement (disp_last)
            - CN, planar, graphR, stiffness, torsion, stress
            - SBU id (metal cluster coarse id)
            - Global force stats: mean|F|, std|F|
            - Atom roles (MOF topology)
            - Bond motifs (Metal–O, Metal–N, OCO, μ-O–Metal, aromatic C–C)
            - Pore-side features (fractional coordinates, center/boundary 거리, pore-lining flag)
        """
        F = self.forces.astype(np.float32)  # (N,3)
        f_norm = np.linalg.norm(F, axis=1) + 1e-12
        f_unit = F / f_norm[:, None]

        # ΔF history
        if self.F_prev is None:
            dF = np.zeros_like(F, dtype=np.float32)
        else:
            dF = (F - self.F_prev).astype(np.float32)

        # displacement history
        if self.disp_last is None:
            disp_prev = np.zeros_like(F, dtype=np.float32)
        else:
            disp_prev = self.disp_last.astype(np.float32)

        CN = self._coordination_numbers()
        planar = self._local_planarity()
        graphR = self._local_graph_radius()
        stiff = self._local_stiffness()
        tors = self._torsion()
        stress = self._local_stress(CN)
        sbu = self._sbu_id()

        # Global stats
        gF_mean = float(np.mean(f_norm))
        gF_std = float(np.std(f_norm) + 1e-12)
        logF = np.log(f_norm)

        # MOF topology-specific features
        roles = self._atom_roles(CN=CN, planar=planar)        # (N,7)
        motifs = self._bond_motif_feats()                     # (N,5)
        pore_feats = self._pore_side_feats()                  # (N,6)

        # Broadcast global stats
        gF_mean_col = np.full((self.N, 1), gF_mean, dtype=np.float32)
        gF_std_col = np.full((self.N, 1), gF_std, dtype=np.float32)

        # Concatenate all
        f = np.concatenate(
            [
                f_unit,                    # 3
                logF[:, None],             # 1
                F,                         # 3
                dF,                        # 3
                disp_prev,                 # 3
                CN[:, None],               # 1
                planar[:, None],           # 1
                graphR[:, None],           # 1
                stiff[:, None],            # 1
                tors[:, None],             # 1
                stress[:, None],           # 1
                sbu[:, None],              # 1
                gF_mean_col,               # 1
                gF_std_col,                # 1
                roles,                     # 7
                motifs,                    # 5
                pore_feats,                # 6
            ],
            axis=1,
        ).astype(np.float32)

        # shape = (N, feat_dim)
        return f

    # ------------------------------------------------------------------
    # Build observation with k-NN neighbors
    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        """
        MACS-style per-atom observation with neighbor info.

        Returns
        -------
        obs : (N, FEAT) float32
            각 row = [f_i, f_neighbors(flat), relpos_neighbors(flat), dist_neighbors(flat)]
        """
        f = self._build_f()
        nbr_idx, relpos, dist = self._kNN()

        N, k = self.N, self.k
        Fdim = f.shape[1]

        obs_dim = Fdim + k * Fdim + k * 3 + k
        obs = np.zeros((N, obs_dim), dtype=np.float32)

        for i in range(N):
            neigh_f = f[nbr_idx[i]].reshape(-1)  # (k*Fdim,)
            obs[i] = np.concatenate(
                [
                    f[i],                        # center features
                    neigh_f,                     # neighbor features
                    relpos[i].reshape(-1),       # k*3
                    dist[i].reshape(-1),         # k
                ]
            )

        return obs

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        """
        새 Atoms 불러와서 초기 forces, bonds, features 계산 후
        (obs_atom, obs_global_flat) 반환.
        """
        self.atoms = self.loader()
        self.N = len(self.atoms)
        self.atomic_numbers = np.array(
            [a.number for a in self.atoms], dtype=int
        )

        self.forces = self.atoms.get_forces().astype(np.float32)
        self.bond_pairs, self.bond_d0 = self._detect_bonds()

        self.F_prev = None
        self.disp_last = None
        self.com_prev = self.atoms.positions.mean(axis=0)
        self.step_count = 0

        obs_atom = self._obs()
        obs_global = self.flatten_obs(obs_atom)

        logger.debug(
            f"[MOFEnv.reset] N={self.N}, bonds={len(self.bond_pairs)}, "
            f"obs_atom_dim={obs_atom.shape[1]}"
        )
        return obs_atom, obs_global

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action_u: np.ndarray):
        """
        Parameters
        ----------
        action_u : (N, 3) ndarray
            각 원자별 [-1,1]^3 범위의 displacement 방향 (policy output).

        Returns
        -------
        obs_atom      : (N, FEAT)
        obs_global    : (N*FEAT,)
        reward_scalar : float
        done          : bool
        done_reason   : str or None
        Etot_stub     : float (에너지 placeholder, 필요 시 DFT/ML 에너지 사용)
        Fmax          : float (현재 max |F|)
        """
        self.step_count += 1

        F_old = self.forces
        old_norm = np.linalg.norm(F_old, axis=1) + 1e-12

        # --------------------------------------------------------------
        # 0) MACS Eq.4 scaling: c_i = min( |F_i|, base_disp_scale )
        # --------------------------------------------------------------
        c_i = np.minimum(old_norm, self.base_disp_scale)

        u = np.clip(action_u, -1.0, 1.0)
        disp = c_i[:, None] * u
        self.disp_last = disp.astype(np.float32)

        # apply displacement
        self.atoms.positions += disp

        new_F = self.atoms.get_forces().astype(np.float32)
        new_norm = np.linalg.norm(new_F, axis=1)

        # --------------------------------------------------------------
        # 1) Force-based reward (log|F| 감소)
        # --------------------------------------------------------------
        R_vec = np.log(old_norm) - np.log(new_norm + 1e-12)
        R_vec = np.clip(R_vec, -5.0, 5.0)
        reward_scalar = float(np.mean(R_vec))

        # --------------------------------------------------------------
        # 2) COM drift penalty + termination
        # --------------------------------------------------------------
        com_new = self.atoms.positions.mean(axis=0)
        dCOM = np.linalg.norm(com_new - self.com_prev)
        reward_scalar -= self.com_lambda * np.tanh(4.0 * dCOM)
        self.com_prev = com_new.copy()

        if dCOM > self.com_threshold:
            self.F_prev = new_F.copy()
            self.forces = new_F
            obs_atom = self._obs()
            obs_global = self.flatten_obs(obs_atom)
            logger.debug(
                f"[MOFEnv.step] done='com', step={self.step_count}, "
                f"dCOM={dCOM:.4f}, reward={reward_scalar:.4f}"
            )
            return (
                obs_atom,
                obs_global,
                reward_scalar,
                True,
                "com",
                0.0,
                float(new_norm.max()),
            )

        # --------------------------------------------------------------
        # 3) Bond stretch penalty + termination
        #    - ratio = d / d0
        #    - if ratio > bond_break_ratio:
        #        over    = ratio - bond_break_ratio
        #        penalty = min(k_bond * over, max_penalty)
        #        reward -= penalty
        #        done = True, reason="bond"
        # --------------------------------------------------------------
        Fmax = float(new_norm.max())
        for idx, (i, j) in enumerate(self.bond_pairs):
            v = self._pbc_vec(i, j)
            d = np.linalg.norm(v)
            ratio = d / (self.bond_d0[idx] + 1e-12)

            if ratio > self.bond_break_ratio:
                over = ratio - self.bond_break_ratio
                penalty = self.k_bond * over
                if penalty > self.max_penalty:
                    penalty = self.max_penalty
                reward_scalar -= penalty

                if self.debug_bond:
                    logger.debug(
                        "[MOFEnv.step][bond] step=%d, i=%d, j=%d, "
                        "d0=%.4f, d=%.4f, ratio=%.3f, over=%.3f, "
                        "penalty=%.3f",
                        self.step_count,
                        i,
                        j,
                        self.bond_d0[idx],
                        d,
                        ratio,
                        over,
                        penalty,
                    )

                self.F_prev = new_F.copy()
                self.forces = new_F
                obs_atom = self._obs()
                obs_global = self.flatten_obs(obs_atom)
                logger.debug(
                    f"[MOFEnv.step] done='bond', step={self.step_count}, "
                    f"ratio={ratio:.3f}, reward={reward_scalar:.4f}"
                )
                return (
                    obs_atom,
                    obs_global,
                    reward_scalar,
                    True,
                    "bond",
                    0.0,
                    Fmax,
                )

        # --------------------------------------------------------------
        # 4) Fmax termination
        # --------------------------------------------------------------
        if Fmax < self.fmax_threshold:
            self.F_prev = new_F.copy()
            self.forces = new_F
            obs_atom = self._obs()
            obs_global = self.flatten_obs(obs_atom)
            logger.debug(
                f"[MOFEnv.step] done='fmax', step={self.step_count}, "
                f"Fmax={Fmax:.4e}, reward={reward_scalar:.4f}"
            )
            return (
                obs_atom,
                obs_global,
                reward_scalar,
                True,
                "fmax",
                0.0,
                Fmax,
            )

        # --------------------------------------------------------------
        # 5) max-step termination
        # --------------------------------------------------------------
        if self.step_count >= self.max_steps:
            self.F_prev = new_F.copy()
            self.forces = new_F
            obs_atom = self._obs()
            obs_global = self.flatten_obs(obs_atom)
            logger.debug(
                f"[MOFEnv.step] done='max_steps', step={self.step_count}, "
                f"Fmax={Fmax:.4e}, reward={reward_scalar:.4f}"
            )
            return (
                obs_atom,
                obs_global,
                reward_scalar,
                True,
                "max_steps",
                0.0,
                Fmax,
            )

        # --------------------------------------------------------------
        # 6) Normal step
        # --------------------------------------------------------------
        self.F_prev = new_F.copy()
        self.forces = new_F
        obs_atom = self._obs()
        obs_global = self.flatten_obs(obs_atom)

        logger.debug(
            f"[MOFEnv.step] step={self.step_count}, "
            f"Fmax={Fmax:.4e}, reward={reward_scalar:.4f}"
        )

        return (
            obs_atom,
            obs_global,
            reward_scalar,
            False,
            None,
            0.0,
            Fmax,
        )
