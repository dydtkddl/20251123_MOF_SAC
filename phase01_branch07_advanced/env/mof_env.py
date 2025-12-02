# env/mof_env.py
# ============================================================
# MACS-MOF Environment (Phase2, 4D gate + disp, v2)
# - Per-atom SAC용 MOFEnv
# - 4D action: [gate_raw, dx, dy, dz]
#   * gate_raw ∈ [-1,1] → Env 내부에서 [0,1]로 매핑
# - Force 기반 reward + COM penalty + bond penalty
# - Step-limit fail penalty + time penalty + terminal bonus
# - (옵션) Graph Laplacian 기반 mode smoothing (low-pass)
#   -> get_graph_mode_basis, apply_graph_mode_smoothing 사용
# - (옵션) 1A/1B/3A/3B action post-processing / reward 구성
# ============================================================

import logging
from typing import Dict, List

import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list

from env.modes import get_graph_mode_basis, apply_graph_mode_smoothing
from env.action_postproc import (
    compute_force_increase_penalty,
    compute_fd_direction_penalty,
    project_to_bond_subspace,
    apply_local_frame_scaling,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class MOFEnv:
    # ============================================================
    # INIT
    # ============================================================
    def __init__(
        self,
        atoms_loader,
        k_neighbors: int = 12,
        cmax: float = 0.4,
        max_steps: int = 300,
        fmax_threshold: float = 0.05,
        bond_break_ratio: float = 2.4,
        k_bond: float = 3.0,
        max_penalty: float = 10.0,
        debug_bond: bool = False,
        # ---------- Phase2: perturb 옵션 ----------
        random_perturb: bool = False,
        perturb_sigma: float = 0.05,
        max_perturb: float = 0.3,
        # ---------- 종료/시간 관련 하이퍼파라미터 ----------
        terminal_bonus_base: float = 10.0,
        time_penalty: float = 0.025,
        fail_penalty: float = 15.0,
        # ---------- Mode-based smoothing (v1) ----------
        use_mode_smoothing: bool = False,
        mode_type: str = "graph_eig",
        mode_num_modes: int = 16,
        mode_eig_cutoff: float = 4.0,
        mode_id: str = None,
        # ---------- Advanced penalties / local frames ----------
        use_force_increase_penalty: bool = False,   # 1A
        lambda_force_up: float = 2.0,
        use_fd_direction_penalty: bool = False,     # 1B
        lambda_fd_penalty: float = 1.0,
        use_bond_projection: bool = False,          # 3A
        use_local_frame: bool = False,              # 3B
        radial_scale: float = 1.0,
        tangent_scale: float = 1.0,
    ):
        """
        Parameters
        ----------
        atoms_loader : callable
            매 reset()마다 깨끗한 ASE Atoms를 반환하는 함수.
        """

        self.atoms_loader = atoms_loader

        # Neighbors / action scaling
        self.k = k_neighbors
        self.cmax = cmax
        self.max_steps = max_steps
        self.fmax_threshold = fmax_threshold

        # Bond-related
        self.bond_break_ratio = bond_break_ratio
        self.k_bond = k_bond
        self.max_penalty = max_penalty
        self.debug_bond = debug_bond

        # COM control (stabilized)
        self.com_threshold = 0.30
        self.com_lambda = 20.0  # previously 100 × 0.1=10, now stabilized

        # Phase2: perturb options
        self.random_perturb = random_perturb
        self.perturb_sigma = perturb_sigma
        self.max_perturb = max_perturb

        # Termination / time hyperparameters
        self.terminal_bonus_base = terminal_bonus_base
        self.time_penalty = time_penalty
        self.fail_penalty = fail_penalty

        # Mode-based smoothing 옵션 (v1: graph_eig만)
        self.use_mode_smoothing = use_mode_smoothing
        self.mode_type = mode_type
        self.mode_num_modes = mode_num_modes
        self.mode_eig_cutoff = mode_eig_cutoff
        self.mode_id = mode_id
        self.mode_U = None  # (N, K) or None

        # Advanced penalties / local frame options
        self.use_force_increase_penalty = use_force_increase_penalty
        self.lambda_force_up = lambda_force_up
        self.use_fd_direction_penalty = use_fd_direction_penalty
        self.lambda_fd_penalty = lambda_fd_penalty
        self.use_bond_projection = use_bond_projection
        self.use_local_frame = use_local_frame
        self.radial_scale = radial_scale
        self.tangent_scale = tangent_scale

        # Bond-direction basis for 3A/3B (reset 시점에서 구성)
        self.bond_dirs = None  # (N, 3) or None

        # 내부 상태
        self.feature_dim = None

        # Reward component logging (for train loop)
        self.last_r_f_mean = 0.0          # mean log(force) reward
        self.last_com_penalty = 0.0       # COM penalty (scalar)
        self.last_bond_penalty = 0.0      # sum of bond penalties
        self.last_time_penalty = 0.0      # time_penalty (scalar)
        self.last_fail_penalty = 0.0      # fail_penalty (scalar)
        self.last_terminal_bonus = 0.0    # success terminal bonus (scalar)
        self.last_force_up_penalty = 0.0  # 1A scalar (mean)
        self.last_fd_penalty = 0.0        # 1B scalar (mean)
        self.last_reward_mean = 0.0       # final reward mean

    # ============================================================
    # True Bonds from reference structure
    # ============================================================
    def _detect_true_bonds(self, atoms):
        """
        neighbor_list 기반으로 'reference' 구조에서의 진짜 bond를 정의.
        - cutoff=4.0 Å 내에서 covalent radius 합 + 0.4 Å 이하인 것만 인정
        """
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
    # Aromatic detection (simple 6-cycle finder)
    # ============================================================
    def _detect_aromatic_nodes(self, adj: Dict[int, List[int]], Z: np.ndarray):
        """
        매우 단순한 6원자 고리 기반 aromatic node 탐지.
        - C6 ring, 각 노드 Z=6, degree <= 3 인 경우 aromatic으로 마킹.
        """
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
    # Role flags
    # ============================================================
    def _assign_metal_flags(self, Z: np.ndarray) -> np.ndarray:
        MOF_METALS = {12, 13, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 40, 72}
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
    # Phase2: reset-time perturbation
    # ============================================================
    def _apply_random_perturbation(self):
        """
        Apply small Gaussian noise and clip per-atom displacement.

        - sigma = self.perturb_sigma
        - per-atom ||delta|| <= self.max_perturb
        """
        if (not self.random_perturb) or self.perturb_sigma <= 0.0:
            return

        pos = self.atoms.positions.copy()

        # 1) basic Gaussian noise
        delta = np.random.normal(
            loc=0.0,
            scale=self.perturb_sigma,
            size=pos.shape,
        )

        # 2) per-atom max_perturb clipping
        if self.max_perturb is not None:
            norms = np.linalg.norm(delta, axis=1, keepdims=True)  # (N,1)
            norms_safe = np.maximum(norms, 1e-12)
            scale = np.minimum(1.0, self.max_perturb / norms_safe)
            delta = delta * scale

        self.atoms.positions = pos + delta

        max_disp = np.linalg.norm(delta, axis=1).max()
        logger.info(
            "[PERTURB] sigma=%.3f Å, max_disp=%.3f Å (max_perturb=%.3f Å)",
            self.perturb_sigma,
            max_disp,
            self.max_perturb,
        )

    # ============================================================
    # RESET
    # ============================================================
    def reset(self):
        # 1) clean QMOF structure
        self.atoms = self.atoms_loader()
        self.N = len(self.atoms)

        Z = np.array([a.number for a in self.atoms])
        self.covalent_radii = np.array(
            [covalent_radii[z] for z in Z], dtype=np.float32
        )

        # 2) bonds & bond_d0 from reference structure
        self.bond_pairs, self.bond_d0 = self._detect_true_bonds(self.atoms)
        logger.info("[INIT] Detected true bonds = %d", len(self.bond_pairs))

        # 3) adjacency & role flags
        self.adj = {i: [] for i in range(self.N)}
        for a, b in self.bond_pairs:
            self.adj[a].append(b)
            self.adj[b].append(a)

        aromatic_nodes = self._detect_aromatic_nodes(self.adj, Z)
        self.is_aromatic = np.zeros(self.N, float)
        self.is_aromatic[list(aromatic_nodes)] = 1.0

        self.is_metal = self._assign_metal_flags(Z)
        self.is_carboxylate_O = self._detect_carboxylate_O(
            Z, self.adj, self.is_metal
        )
        self.is_mu2O, self.is_mu3O = self._detect_mu_oxygens(
            Z, self.adj, self.is_metal
        )

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
                and Z[i] in [6, 7]
            ):
                self.is_linker[i] = 1.0

        # bond type count features (per atom, 6 dim)
        self.bond_types = np.zeros((self.N, 6), float)

        for a, b in self.bond_pairs:
            Za, Zb = Z[a], Z[b]

            # M–O
            if self.is_metal[a] and Zb == 8:
                self.bond_types[a][0] += 1
            if self.is_metal[b] and Za == 8:
                self.bond_types[b][0] += 1

            # M–N
            if self.is_metal[a] and Zb == 7:
                self.bond_types[a][1] += 1
            if self.is_metal[b] and Za == 7:
                self.bond_types[b][1] += 1

            # carboxylate O 연결
            if self.is_carboxylate_O[a]:
                self.bond_types[b][2] += 1
            if self.is_carboxylate_O[b]:
                self.bond_types[a][2] += 1

            # aromatic C–C
            if self.is_aromatic_C[a] and self.is_aromatic_C[b]:
                self.bond_types[a][3] += 1
                self.bond_types[b][3] += 1

            # μ2 / μ3 산소
            if self.is_mu2O[a]:
                self.bond_types[b][4] += 1
            if self.is_mu3O[a]:
                self.bond_types[b][5] += 1
            if self.is_mu2O[b]:
                self.bond_types[a][4] += 1
            if self.is_mu3O[b]:
                self.bond_types[a][5] += 1

        # 4) Bond-direction basis for 3A/3B (reference 구조 기준)
        self.bond_dirs = np.zeros((self.N, 3), dtype=np.float32)
        for a, b in self.bond_pairs:
            rel = self._rel_vec(a, b)
            norm = np.linalg.norm(rel)
            if norm < 1e-8:
                continue
            unit = rel / norm
            # a: a→b 방향, b: b→a 방향 (부호 반대)
            self.bond_dirs[a] += unit
            self.bond_dirs[b] -= unit

        for i in range(self.N):
            n = np.linalg.norm(self.bond_dirs[i])
            if n > 1e-8:
                self.bond_dirs[i] = (self.bond_dirs[i] / n).astype(np.float32)
            else:
                self.bond_dirs[i] = np.zeros(3, dtype=np.float32)

        # 5) Phase2: apply perturbation
        self._apply_random_perturbation()

        # 6) initialize forces based on perturbed structure
        self.forces = self.atoms.get_forces().astype(np.float32)
        self.prev_forces = np.zeros_like(self.forces)
        self.prev_disp = np.zeros_like(self.forces)

        self.step_count = 0
        self.COM_prev = self.atoms.positions.mean(axis=0).astype(np.float32)

        # feature_dim after force init
        self.feature_dim = len(self._make_feature(0))

        # reset reward components
        self.last_r_f_mean = 0.0
        self.last_com_penalty = 0.0
        self.last_bond_penalty = 0.0
        self.last_time_penalty = 0.0
        self.last_fail_penalty = 0.0
        self.last_terminal_bonus = 0.0
        self.last_force_up_penalty = 0.0
        self.last_fd_penalty = 0.0
        self.last_reward_mean = 0.0

        # ---------- Mode basis 준비 (graph_eig) ----------
        self.mode_U = None
        if self.use_mode_smoothing and self.mode_type == "graph_eig":
            try:
                self.mode_U = get_graph_mode_basis(
                    self.atoms,
                    num_modes=self.mode_num_modes,
                    cutoff=self.mode_eig_cutoff,
                    mode_id=self.mode_id,
                )
                if self.mode_U is not None:
                    logger.info(
                        "[MODE] Basis ready: id=%s, N=%d, num_modes=%d",
                        str(self.mode_id),
                        self.N,
                        self.mode_U.shape[1],
                    )
            except Exception as e:
                logger.warning(
                    "[MODE] Failed to build graph mode basis for id=%s: %s",
                    str(self.mode_id),
                    repr(e),
                )
                self.mode_U = None

        return self._obs()

    # ============================================================
    # PBC-relative vector
    # ============================================================
    def _rel_vec(self, i: int, j: int) -> np.ndarray:
        disp = self.atoms.positions[j] - self.atoms.positions[i]
        cell = self.atoms.cell.array
        frac = np.linalg.solve(cell.T, disp)
        frac -= np.round(frac)
        return frac @ cell

    # ============================================================
    # Hops 1–3 (for neighbor selection)
    # ============================================================
    def _get_hop_sets(self, idx: int, max_hop: int = 3):
        visited = set([idx])
        frontier = [idx]
        hop_map = {1: [], 2: [], 3: []}

        for hop in range(1, max_hop + 1):
            nxt_frontier = []
            for node in frontier:
                for nxt in self.adj[node]:
                    if nxt not in visited:
                        visited.add(nxt)
                        nxt_frontier.append(nxt)
                        hop_map[hop].append(nxt)
            frontier = nxt_frontier

        return hop_map

    # ============================================================
    # Per-atom feature
    # ============================================================
    def _make_feature(self, idx: int) -> np.ndarray:
        ri = self.covalent_radii[idx]
        gi = self.forces[idx]
        gprev = self.prev_forces[idx]

        gnorm = max(np.linalg.norm(gi), 1e-12)

        core = np.concatenate(
            [
                np.array([ri, min(gnorm, self.cmax), np.log(gnorm + 1e-6)]),
                gi,
                self.prev_disp[idx],
                gi - gprev,
            ]
        )

        roles = np.array(
            [
                self.is_aromatic[idx],
                self.is_metal[idx],
                self.is_linker[idx],
                self.is_carboxylate_O[idx],
                self.is_mu2O[idx],
                self.is_mu3O[idx],
            ]
        )

        return np.concatenate([core, roles, self.bond_types[idx]])

    # ============================================================
    # Observation (per-atom + k neighbors)
    # ============================================================
    def _obs(self) -> np.ndarray:
        obs_list = []

        for i in range(self.N):
            fi = self._make_feature(i)
            hop_sets = self._get_hop_sets(i)

            selected = []

            # 1-hop 우선
            for j in hop_sets[1]:
                if len(selected) < self.k:
                    selected.append(j)

            # 그 다음 2-hop
            for j in hop_sets[2]:
                if len(selected) < self.k:
                    selected.append(j)

            # 3-hop에서 나머지 채우기 (랜덤 샘플링)
            remain = self.k - len(selected)
            if remain > 0 and len(hop_sets[3]) > 0:
                cand = hop_sets[3]
                if len(cand) <= remain:
                    selected += cand
                else:
                    selected += list(np.random.choice(cand, remain, False))

            # 부족하면 dummy neighbor(None)로 채움
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
    # Reward component logger
    # ============================================================
    def _set_last_reward_components(
        self,
        r_f_mean: float,
        com_penalty: float,
        bond_penalty: float,
        time_penalty: float,
        fail_penalty: float,
        terminal_bonus: float,
        reward_vec: np.ndarray,
        force_up_penalty: float = 0.0,
        fd_penalty: float = 0.0,
    ):
        """
        reward_vec: per-atom reward (N,)
        나머지는 scalar component
        """
        self.last_r_f_mean = float(r_f_mean)
        self.last_com_penalty = float(com_penalty)
        self.last_bond_penalty = float(bond_penalty)
        self.last_time_penalty = float(time_penalty)
        self.last_fail_penalty = float(fail_penalty)
        self.last_terminal_bonus = float(terminal_bonus)
        self.last_force_up_penalty = float(force_up_penalty)
        self.last_fd_penalty = float(fd_penalty)
        self.last_reward_mean = float(np.mean(reward_vec))

    # ============================================================
    # STEP  (4D action: gate + disp, v2)
    # ============================================================
    def step(self, action: np.ndarray):
        """
        Parameters
        ----------
        action : np.ndarray
            Shape (N, act_dim)
            - act_dim == 4: [gate_raw, dx, dy, dz]
            - act_dim == 3: [dx, dy, dz] (gate=1.0로 간주, backward compatibility)

        Returns
        -------
        obs_next : np.ndarray
            (N, obs_dim)
        reward : np.ndarray
            (N,)
        done : bool
        """
        self.step_count += 1

        # -----------------------------
        # 0) Action shape & clipping
        # -----------------------------
        action = np.clip(action, -1.0, 1.0)

        if action.ndim != 2 or action.shape[0] != self.N:
            raise ValueError(
                f"action shape mismatch: expected (N={self.N}, act_dim), "
                f"got {action.shape}"
            )

        if action.shape[1] == 4:
            gate_raw = action[:, 0:1]        # (N,1)
            disp_raw = action[:, 1:4]        # (N,3)
            gate = (gate_raw + 1.0) * 0.5    # [-1,1] → [0,1]
        elif action.shape[1] == 3:
            gate = np.ones((self.N, 1), dtype=np.float32)
            disp_raw = action
        else:
            raise ValueError(
                f"Unsupported action dimension: {action.shape[1]} "
                f"(expected 3 or 4)"
            )

        # -----------------------------
        # 1) Force-adaptive displacement
        # -----------------------------
        gnorm = np.linalg.norm(self.forces, axis=1)
        scale = np.minimum(gnorm, self.cmax).reshape(-1, 1)

        # gate ∈ [0,1], disp_raw ∈ [-1,1]^3
        disp = 0.003 * gate * disp_raw * (scale / self.cmax)

        # ---------- Mode-based smoothing (graph mode) ----------
        if self.use_mode_smoothing and self.mode_U is not None:
            orig_mean = float(np.linalg.norm(disp, axis=1).mean())
            disp = apply_graph_mode_smoothing(disp, self.mode_U)
            smooth_mean = float(np.linalg.norm(disp, axis=1).mean())
            logger.debug(
                "[MODE] step=%d mean_disp_norm(before)=%.6f -> (after)=%.6f",
                self.step_count,
                orig_mean,
                smooth_mean,
            )

        # ---------- 3A: Bond-subspace projection (optional) ----------
        if self.use_bond_projection and self.bond_dirs is not None:
            before = float(np.linalg.norm(disp, axis=1).mean())
            disp = project_to_bond_subspace(disp, self.bond_dirs)
            after = float(np.linalg.norm(disp, axis=1).mean())
            logger.debug(
                "[MODE-3A] step=%d bond_projection mean_norm: %.6f -> %.6f",
                self.step_count,
                before,
                after,
            )

        # ---------- 3B: Local frame scaling (optional) ----------
        if self.use_local_frame and self.bond_dirs is not None:
            before = float(np.linalg.norm(disp, axis=1).mean())
            disp = apply_local_frame_scaling(
                disp,
                self.bond_dirs,
                radial_scale=self.radial_scale,
                tangent_scale=self.tangent_scale,
            )
            after = float(np.linalg.norm(disp, axis=1).mean())
            logger.debug(
                "[MODE-3B] step=%d local_frame (rad=%.3f, tan=%.3f) "
                "mean_norm: %.6f -> %.6f",
                self.step_count,
                self.radial_scale,
                self.tangent_scale,
                before,
                after,
            )

        # 실제 좌표 업데이트
        self.atoms.positions += disp

        new_forces = self.atoms.get_forces().astype(np.float32)

        old_norm = np.maximum(np.linalg.norm(self.forces, axis=1), 1e-12)
        new_norm = np.maximum(np.linalg.norm(new_forces, axis=1), 1e-12)

        # -----------------------------
        # 2) Force reward (×10)
        # -----------------------------
        r_f = 10.0 * (np.log(old_norm + 1e-6) - np.log(new_norm + 1e-6))
        reward = r_f.copy()  # (N,)
        r_f_mean = float(np.mean(r_f))

        # -----------------------------
        # 2A) Optional: force increase penalty (1A)
        # -----------------------------
        force_up_pen_scalar = 0.0
        if self.use_force_increase_penalty and self.lambda_force_up > 0.0:
            fup_vec, force_up_pen_scalar = compute_force_increase_penalty(
                self.forces,
                new_forces,
                self.lambda_force_up,
            )
            reward -= fup_vec

        # -----------------------------
        # 2B) Optional: F·disp directional penalty (1B)
        # -----------------------------
        fd_pen_scalar = 0.0
        if self.use_fd_direction_penalty and self.lambda_fd_penalty > 0.0:
            fd_vec, fd_pen_scalar = compute_fd_direction_penalty(
                self.forces,
                disp,
                self.lambda_fd_penalty,
            )
            reward -= fd_vec

        # -----------------------------
        # 3) COM penalty (stabilized)
        # -----------------------------
        COM_new = self.atoms.positions.mean(axis=0)
        delta_COM = np.linalg.norm(COM_new - self.COM_prev)

        com_pen = self.com_lambda * delta_COM
        reward -= com_pen
        self.COM_prev = COM_new.copy()

        # COM 폭주 조기 종료
        if delta_COM > self.com_threshold:
            time_pen = 0.0
            if self.time_penalty > 0.0:
                time_pen = self.time_penalty
                reward -= self.time_penalty

            self._set_last_reward_components(
                r_f_mean,
                com_penalty=com_pen,
                bond_penalty=0.0,
                time_penalty=time_pen,
                fail_penalty=0.0,
                terminal_bonus=0.0,
                reward_vec=reward,
                force_up_penalty=force_up_pen_scalar,
                fd_penalty=fd_pen_scalar,
            )
            return self._obs(), reward, True

        # -----------------------------
        # 4) Bond penalty (×1, capped at 3)
        # -----------------------------
        bond_pen_sum = 0.0

        for idx, (a, b) in enumerate(self.bond_pairs):
            rel = self._rel_vec(a, b)
            d = np.linalg.norm(rel)
            d0 = self.bond_d0[idx]
            ratio = d / d0

            stretch = max(0.0, ratio - self.bond_break_ratio)
            compress = max(0.0, 0.6 - ratio)

            penalty = 1.0 * self.k_bond * np.sqrt(stretch**2 + compress**2)
            penalty = min(penalty, 3.0)

            bond_pen_sum += penalty
            reward -= penalty

            # bond 완전 붕괴/비정상 시 조기 종료
            if ratio > 6.0 or ratio < 0.25:
                time_pen = 0.0
                if self.time_penalty > 0.0:
                    time_pen = self.time_penalty
                    reward -= self.time_penalty

                self._set_last_reward_components(
                    r_f_mean,
                    com_penalty=com_pen,
                    bond_penalty=bond_pen_sum,
                    time_penalty=time_pen,
                    fail_penalty=0.0,
                    terminal_bonus=0.0,
                    reward_vec=reward,
                    force_up_penalty=force_up_pen_scalar,
                    fd_penalty=fd_pen_scalar,
                )
                return self._obs(), reward, True

        # -----------------------------
        # 5) Termination conditions
        # -----------------------------
        done = False
        success = False
        fail_pen = 0.0
        bonus = 0.0

        if np.mean(new_norm) < self.fmax_threshold:
            done = True
            success = True

        if self.step_count >= self.max_steps:
            done = True
            if not success and self.fail_penalty > 0.0:
                fail_pen = self.fail_penalty
                reward -= self.fail_penalty

        # -----------------------------
        # 6) Time penalty (living cost)
        # -----------------------------
        time_pen = 0.0
        if self.time_penalty > 0.0:
            time_pen = self.time_penalty
            reward -= self.time_penalty

        # -----------------------------
        # 7) Terminal bonus (성공 시)
        # -----------------------------
        if success and self.terminal_bonus_base > 0.0:
            frac = (self.max_steps - self.step_count) / float(self.max_steps)
            frac = max(frac, 0.0)
            bonus = self.terminal_bonus_base * frac
            reward += bonus

        # -----------------------------
        # Reward composition 기록
        # -----------------------------
        self._set_last_reward_components(
            r_f_mean,
            com_penalty=com_pen,
            bond_penalty=bond_pen_sum,
            time_penalty=time_pen,
            fail_penalty=fail_pen,
            terminal_bonus=bonus,
            reward_vec=reward,
            force_up_penalty=force_up_pen_scalar,
            fd_penalty=fd_pen_scalar,
        )

        # -----------------------------
        # Update memory
        # -----------------------------
        self.prev_disp = disp.copy()
        self.prev_forces = self.forces.copy()
        self.forces = new_forces.copy()

        return self._obs(), reward, done
