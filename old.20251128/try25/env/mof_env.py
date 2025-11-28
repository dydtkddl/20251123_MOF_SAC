import os
import logging
from typing import Tuple, Dict, Any, Optional

import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
from ase.geometry import find_mic


logger = logging.getLogger(__name__)

# -----------------------------
# 기본 화학/원자 특성 테이블
# -----------------------------

# 간단한 파울링 전기음성도 (필요 원소만)
ELECTRONEGATIVITY = {
    "H": 2.20,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "Cl": 3.16,
    "Zn": 1.65,
    "Cu": 1.90,
    "Zr": 1.33,
    "Ti": 1.54,
    "Al": 1.61,
}

# MOF에서 자주 등장하는 금속 원소들
METAL_ELEMENTS = {
    "Li", "Na", "K", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Zr", "Hf", "Al", "Ga", "Y"
}

# neighbor type one-hot 용 공통 원소 리스트 (길이 10)
COMMON_NEIGHBOR_ELEMENTS = ["H", "C", "N", "O", "F", "Cl", "Zn", "Cu", "Zr", "Ti"]
NEIGHBOR_ELEMENT_TO_IDX = {sym: i for i, sym in enumerate(COMMON_NEIGHBOR_ELEMENTS)}


# ============================================================
# AtomsLoader: MOF 구조 로더
# ============================================================

class AtomsLoader:
    """
    mofs/train_pool 아래의 .cif 파일들을 모두 모아서
    매 reset 때마다 하나씩 샘플링해서 Atoms를 반환한다.
    """

    def __init__(self, root_dir: str, file_ext: str = ".cif"):
        self.files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.lower().endswith(file_ext):
                    self.files.append(os.path.join(dirpath, fname))

        if not self.files:
            raise RuntimeError(f"No structure files with ext {file_ext} found under {root_dir}")

        self.n_files = len(self.files)
        logger.info(f"[AtomsLoader] Found {self.n_files} structures under {root_dir}")

    def sample(self) -> Tuple[Atoms, str]:
        idx = np.random.randint(0, self.n_files)
        path = self.files[idx]
        atoms = read(path)
        return atoms, path


# ============================================================
# MOFEnv: Per-atom Multi-agent 환경
# ============================================================

class MOFEnv:
    """
    - 각 원자가 하나의 에이전트
    - observation: (N_atoms, 360)
      - center features: 12
      - neighbor features: 12 neighbors × 29 = 348
      - 총 360 → 기존 OBS_DIM 유지
    - action: (N_atoms, 3)  ∈ [-1, 1]^3
      disp_i = alpha * action_i * (min(||F_i||, cmax) / cmax)
    """

    def __init__(
        self,
        atoms_loader: AtomsLoader,
        calculator,
        k_neighbors: int = 12,
        neighbor_cutoff: float = 6.0,
        cmax: float = 0.4,
        max_steps: int = 300,
        fmax_threshold: float = 0.12,
        bond_break_ratio: float = 2.4,
        k_bond: float = 3.0,
        max_penalty: float = 10.0,
        alpha: float = 0.04,
        w_f: float = 1.0,
        w_bond: float = 1.0,
        w_com: float = 0.1,
        debug_bond: bool = False,
    ):
        self.atoms_loader = atoms_loader
        self.calculator = calculator

        # neighbor / action 설정
        self.k = k_neighbors
        self.neighbor_cutoff = neighbor_cutoff
        self.cmax = cmax
        self.alpha = alpha

        # rollout 설정
        self.max_steps = max_steps
        self.fmax_threshold = fmax_threshold

        # bond penalty 설정
        self.bond_break_ratio = bond_break_ratio
        self.k_bond = k_bond
        self.max_penalty = max_penalty
        self.debug_bond = debug_bond

        # reward weight
        self.w_f = w_f
        self.w_bond = w_bond
        self.w_com = w_com

        # 상태 변수
        self.atoms: Optional[Atoms] = None
        self.structure_id: Optional[str] = None
        self.n_atoms: int = 0
        self.step_count: int = 0

        self.forces: Optional[np.ndarray] = None
        self.energy: float = 0.0
        self.prev_fmax: Optional[float] = None
        self.prev_energy: Optional[float] = None

        self.com0: Optional[np.ndarray] = None  # 초기 COM

        # bond reference
        self.bond_pairs: Optional[np.ndarray] = None  # (M, 2)
        self.bond_d0: Optional[np.ndarray] = None     # (M,)

        # atom 타입 마스크
        self.metal_mask: Optional[np.ndarray] = None
        self.linker_mask: Optional[np.ndarray] = None
        self.aromatic_mask: Optional[np.ndarray] = None

        # obs dimension (center 12 + neighbor 29 × k)
        self.center_dim = 12
        self.per_neighbor_dim = 29
        self.obs_dim = self.center_dim + self.per_neighbor_dim * self.k

        logger.info(
            f"[MOFEnv] Initialized: k={self.k}, obs_dim={self.obs_dim}, "
            f"fmax_threshold={self.fmax_threshold}, max_steps={self.max_steps}"
        )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        새로운 MOF 구조 샘플링 후 에너지/힘 계산, bond reference 설정,
        atom 타입/방향성 feature를 초기화하고 per-atom obs (N, 360)를 반환.
        """
        self.atoms, self.structure_id = self.atoms_loader.sample()
        self.atoms.calc = self.calculator

        # PBC가 없으면 기본적으로 PBC 켬 (MOF이므로)
        if self.atoms.get_pbc() is None:
            self.atoms.set_pbc([True, True, True])

        self.n_atoms = len(self.atoms)
        self.step_count = 0

        # 초기 에너지/힘 계산
        self.energy = float(self.atoms.get_potential_energy())
        self.forces = np.array(self.atoms.get_forces(), dtype=np.float64)
        fmag = np.linalg.norm(self.forces, axis=1)
        self.prev_fmax = float(fmag.max())
        self.prev_energy = self.energy

        # 초기 COM
        self.com0 = self.atoms.get_center_of_mass()

        # bond reference 세팅
        self._build_bond_reference()

        # atom 타입 (metal / linker / aromatic 등)
        self._build_atom_type_masks()

        obs = self._build_observation()

        logger.info(
            f"[MOFEnv.reset] structure={self.structure_id}, "
            f"N_atoms={self.n_atoms}, Fmax0={self.prev_fmax:.3e}, "
            f"E0={self.energy:.6f}"
        )

        return obs

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        actions: (N_atoms, 3) in [-1, 1]
        """
        if self.atoms is None:
            raise RuntimeError("Call reset() before step().")

        if actions.ndim == 1:
            actions = actions.reshape(-1, 3)

        if actions.shape[0] != self.n_atoms or actions.shape[1] != 3:
            raise ValueError(
                f"Actions must be (N_atoms, 3), got {actions.shape}, N_atoms={self.n_atoms}"
            )

        # 기존 힘 기준으로 변위 계산
        fmag = np.linalg.norm(self.forces, axis=1, keepdims=True)  # (N,1)
        scale = np.minimum(fmag, self.cmax) / (self.cmax + 1e-8)   # (N,1)
        disp = self.alpha * actions * scale                         # (N,3)

        disp_mag = np.linalg.norm(disp, axis=1)
        disp_mean = float(disp_mag.mean())
        disp_max = float(disp_mag.max())

        # 좌표 업데이트 + PBC wrap
        pos = self.atoms.get_positions()
        new_pos = pos + disp
        self.atoms.set_positions(new_pos)
        self.atoms.wrap(eps=1e-12)

        # 새 에너지/힘 계산
        self.energy = float(self.atoms.get_potential_energy())
        self.forces = np.array(self.atoms.get_forces(), dtype=np.float64)

        # reward & done 계산
        reward, done, extra = self._compute_reward_and_done(disp_mean, disp_max)

        # 다음 관측
        obs_next = self._build_observation()

        info = {
            "structure_id": self.structure_id,
            "n_atoms": self.n_atoms,
            "Fmax": extra["Fmax"],
            "Fmean": extra["Fmean"],
            "energy": self.energy,
            "energy_per_atom": self.energy / self.n_atoms,
            "bond_penalty": extra["bond_penalty"],
            "n_bonds_stretched": extra["n_bonds_stretched"],
            "com_shift": extra["com_shift"],
            "disp_mean": disp_mean,
            "disp_max": disp_max,
            "step": self.step_count,
        }

        return obs_next, reward, done, info

    # --------------------------------------------------------
    # 내부 유틸
    # --------------------------------------------------------

    def _build_bond_reference(self):
        """
        초기 구조에서 bond reference (pair & d0) 설정.
        bond 판단 기준: d < 1.2 * (r_cov_i + r_cov_j)
        """
        atoms = self.atoms
        Z = atoms.get_atomic_numbers()
        rcov = np.array([covalent_radii[z] for z in Z], dtype=np.float64)

        i_idx, j_idx, S = neighbor_list("ijS", atoms, self.neighbor_cutoff)
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        pos = atoms.get_positions()

        dR = pos[j_idx] + np.dot(S, cell) - pos[i_idx]
        dR_mic, dist = find_mic(dR, cell, pbc)

        r_sum = rcov[i_idx] + rcov[j_idx]
        ref_ratio = 1.2  # reference bond 길이 기준
        mask = dist < ref_ratio * r_sum

        i_b = i_idx[mask]
        j_b = j_idx[mask]
        d0 = dist[mask]

        # i < j 로 정규화해서 중복 제거
        keep = i_b < j_b
        i_b = i_b[keep]
        j_b = j_b[keep]
        d0 = d0[keep]

        self.bond_pairs = np.stack([i_b, j_b], axis=1)
        self.bond_d0 = d0

        logger.info(
            f"[MOFEnv] Bond reference built: n_bonds={len(self.bond_pairs)} "
            f"(structure={self.structure_id})"
        )

    def _build_atom_type_masks(self):
        """
        metal / linker / aromatic 간단 분류.
        aromatic: (C or N)이고, C/N 이웃이 2개 이상인 경우로 heuristic.
        """
        symbols = self.atoms.get_chemical_symbols()
        N = len(symbols)

        metal_mask = np.array([s in METAL_ELEMENTS for s in symbols], dtype=bool)
        linker_mask = ~metal_mask

        # adjacency from bond_pairs
        adj = [[] for _ in range(N)]
        if self.bond_pairs is not None:
            for a, b in self.bond_pairs:
                adj[a].append(b)
                adj[b].append(a)

        aromatic_mask = np.zeros(N, dtype=bool)
        for i, s in enumerate(symbols):
            if s not in ("C", "N"):
                continue
            neigh = adj[i]
            if len(neigh) < 2:
                continue
            n_cn = sum(symbols[j] in ("C", "N") for j in neigh)
            if n_cn >= 2:
                aromatic_mask[i] = True

        self.metal_mask = metal_mask
        self.linker_mask = linker_mask
        self.aromatic_mask = aromatic_mask

        n_metal = int(metal_mask.sum())
        n_arom = int(aromatic_mask.sum())
        logger.info(
            f"[MOFEnv] Atom type masks: metal={n_metal}, aromatic={n_arom}, "
            f"linker={N - n_metal}"
        )

    def _compute_reward_and_done(self, disp_mean: float, disp_max: float):
        """
        - force-based reward: log(F_old) - log(F_new)
        - bond penalty: bond_break_ratio 기반 soft penalty
        - COM penalty: 초기 COM에서의 drift
        """
        assert self.forces is not None
        fmag = np.linalg.norm(self.forces, axis=1)
        Fmax_new = float(fmag.max())
        Fmean_new = float(fmag.mean())
        eps = 1e-8

        if self.prev_fmax is None:
            reward_force = 0.0
        else:
            reward_force = np.log(self.prev_fmax + eps) - np.log(Fmax_new + eps)

        # bond penalty
        bond_penalty, n_stretched = self._bond_penalty()

        # COM penalty
        com_curr = self.atoms.get_center_of_mass()
        com_shift_vec = com_curr - self.com0
        com_shift = float(np.linalg.norm(com_shift_vec))
        com_penalty = com_shift

        reward = (
            self.w_f * reward_force
            - self.w_bond * bond_penalty
            - self.w_com * com_penalty
        )

        # step 증가
        self.step_count += 1

        done = bool(
            Fmax_new < self.fmax_threshold or self.step_count >= self.max_steps
        )

        # state 업데이트
        self.prev_fmax = Fmax_new
        self.prev_energy = self.energy

        extra = {
            "Fmax": Fmax_new,
            "Fmean": Fmean_new,
            "bond_penalty": bond_penalty,
            "n_bonds_stretched": n_stretched,
            "com_shift": com_shift,
        }

        if self.debug_bond:
            logger.debug(
                f"[MOFEnv.step] step={self.step_count} "
                f"reward_force={reward_force:.4f}, bond_penalty={bond_penalty:.4f}, "
                f"com_penalty={com_penalty:.4f}, reward={reward:.4f}"
            )

        return float(reward), done, extra

    def _bond_penalty(self) -> Tuple[float, int]:
        """
        bond 길이가 bond_break_ratio * d0 를 넘는 경우 soft penalty 부여.
        """
        if self.bond_pairs is None or len(self.bond_pairs) == 0:
            return 0.0, 0

        atoms = self.atoms
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        i = self.bond_pairs[:, 0]
        j = self.bond_pairs[:, 1]

        dR = pos[j] - pos[i]
        dR_mic, dist = find_mic(dR, cell, pbc)

        ratio = dist / (self.bond_d0 + 1e-8)
        excess = np.clip(ratio - self.bond_break_ratio, 0.0, None)

        # quadratic penalty, soft-capped
        penalty_per_bond = self.k_bond * excess ** 2
        penalty_per_bond = np.minimum(penalty_per_bond, self.max_penalty)

        n_stretched = int((excess > 0.0).sum())
        if len(penalty_per_bond) == 0:
            bond_penalty = 0.0
        else:
            bond_penalty = float(penalty_per_bond.mean())

        return bond_penalty, n_stretched

    # --------------------------------------------------------
    # Observation builder (N_atoms, 360)
    # --------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        """
        center 12 + neighbor(12) × 29 = 360
        center features (12):
          0: Z / 100
          1: period / 7
          2: group / 18 (대략)
          3: covalent radius
          4: electronegativity (0 if unknown)
          5: is_metal (0/1)
          6: is_O (0/1)
          7: is_N (0/1)
          8: is_C (0/1)
          9: is_aromatic (0/1)
          10: is_metal_cluster_atom (metal_mask)
          11: is_linker_atom (~metal_mask)

        per-neighbor features (29):
          0-2: dx, dy, dz (MIC)
          3:   dist
          4:   dist / neighbor_cutoff
          5:   r_sum (r_cov_i + r_cov_j)
          6:   dist / r_sum
          7:   bond_flag (0/1, ref bond 존재 여부)
          8:   is_metal_metal
          9:   is_metal_O
          10:  is_metal_N
          11:  is_carboxyl_O (O-C/O-M에 해당하는 O 근사)
          12:  is_aromatic_pair (both aromatic)
          13-22: neighbor type one-hot (COMMON_NEIGHBOR_ELEMENTS, dim=10)
          23:  |F_i| / (cmax + eps)
          24:  |F_j| / (cmax + eps)
          25:  alignment(F_i, dR_unit)
          26:  neighbor index / k
          27:  local bond stretch flag (> bond_break_ratio)
          28:  zero padding (reserved)
        """
        atoms = self.atoms
        Z = atoms.get_atomic_numbers()
        symbols = atoms.get_chemical_symbols()
        rcov = np.array([covalent_radii[z] for z in Z], dtype=np.float64)

        fvec = self.forces
        fmag = np.linalg.norm(fvec, axis=1)  # (N,)

        N = self.n_atoms
        obs = np.zeros((N, self.obs_dim), dtype=np.float32)

        # neighbor 정보 (k-NN, PBC MIC)
        nbr_indices, nbr_dR, nbr_dist = self._build_neighbor_info()

        # bond reference를 빠르게 lookup 하기 위한 set
        bond_set = set()
        if self.bond_pairs is not None:
            for a, b in self.bond_pairs:
                if a < b:
                    bond_set.add((int(a), int(b)))
                else:
                    bond_set.add((int(b), int(a)))

        metal_mask = self.metal_mask
        aromatic_mask = self.aromatic_mask

        # center & neighbor loop
        for i in range(N):
            z = Z[i]
            sym = symbols[i]

            # ------------- center 12 -------------
            center_feat = np.zeros(self.center_dim, dtype=np.float32)

            # 원자번호 / 주기 / 족 (대략적인 group)
            center_feat[0] = z / 100.0
            period = self._period_from_Z(z)
            group = self._group_from_Z(z)
            center_feat[1] = period / 7.0
            center_feat[2] = group / 18.0

            center_feat[3] = float(rcov[i])

            chi = ELECTRONEGATIVITY.get(sym, 0.0)
            center_feat[4] = float(chi)

            center_feat[5] = float(metal_mask[i])
            center_feat[6] = float(sym == "O")
            center_feat[7] = float(sym == "N")
            center_feat[8] = float(sym == "C")
            center_feat[9] = float(aromatic_mask[i])
            center_feat[10] = float(metal_mask[i])  # metal cluster
            center_feat[11] = float(not metal_mask[i])

            # ------------- neighbor 29 × k -------------
            neighbor_feat = np.zeros((self.k, self.per_neighbor_dim), dtype=np.float32)

            Fi = fvec[i]
            Fi_mag = fmag[i]
            Fi_norm = Fi_mag / (self.cmax + 1e-6)

            for n in range(self.k):
                j = int(nbr_indices[i, n])
                if j < 0:
                    # padding
                    continue

                sym_j = symbols[j]
                dr = nbr_dR[i, n]
                d = nbr_dist[i, n]
                r_sum = rcov[i] + rcov[j]
                r_sum = r_sum if r_sum > 1e-6 else 1e-6

                # feature vector (29)
                feat = np.zeros(self.per_neighbor_dim, dtype=np.float32)

                # geom
                feat[0:3] = dr.astype(np.float32)
                feat[3] = float(d)
                feat[4] = float(d / (self.neighbor_cutoff + 1e-6))
                feat[5] = float(r_sum)
                feat[6] = float(d / r_sum)

                # bond flag
                key = (i, j) if i < j else (j, i)
                bond_flag = 1.0 if key in bond_set else 0.0
                feat[7] = bond_flag

                # motif-like 플래그
                is_metal_i = metal_mask[i]
                is_metal_j = metal_mask[j]
                feat[8] = float(is_metal_i and is_metal_j)  # metal-metal
                feat[9] = float((is_metal_i or is_metal_j) and ("O" in {sym, sym_j}))
                feat[10] = float((is_metal_i or is_metal_j) and ("N" in {sym, sym_j}))

                # carboxylate O 근사 (O에 metal 또는 C가 붙은 경우)
                feat[11] = float(
                    (sym == "O" or sym_j == "O")
                    and (("C" in {sym, sym_j}) or (is_metal_i or is_metal_j))
                )

                # aromatic pair
                feat[12] = float(aromatic_mask[i] and aromatic_mask[j])

                # neighbor type one-hot (13-22)
                base_idx = 13
                idx = NEIGHBOR_ELEMENT_TO_IDX.get(sym_j, None)
                if idx is not None:
                    feat[base_idx + idx] = 1.0

                # force / alignment
                Fj_mag = fmag[j]
                Fj_norm = Fj_mag / (self.cmax + 1e-6)
                feat[23] = float(Fi_norm)
                feat[24] = float(Fj_norm)

                if d > 1e-6 and Fi_mag > 1e-6:
                    dR_unit = dr / d
                    align = float(np.dot(Fi, dR_unit) / (Fi_mag + 1e-6))
                else:
                    align = 0.0
                feat[25] = align

                # neighbor index / k
                feat[26] = float(n) / float(self.k)

                # local bond stretch flag (> bond_break_ratio)
                if key in bond_set:
                    # 대략 현재 bond 길이 vs d0 비교
                    # (d0는 pair index에서 찾아야 하지만 여기선 근사로 ratio만 사용)
                    ratio = d / r_sum
                    feat[27] = float(ratio > self.bond_break_ratio)
                else:
                    feat[27] = 0.0

                # 28: reserved zero
                feat[28] = 0.0

                neighbor_feat[n, :] = feat

            # concat
            obs[i, :] = np.concatenate(
                [center_feat.astype(np.float32), neighbor_feat.reshape(-1)], axis=0
            )

        return obs

    def _build_neighbor_info(self):
        """
        neighbor_list + MIC로 각 원자별 최대 self.k 개 neighbor 정보를 생성.
        반환:
          nbr_indices: (N, k)  [-1 padding]
          nbr_dR:      (N, k, 3)
          nbr_dist:    (N, k)
        """
        atoms = self.atoms
        N = len(atoms)
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        i_idx, j_idx, S = neighbor_list("ijS", atoms, self.neighbor_cutoff)
        dR = pos[j_idx] + np.dot(S, cell) - pos[i_idx]
        dR_mic, dist = find_mic(dR, cell, pbc)

        # 거리 순으로 sort 후 상위 k개만 채우기
        order = np.argsort(dist)
        i_idx = i_idx[order]
        j_idx = j_idx[order]
        dR_mic = dR_mic[order]
        dist = dist[order]

        nbr_indices = -np.ones((N, self.k), dtype=np.int64)
        nbr_dR = np.zeros((N, self.k, 3), dtype=np.float32)
        nbr_dist = np.zeros((N, self.k), dtype=np.float32)

        counts = np.zeros(N, dtype=np.int32)
        for i, j, dr, d in zip(i_idx, j_idx, dR_mic, dist):
            c = counts[i]
            if c >= self.k:
                continue
            nbr_indices[i, c] = j
            nbr_dR[i, c, :] = dr.astype(np.float32)
            nbr_dist[i, c] = float(d)
            counts[i] += 1

        return nbr_indices, nbr_dR, nbr_dist

    # --------------------------------------------------------
    # 간단한 Z → period / group 헬퍼
    # (정확 group 필요 없고 대략적인 scaling 용)
    # --------------------------------------------------------

    @staticmethod
    def _period_from_Z(Z_val: int) -> int:
        if Z_val <= 2:
            return 1
        elif Z_val <= 10:
            return 2
        elif Z_val <= 18:
            return 3
        elif Z_val <= 36:
            return 4
        elif Z_val <= 54:
            return 5
        elif Z_val <= 86:
            return 6
        else:
            return 7

    @staticmethod
    def _group_from_Z(Z_val: int) -> int:
        """
        매우 대략적인 group mapping. RL feature scaling용이라
        대충만 맞으면 됨.
        """
        sym = chemical_symbols[Z_val]
        # 간단 heuristic
        if sym in {"H", "Li", "Na", "K"}:
            return 1
        if sym in {"Be", "Mg", "Ca", "Sr", "Ba"}:
            return 2
        if sym in {"B", "Al", "Ga"}:
            return 13
        if sym in {"C", "Si"}:
            return 14
        if sym in {"N", "P"}:
            return 15
        if sym in {"O", "S"}:
            return 16
        if sym in {"F", "Cl"}:
            return 17
        if sym in {"He", "Ne", "Ar", "Kr", "Xe"}:
            return 18
        # 전이금속 등은 대충 10 부근으로
        return 10
