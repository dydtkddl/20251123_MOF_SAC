# env/modes.py

import os
import logging
from typing import Optional, Tuple, Dict

import numpy as np
from ase.neighborlist import neighbor_list

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ------------------------------------------------------------------
# (옵션) precomputed mode 파일 관련 유틸
# ------------------------------------------------------------------


def modes_filename_from_cif(cif_path: str, suffix: str = "_modes.npy") -> str:
    """
    CIF 파일 이름으로부터 모드 파일 이름 생성.

    e.g.)
        cif_path = ".../QMOF_1234.cif"
        -> "QMOF_1234_modes.npy"
    """
    base = os.path.basename(cif_path)
    stem, _ = os.path.splitext(base)
    return f"{stem}{suffix}"


def load_modes_for_cif(
    cif_path: str,
    modes_dir: str,
    num_modes: Optional[int] = None,
    suffix: str = "_modes.npy",
) -> Optional[np.ndarray]:
    """
    특정 CIF에 대응하는 모드 데이터(.npy)를 로드.

    Parameters
    ----------
    cif_path : str
        원본 CIF 경로
    modes_dir : str
        모드 파일들이 저장된 디렉토리
    num_modes : int, optional
        앞에서부터 사용할 모드 개수 (None이면 전체 사용)
    suffix : str
        파일 이름 접미사. 기본 "_modes.npy"

    Returns
    -------
    modes : np.ndarray or None
        shape (M, N, 3) 또는 (M, N) 등. 없으면 None.
    """
    fname = modes_filename_from_cif(cif_path, suffix=suffix)
    path = os.path.join(modes_dir, fname)

    if not os.path.exists(path):
        logger.warning(f"[modes] Mode file not found for CIF: {path}")
        return None

    modes = np.load(path, allow_pickle=False)
    if num_modes is not None and num_modes < modes.shape[0]:
        modes = modes[:num_modes]

    logger.info(
        f"[modes] Loaded modes for CIF={os.path.basename(cif_path)} "
        f"from {path} (num_modes={modes.shape[0]})"
    )
    return modes


def apply_modes(
    coeffs: np.ndarray,
    modes: np.ndarray,
) -> np.ndarray:
    """
    mode 계수와 모드를 이용해 per-atom displacement 생성.

    Parameters
    ----------
    coeffs : np.ndarray
        shape (M,) 또는 (B, M)
    modes : np.ndarray
        shape (M, N, 3)  (M: 모드 개수, N: 원자 수)

    Returns
    -------
    disp : np.ndarray
        - coeffs shape가 (M,)이면 (N, 3)
        - coeffs shape가 (B, M)이면 (B, N, 3)
    """
    if modes.ndim != 3:
        raise ValueError(
            f"[apply_modes] modes must have shape (M, N, 3), got {modes.shape}"
        )

    M, N, D = modes.shape
    if coeffs.ndim == 1:
        if coeffs.shape[0] != M:
            raise ValueError(
                f"[apply_modes] coeffs length {coeffs.shape[0]} != num_modes {M}"
            )
        # (M,) · (M, N, 3) → (N, 3)
        disp = np.tensordot(coeffs, modes, axes=(0, 0))  # (N, 3)
        return disp

    elif coeffs.ndim == 2:
        B, M2 = coeffs.shape
        if M2 != M:
            raise ValueError(
                f"[apply_modes] coeffs last dim {M2} != num_modes {M}"
            )
        # (B, M) x (M, N, 3) → (B, N, 3)
        disp = np.tensordot(coeffs, modes, axes=(1, 0))  # (B, N, 3)
        return disp

    else:
        raise ValueError(
            f"[apply_modes] coeffs must be 1D or 2D, got {coeffs.shape}"
        )


def summarize_modes(modes: np.ndarray) -> Tuple[int, int]:
    """
    모드 배열의 간단한 요약 정보를 반환.

    Returns
    -------
    num_modes : int
    num_atoms : int
    """
    if modes.ndim != 3:
        raise ValueError(
            f"[summarize_modes] modes must have shape (M, N, 3), got {modes.shape}"
        )
    M, N, _ = modes.shape
    return M, N


# ------------------------------------------------------------------
# NEW: Graph Laplacian 기반 mode-basis (4A/4B smoothing용)
# ------------------------------------------------------------------

# mode_id 기준 간단 캐시 (같은 CIF 여러 번 쓸 때 재계산 방지)
_MODE_CACHE: Dict[str, np.ndarray] = {}


def _build_graph_laplacian(
    atoms,
    cutoff: float = 4.0,
) -> np.ndarray:
    """
    ASE Atoms 객체로부터 단순 unweighted graph Laplacian 생성.

    - neighbor_list("ij")를 사용해 인접 행렬 A 구성
    - L = D - A (D: degree matrix)
    """
    N = len(atoms)
    if N == 0:
        raise ValueError("[modes] Atoms has zero length.")

    # neighbor_list 로 i, j 인덱스 추출 (주기 경계 반영)
    i, j = neighbor_list("ij", atoms, cutoff=cutoff)

    A = np.zeros((N, N), dtype=float)
    # 단순 unweighted undirected graph
    for a, b in zip(i, j):
        if a == b:
            continue
        A[a, b] = 1.0
        A[b, a] = 1.0

    deg = A.sum(axis=1)
    D = np.diag(deg)
    L = D - A
    return L


def get_graph_mode_basis(
    atoms,
    num_modes: int = 16,
    cutoff: float = 4.0,
    mode_id: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    그래프 Laplacian 고유벡터 기반 mode basis 생성.

    Parameters
    ----------
    atoms : ase.Atoms
        현재 MOF 구조 (perturb 적용 후 reset 시점).
    num_modes : int
        사용할 모드 개수 (constant mode는 제외).
    cutoff : float
        neighbor_list cutoff (Å).
    mode_id : str, optional
        동일 CIF 구조 재사용 시 캐시 키 (예: CIF basename).

    Returns
    -------
    U : np.ndarray or None
        shape (N, K)  (N: 원자 수, K: 모드 개수),
        constant mode(λ≈0) 제외 후 작은 eigenvalue 순으로 K개 선택.
        실패 시 None 반환.
    """
    N = len(atoms)
    if N == 0:
        logger.warning("[MODE] Atoms length=0, skip mode basis.")
        return None

    # 캐시 확인
    if mode_id is not None and mode_id in _MODE_CACHE:
        U_cached = _MODE_CACHE[mode_id]
        if U_cached.shape[0] == N:
            logger.info(
                "[MODE] Using cached graph mode basis: id=%s, N=%d, num_modes=%d",
                mode_id,
                N,
                U_cached.shape[1],
            )
            return U_cached
        else:
            logger.warning(
                "[MODE] Cached basis for id=%s has mismatched N: %d != %d. "
                "Recomputing.",
                mode_id,
                U_cached.shape[0],
                N,
            )

    # Laplacian 구성
    try:
        L = _build_graph_laplacian(atoms, cutoff=cutoff)
    except Exception as e:
        logger.warning(
            "[MODE] Failed to build Laplacian for id=%s: %s",
            str(mode_id),
            repr(e),
        )
        return None

    # 고유값/고유벡터 계산
    try:
        # sym. positive semi-definite → eigh 사용
        evals, evecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError as e:
        logger.warning(
            "[MODE] eigh failed for id=%s: %s",
            str(mode_id),
            repr(e),
        )
        return None

    # 작은 eigenvalue 순으로 정렬
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]  # (N, N)

    # 첫 번째 eigenvector는 거의 항상 constant mode(λ≈0) → 제외
    if N <= 1:
        logger.warning(
            "[MODE] N <= 1 (N=%d), skip mode basis.", N
        )
        return None

    # 사용할 모드 개수 결정 (constant mode 제외한 나머지 중)
    max_modes = max(0, N - 1)  # constant 하나 제외 가능 최대
    K = min(num_modes, max_modes)
    if K <= 0:
        logger.warning(
            "[MODE] num_modes=%d, N=%d → usable modes=0. Skip.",
            num_modes,
            N,
        )
        return None

    # constant mode(0번째) 제외 후 앞에서부터 K개
    U = evecs[:, 1:1 + K].astype(np.float32)  # (N, K)

    logger.info(
        "[MODE] Built graph mode basis: id=%s, N=%d, K=%d, "
        "min_eig=%.6e, max_eig=%.6e",
        str(mode_id),
        N,
        K,
        float(evals[0]),
        float(evals[-1]),
    )

    # 캐시에 저장
    if mode_id is not None:
        _MODE_CACHE[mode_id] = U

    return U


def apply_graph_mode_smoothing(
    disp: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    """
    Graph mode basis를 이용한 low-pass smoothing.

    Parameters
    ----------
    disp : np.ndarray
        (N, 3) per-atom displacement (gate, scale 적용 후).
    U : np.ndarray
        (N, K) mode basis (get_graph_mode_basis의 출력).

    Returns
    -------
    disp_smooth : np.ndarray
        (N, 3) smoothed displacement.
        shape mismatch나 K=0이면 원본 disp를 그대로 반환.
    """
    if disp.ndim != 2 or disp.shape[1] != 3:
        raise ValueError(
            f"[MODE] disp must have shape (N, 3), got {disp.shape}"
        )

    if U is None:
        return disp

    if U.ndim != 2:
        logger.warning(
            "[MODE] U must have shape (N, K), got %s. Skip smoothing.",
            str(U.shape),
        )
        return disp

    N, D = disp.shape
    if U.shape[0] != N:
        logger.warning(
            "[MODE] U.shape[0] (%d) != disp.shape[0] (%d). Skip smoothing.",
            U.shape[0],
            N,
        )
        return disp

    K = U.shape[1]
    if K == 0:
        logger.warning("[MODE] U has zero modes (K=0). Skip smoothing.")
        return disp

    # low-pass: disp_smooth = U @ (U^T @ disp)
    #   - U^T @ disp : (K, 3)
    #   - 결과 : (N, 3)
    coeff = U.T @ disp        # (K, 3)
    disp_smooth = U @ coeff   # (N, 3)

    return disp_smooth.astype(np.float32)
