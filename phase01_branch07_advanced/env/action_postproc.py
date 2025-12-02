# env/action_postproc.py
# -*- coding: utf-8 -*-

"""
Action post-processing utilities for MACS-MOF SAC
=================================================

옵션별 기능
----------

1A) Force norm increase penalty
    - 이전 step 대비 force norm 이 증가한 atom 에 페널티 부여
    - r_f(log force 감소 보상)과 별도로, 강하게 상승하는 경우 추가 제어

1B) F·disp 방향 penalty
    - disp 방향이 force와 반대(에너지 증가 방향)일 때 페널티
    - ASE convention: F = -∇E
      → δx 를 F 방향(gradient descent)으로 움직이는 것이 에너지 감소에 유리
      → F · disp < 0 인 경우를 벌점

3A) Bond-subspace projection
    - per-atom displacement 를 "bond 기반 서브스페이스"로 투영
    - 여기서는 reset 시점의 bond 방향들의 합으로 정의된 1D 축에 투영

3B) Local frame (radial / tangent) scaling
    - bond 축 기준으로 disp 를 radial / tangential 로 분해
    - radial_scale, tangent_scale 로 스케일 조절
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ============================================================
# 1A) Force increase penalty
# ============================================================

def compute_force_increase_penalty(
    old_forces: np.ndarray,
    new_forces: np.ndarray,
    lambda_force_up: float,
) -> Tuple[np.ndarray, float]:
    """
    Force norm 증가에 대한 per-atom penalty.

    Parameters
    ----------
    old_forces : (N, 3)
    new_forces : (N, 3)
    lambda_force_up : float
        증가량에 곱해줄 weight.

    Returns
    -------
    penalty_vec : (N,)
        각 atom별 penalty.
    penalty_mean : float
        penalty_vec 의 평균 (로그용 스칼라).
    """
    if old_forces.shape != new_forces.shape:
        raise ValueError(
            f"[force_up] shape mismatch: old={old_forces.shape}, "
            f"new={new_forces.shape}"
        )

    old_norm = np.linalg.norm(old_forces, axis=1)
    new_norm = np.linalg.norm(new_forces, axis=1)

    delta = new_norm - old_norm
    delta_pos = np.maximum(delta, 0.0)  # 증가분만 사용

    penalty_vec = lambda_force_up * delta_pos
    penalty_mean = float(penalty_vec.mean())

    logger.debug(
        "[force_up] mean_old=%.6f mean_new=%.6f mean_delta+=%.6f lambda=%.3f "
        "penalty_mean=%.6f",
        float(old_norm.mean()),
        float(new_norm.mean()),
        float(delta_pos.mean()),
        lambda_force_up,
        penalty_mean,
    )

    return penalty_vec.astype(np.float32), penalty_mean


# ============================================================
# 1B) F·disp 방향 penalty
# ============================================================

def compute_fd_direction_penalty(
    forces: np.ndarray,
    disp: np.ndarray,
    lambda_fd_penalty: float,
) -> Tuple[np.ndarray, float]:
    """
    Force와 displacement 방향이 반대일 때 페널티.

    ASE convention: F = -∇E
    ------------------------
    - 에너지 감소를 원하면 δx 를 F 방향으로 움직이는 것이 유리
      (E(x + δx) ≈ E(x) - F·δx 이므로, F·δx > 0 이면 ΔE < 0)
    - 따라서 F·disp < 0 인 경우를 벌점으로 본다.

    Parameters
    ----------
    forces : (N, 3)
        현재 step 직전의 force (old forces).
    disp : (N, 3)
        이번 step 에 적용된 per-atom displacement.
    lambda_fd_penalty : float
        misalignment 에 곱해줄 weight.

    Returns
    -------
    penalty_vec : (N,)
    penalty_mean : float
    """
    if forces.shape != disp.shape:
        raise ValueError(
            f"[fd_dir] shape mismatch: forces={forces.shape}, disp={disp.shape}"
        )

    dot = np.einsum("ij,ij->i", forces, disp)
    norm_f = np.linalg.norm(forces, axis=1)
    norm_d = np.linalg.norm(disp, axis=1)

    denom = np.maximum(norm_f * norm_d, 1e-8)
    cos = dot / denom  # [-1, 1] 근사

    # cos < 0 (force와 반대 방향) → penalty
    misalign = np.maximum(-cos, 0.0)  # cos>=0 → 0, cos=-1 → 1

    penalty_vec = lambda_fd_penalty * misalign
    penalty_mean = float(penalty_vec.mean())

    logger.debug(
        "[fd_dir] mean_cos=%.6f lambda=%.3f penalty_mean=%.6f",
        float(cos.mean()),
        lambda_fd_penalty,
        penalty_mean,
    )

    return penalty_vec.astype(np.float32), penalty_mean


# ============================================================
# 3A) Bond-subspace projection
# ============================================================

def project_to_bond_subspace(
    disp: np.ndarray,
    bond_dirs: np.ndarray,
) -> np.ndarray:
    """
    1D bond 축으로 per-atom displacement 를 투영.

    Parameters
    ----------
    disp : (N, 3)
        원래 per-atom displacement.
    bond_dirs : (N, 3)
        reset 시점에서 precompute 된 bond 기반 방향 벡터
        (모두 정규화되어 있다고 가정; norm=0 인 atom 은 그대로 둔다).

    Returns
    -------
    disp_proj : (N, 3)
        bond 축에 투영된 displacement.
    """
    if disp.shape != bond_dirs.shape:
        raise ValueError(
            f"[bond_proj] shape mismatch: disp={disp.shape}, "
            f"bond_dirs={bond_dirs.shape}"
        )

    d = disp.astype(np.float32, copy=False)
    b = bond_dirs.astype(np.float32, copy=False)

    b_norm2 = np.sum(b * b, axis=1, keepdims=True)  # (N, 1)
    valid = (b_norm2[:, 0] > 1e-8)

    disp_proj = d.copy()

    if not np.any(valid):
        # bond 방향이 정의된 atom 이 없으면 그대로 반환
        return disp_proj

    coeff = np.zeros_like(b_norm2, dtype=np.float32)
    # d·b / |b|^2
    coeff[valid] = (
        np.sum(d[valid] * b[valid], axis=1, keepdims=True) /
        b_norm2[valid]
    )

    disp_proj[valid] = coeff[valid] * b[valid]

    logger.debug(
        "[bond_proj] valid_atoms=%d / %d, "
        "mean_norm_before=%.6f, mean_norm_after=%.6f",
        int(valid.sum()),
        disp.shape[0],
        float(np.linalg.norm(d, axis=1).mean()),
        float(np.linalg.norm(disp_proj, axis=1).mean()),
    )

    return disp_proj.astype(np.float32, copy=False)


# ============================================================
# 3B) Local frame scaling (radial / tangent)
# ============================================================

def apply_local_frame_scaling(
    disp: np.ndarray,
    bond_dirs: np.ndarray,
    radial_scale: float,
    tangent_scale: float,
) -> np.ndarray:
    """
    bond 축 기준 local frame (radial / tangential) 스케일링.

    Parameters
    ----------
    disp : (N, 3)
    bond_dirs : (N, 3)
        reset 시점 기준으로 정규화된 bond 방향.
    radial_scale : float
        bond 축 방향 성분 scale.
    tangent_scale : float
        bond 축에 수직한 성분 scale.

    Returns
    -------
    disp_scaled : (N, 3)
    """
    if disp.shape != bond_dirs.shape:
        raise ValueError(
            f"[local_frame] shape mismatch: disp={disp.shape}, "
            f"bond_dirs={bond_dirs.shape}"
        )

    d = disp.astype(np.float32, copy=False)
    b = bond_dirs.astype(np.float32, copy=False)

    b_norm2 = np.sum(b * b, axis=1, keepdims=True)
    valid = (b_norm2[:, 0] > 1e-8)

    disp_scaled = d.copy()

    if not np.any(valid):
        return disp_scaled

    # radial component: (d·b / |b|^2) * b
    coeff = np.zeros_like(b_norm2, dtype=np.float32)
    coeff[valid] = (
        np.sum(d[valid] * b[valid], axis=1, keepdims=True) /
        b_norm2[valid]
    )
    d_radial = coeff * b
    d_tangent = d - d_radial

    disp_scaled[valid] = (
        radial_scale * d_radial[valid] +
        tangent_scale * d_tangent[valid]
    )

    logger.debug(
        "[local_frame] valid_atoms=%d / %d, radial_scale=%.3f, "
        "tangent_scale=%.3f, mean_norm_before=%.6f, mean_norm_after=%.6f",
        int(valid.sum()),
        disp.shape[0],
        radial_scale,
        tangent_scale,
        float(np.linalg.norm(d, axis=1).mean()),
        float(np.linalg.norm(disp_scaled, axis=1).mean()),
    )

    return disp_scaled.astype(np.float32, copy=False)
