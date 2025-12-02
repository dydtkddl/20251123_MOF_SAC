# utils/config_utils.py
# -*- coding: utf-8 -*-

"""
Phase2 설정 로더 (configs/train_phase2.yaml 전용)
================================================

- YAML 구조
  - train: TrainConfig
  - env:   EnvConfig
  - sac:   SACConfig
  - replay: ReplayConfig
  - bc:    BCConfig
  - modes: ModesConfig
  - bfgs:  BFGSConfig

- 기능
  1) YAML → dataclass (Phase2Config) 로 안전하게 변환
  2) 섹션별로 "필수 키 누락" / "알 수 없는 키"를 엄격하게 체크
  3) 주요 하이퍼파라미터 요약을 logging 으로 출력

YAML 예시는 configs/train_phase2.yaml 참고.
"""

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyYAML이 설치되어 있지 않습니다. "
        "`pip install pyyaml` 로 설치 후 다시 실행해주세요."
    ) from e


# ---------------------------------------------------------------------
# Logging 설정
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# Dataclasses 정의 (YAML 과 1:1 매핑)
# ---------------------------------------------------------------------


@dataclass
class TrainConfig:
    """
    train 섹션
    -----------
    - Global training schedule + surrogate/QMOF 설정
    """
    epochs: int
    base_steps: int
    final_steps: int
    horizon_sch: int

    fmax_thresh: float

    buffer_size: int
    batch_size: int
    warmup_transitions: int

    checkpoint_interval: int

    pool_dir: str
    mace_model_paths: Sequence[str]
    mace_head: str
    device: str


@dataclass
class EnvConfig:
    """
    env 섹션
    --------
    - MOFEnv 하이퍼파라미터
    - cmax curriculum, perturb 스케줄, reward 구성 옵션 등
    """
    k_neighbors: int

    cmax_min: float
    cmax_max: float
    cmax_sch_start_ep: int
    cmax_sch_end_ep: int

    random_perturb: bool
    sigma_min: float
    sigma_max: float
    max_perturb: float

    terminal_bonus_base: float
    time_penalty: float
    fail_penalty: float

    # 1A: Force norm 증가 패널티
    use_force_increase_penalty: bool
    lambda_force_up: float

    # 1B: F·disp 방향 패널티
    use_fd_direction_penalty: bool
    lambda_fd_penalty: float

    # 3A: Bond-subspace projection
    use_bond_projection: bool

    # 3B: Local frame scaling
    use_local_frame: bool
    radial_scale: float
    tangent_scale: float

    # 4A/4B: Mode 기반 액션/후처리 사용 여부 (env flag)
    use_mode_basis: bool


@dataclass
class SACConfig:
    """
    sac 섹션
    ---------
    - per-atom SAC 하이퍼파라미터
    - BC loss (on-policy BC+RL) 옵션 포함
    """
    lr: float
    gamma: float
    tau: float
    target_entropy: float

    use_bc_loss: bool
    bc_lambda: float
    bc_dataset_path: Optional[str]
    bc_batch_ratio: float


@dataclass
class ReplayConfig:
    """
    replay 섹션
    ------------
    - ReplayBuffer 크기 및 expert seeding 옵션
    """
    max_size: int
    log_interval: int
    use_expert_replay_seed: bool
    expert_replay_path: Optional[str]


@dataclass
class BCConfig:
    """
    bc 섹션
    --------
    - 사전 BC pretrain 용 설정 (optional)
    """
    use_pretrain: bool
    dataset_path: Optional[str]
    epochs: int
    batch_size: int
    output_path: str


@dataclass
class ModesConfig:
    """
    modes 섹션
    -----------
    - Graph/Fourier 기반 mode basis 설정
    - env.use_mode_basis 플래그와는 별개로,
      실제 mode type/num_modes/dir 을 관리
    """
    use_mode_basis: bool
    type: str
    num_modes: int
    dir: str


@dataclass
class BFGSConfig:
    """
    bfgs 섹션
    ----------
    - BFGS 기반 expert trajectory 생성용 설정
    """
    pool_dir: str
    output_dir: str
    num_structures: int
    max_steps: int
    fmax: float


@dataclass
class Phase2Config:
    """
    전체 Phase2 설정 컨테이너
    -------------------------
    train, env, sac, replay, bc, modes, bfgs 블록을 모두 포함.
    """
    train: TrainConfig
    env: EnvConfig
    sac: SACConfig
    replay: ReplayConfig
    bc: BCConfig
    modes: ModesConfig
    bfgs: BFGSConfig


# ---------------------------------------------------------------------
# 내부 유틸: 키 검증
# ---------------------------------------------------------------------


def _check_keys(
    section_name: str,
    d: Dict[str, Any],
    required: Sequence[str],
    optional: Sequence[str],
) -> None:
    """
    섹션별로 필수/옵션 키를 엄격하게 검증.

    - 필수 키 누락: KeyError 발생
    - 알 수 없는 키 존재: KeyError 발생
    """
    keys = set(d.keys())
    required_set = set(required)
    optional_set = set(optional)
    allowed = required_set | optional_set

    missing = required_set - keys
    if missing:
        raise KeyError(
            f"[CONFIG] Section '{section_name}' 에 필수 키가 누락되었습니다: "
            f"{sorted(missing)}"
        )

    unknown = keys - allowed
    if unknown:
        raise KeyError(
            f"[CONFIG] Section '{section_name}' 에 알 수 없는 키가 있습니다: "
            f"{sorted(unknown)}"
        )


# ---------------------------------------------------------------------
# YAML Dict → Dataclass 변환 함수들
# ---------------------------------------------------------------------


def _to_train_cfg(d: Dict[str, Any]) -> TrainConfig:
    required = [
        "epochs",
        "base_steps",
        "final_steps",
        "horizon_sch",
        "fmax_thresh",
        "buffer_size",
        "batch_size",
        "warmup_transitions",
        "checkpoint_interval",
        "pool_dir",
    ]
    optional = [
        "mace_model_paths",
        "mace_head",
        "device",
    ]
    _check_keys("train", d, required, optional)

    return TrainConfig(
        epochs=int(d["epochs"]),
        base_steps=int(d["base_steps"]),
        final_steps=int(d["final_steps"]),
        horizon_sch=int(d["horizon_sch"]),
        fmax_thresh=float(d["fmax_thresh"]),
        buffer_size=int(d["buffer_size"]),
        batch_size=int(d["batch_size"]),
        warmup_transitions=int(d["warmup_transitions"]),
        checkpoint_interval=int(d["checkpoint_interval"]),
        pool_dir=str(d["pool_dir"]),
        mace_model_paths=d.get("mace_model_paths", []),
        mace_head=str(d.get("mace_head", "pbe_d3")),
        device=str(d.get("device", "cuda")),
    )


def _to_env_cfg(d: Dict[str, Any]) -> EnvConfig:
    required = [
        "k_neighbors",
        "cmax_min",
        "cmax_max",
        "cmax_sch_start_ep",
        "cmax_sch_end_ep",
        "random_perturb",
        "sigma_min",
        "sigma_max",
        "max_perturb",
        "terminal_bonus_base",
        "time_penalty",
        "fail_penalty",
        "use_mode_basis",
    ]
    optional = [
        "use_force_increase_penalty",
        "lambda_force_up",
        "use_fd_direction_penalty",
        "lambda_fd_penalty",
        "use_bond_projection",
        "use_local_frame",
        "radial_scale",
        "tangent_scale",
    ]
    _check_keys("env", d, required, optional)

    return EnvConfig(
        k_neighbors=int(d["k_neighbors"]),
        cmax_min=float(d["cmax_min"]),
        cmax_max=float(d["cmax_max"]),
        cmax_sch_start_ep=int(d["cmax_sch_start_ep"]),
        cmax_sch_end_ep=int(d["cmax_sch_end_ep"]),
        random_perturb=bool(d["random_perturb"]),
        sigma_min=float(d["sigma_min"]),
        sigma_max=float(d["sigma_max"]),
        max_perturb=float(d["max_perturb"]),
        terminal_bonus_base=float(d["terminal_bonus_base"]),
        time_penalty=float(d["time_penalty"]),
        fail_penalty=float(d["fail_penalty"]),
        use_force_increase_penalty=bool(d.get("use_force_increase_penalty", False)),
        lambda_force_up=float(d.get("lambda_force_up", 2.0)),
        use_fd_direction_penalty=bool(d.get("use_fd_direction_penalty", False)),
        lambda_fd_penalty=float(d.get("lambda_fd_penalty", 1.0)),
        use_bond_projection=bool(d.get("use_bond_projection", False)),
        use_local_frame=bool(d.get("use_local_frame", False)),
        radial_scale=float(d.get("radial_scale", 1.0)),
        tangent_scale=float(d.get("tangent_scale", 1.0)),
        use_mode_basis=bool(d["use_mode_basis"]),
    )


def _to_sac_cfg(d: Dict[str, Any]) -> SACConfig:
    required = [
        "lr",
        "gamma",
        "tau",
    ]
    optional = [
        "target_entropy",
        "use_bc_loss",
        "bc_lambda",
        "bc_dataset_path",
        "bc_batch_ratio",
    ]
    _check_keys("sac", d, required, optional)

    return SACConfig(
        lr=float(d["lr"]),
        gamma=float(d["gamma"]),
        tau=float(d["tau"]),
        target_entropy=float(d.get("target_entropy", -1.0)),
        use_bc_loss=bool(d.get("use_bc_loss", False)),
        bc_lambda=float(d.get("bc_lambda", 0.1)),
        bc_dataset_path=d.get("bc_dataset_path"),
        bc_batch_ratio=float(d.get("bc_batch_ratio", 0.0)),
    )


def _to_replay_cfg(d: Dict[str, Any]) -> ReplayConfig:
    required = [
        "max_size",
        "log_interval",
    ]
    optional = [
        "use_expert_replay_seed",
        "expert_replay_path",
    ]
    _check_keys("replay", d, required, optional)

    return ReplayConfig(
        max_size=int(d["max_size"]),
        log_interval=int(d["log_interval"]),
        use_expert_replay_seed=bool(d.get("use_expert_replay_seed", False)),
        expert_replay_path=d.get("expert_replay_path"),
    )


def _to_bc_cfg(d: Dict[str, Any]) -> BCConfig:
    required = []
    optional = [
        "use_pretrain",
        "dataset_path",
        "epochs",
        "batch_size",
        "output_path",
    ]
    _check_keys("bc", d, required, optional)

    return BCConfig(
        use_pretrain=bool(d.get("use_pretrain", False)),
        dataset_path=d.get("dataset_path"),
        epochs=int(d.get("epochs", 0)),
        batch_size=int(d.get("batch_size", 0)),
        output_path=str(d.get("output_path", "checkpoints_phase2/actor_bc.pt")),
    )


def _to_modes_cfg(d: Dict[str, Any]) -> ModesConfig:
    required = [
        "use_mode_basis",
    ]
    optional = [
        "type",
        "num_modes",
        "dir",
    ]
    _check_keys("modes", d, required, optional)

    return ModesConfig(
        use_mode_basis=bool(d["use_mode_basis"]),
        type=str(d.get("type", "graph_eig")),
        num_modes=int(d.get("num_modes", 0)),
        dir=str(d.get("dir", "data/modes")),
    )


def _to_bfgs_cfg(d: Dict[str, Any]) -> BFGSConfig:
    required = [
        "pool_dir",
        "output_dir",
        "num_structures",
        "max_steps",
        "fmax",
    ]
    optional: Sequence[str] = []
    _check_keys("bfgs", d, required, optional)

    return BFGSConfig(
        pool_dir=str(d["pool_dir"]),
        output_dir=str(d["output_dir"]),
        num_structures=int(d["num_structures"]),
        max_steps=int(d["max_steps"]),
        fmax=float(d["fmax"]),
    )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def load_config(path: str) -> Phase2Config:
    """
    configs/train_phase2.yaml 을 읽어 Phase2Config 로 변환.

    Parameters
    ----------
    path : str
        YAML 설정 파일 경로

    Returns
    -------
    Phase2Config
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    logger.info(f"[CONFIG] Loading Phase2 config from: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"[CONFIG] YAML root 가 dict 형태가 아닙니다: type={type(raw)}"
        )

    required_top = {"train", "env", "sac", "replay", "bc", "modes", "bfgs"}
    top_keys = set(raw.keys())

    missing_top = required_top - top_keys
    if missing_top:
        raise KeyError(
            f"[CONFIG] Top-level 섹션이 누락되었습니다: {sorted(missing_top)}"
        )

    unknown_top = top_keys - required_top
    if unknown_top:
        raise KeyError(
            f"[CONFIG] 알 수 없는 top-level 섹션이 있습니다: {sorted(unknown_top)}"
        )

    train_cfg = _to_train_cfg(raw["train"])
    env_cfg = _to_env_cfg(raw["env"])
    sac_cfg = _to_sac_cfg(raw["sac"])
    replay_cfg = _to_replay_cfg(raw["replay"])
    bc_cfg = _to_bc_cfg(raw["bc"])
    modes_cfg = _to_modes_cfg(raw["modes"])
    bfgs_cfg = _to_bfgs_cfg(raw["bfgs"])

    cfg = Phase2Config(
        train=train_cfg,
        env=env_cfg,
        sac=sac_cfg,
        replay=replay_cfg,
        bc=bc_cfg,
        modes=modes_cfg,
        bfgs=bfgs_cfg,
    )

    # -------------------------------
    # 요약 로그
    # -------------------------------
    logger.info(
        "[CONFIG] Train: epochs=%d, steps=%d→%d (horizon_sch=%d), "
        "fmax_thresh=%.4f, buffer_size=%d, batch_size=%d, warmup=%d",
        cfg.train.epochs,
        cfg.train.base_steps,
        cfg.train.final_steps,
        cfg.train.horizon_sch,
        cfg.train.fmax_thresh,
        cfg.train.buffer_size,
        cfg.train.batch_size,
        cfg.train.warmup_transitions,
    )
    logger.info(
        "[CONFIG] Env: k=%d, cmax=%.3f→%.3f (ep %d→%d), "
        "perturb_sigma=%.3f→%.3f, max_perturb=%.3f, "
        "time_penalty=%.4f, fail_penalty=%.3f, use_mode_basis(env)=%s",
        cfg.env.k_neighbors,
        cfg.env.cmax_min,
        cfg.env.cmax_max,
        cfg.env.cmax_sch_start_ep,
        cfg.env.cmax_sch_end_ep,
        cfg.env.sigma_min,
        cfg.env.sigma_max,
        cfg.env.max_perturb,
        cfg.env.time_penalty,
        cfg.env.fail_penalty,
        cfg.env.use_mode_basis,
    )
    logger.info(
        "[CONFIG] SAC: lr=%.6f, gamma=%.4f, tau=%.4f, target_entropy=%.3f, "
        "use_bc_loss=%s, bc_lambda=%.3f, bc_batch_ratio=%.3f",
        cfg.sac.lr,
        cfg.sac.gamma,
        cfg.sac.tau,
        cfg.sac.target_entropy,
        cfg.sac.use_bc_loss,
        cfg.sac.bc_lambda,
        cfg.sac.bc_batch_ratio,
    )
    logger.info(
        "[CONFIG] Replay: max_size=%d, log_interval=%d, "
        "use_expert_replay_seed=%s, expert_replay_path=%s",
        cfg.replay.max_size,
        cfg.replay.log_interval,
        cfg.replay.use_expert_replay_seed,
        str(cfg.replay.expert_replay_path),
    )
    logger.info(
        "[CONFIG] Modes: use_mode_basis(modes)=%s, type=%s, "
        "num_modes=%d, dir=%s",
        cfg.modes.use_mode_basis,
        cfg.modes.type,
        cfg.modes.num_modes,
        cfg.modes.dir,
    )
    logger.info(
        "[CONFIG] BFGS: pool_dir=%s, output_dir=%s, n_structures=%d, "
        "max_steps=%d, fmax=%.4f",
        cfg.bfgs.pool_dir,
        cfg.bfgs.output_dir,
        cfg.bfgs.num_structures,
        cfg.bfgs.max_steps,
        cfg.bfgs.fmax,
    )

    return cfg
