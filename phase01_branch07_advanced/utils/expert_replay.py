# utils/expert_replay.py
# -*- coding: utf-8 -*-

"""
Expert replay seeding utilities
===============================

- Expert (BFGS / offline policy) trajectory를 ReplayBuffer에 채워 넣기 위한 헬퍼.
- npz 파일에서 ('obs', 'act', 'rew', 'nobs', 'done') 키를 읽어 dict로 반환.
- ReplayBuffer에 최대 max_samples까지 주입 (필요시 셔플).
- act_dim mismatch (3D -> 4D) 자동 패딩 지원:
    * expert act_dim == 3, replay.act_dim == 4 인 경우:
        gate = 1.0 을 앞에 붙여 [1.0, dx, dy, dz]로 변환.
"""

import os
import logging
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


REQUIRED_KEYS = ("obs", "act", "rew", "nobs", "done")


def load_expert_replay_npz(
    path: str,
    allow_pickle: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Load expert replay data from a .npz file.

    Parameters
    ----------
    path : str
        Path to .npz file that contains at least the keys:
        'obs', 'act', 'rew', 'nobs', 'done'.
    allow_pickle : bool
        Passed to np.load (default False).

    Returns
    -------
    data : dict
        {
            "obs"  : (N, obs_dim) float32,
            "act"  : (N, act_dim) float32,
            "rew"  : (N,) float32,
            "nobs" : (N, obs_dim) float32,
            "done" : (N,) bool,
        }

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    KeyError
        If any required key is missing.
    ValueError
        If shapes are inconsistent.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ExpertReplay] File not found: {path}")

    logger.info(f"[ExpertReplay] Loading expert replay npz: {path}")
    npz = np.load(path, allow_pickle=allow_pickle)

    for k in REQUIRED_KEYS:
        if k not in npz:
            raise KeyError(
                f"[ExpertReplay] Missing key '{k}' in npz file: {path}"
            )

    obs = np.asarray(npz["obs"], dtype=np.float32)
    act = np.asarray(npz["act"], dtype=np.float32)
    rew = np.asarray(npz["rew"], dtype=np.float32)
    nobs = np.asarray(npz["nobs"], dtype=np.float32)
    done = np.asarray(npz["done"])

    if obs.shape != nobs.shape:
        raise ValueError(
            f"[ExpertReplay] obs shape {obs.shape} "
            f"!= nobs shape {nobs.shape}"
        )
    if obs.shape[0] != act.shape[0] or obs.shape[0] != rew.shape[0] \
       or obs.shape[0] != done.shape[0]:
        raise ValueError(
            "[ExpertReplay] Inconsistent first dimension among "
            f"obs({obs.shape[0]}), act({act.shape[0]}), "
            f"rew({rew.shape[0]}), done({done.shape[0]})"
        )

    N = obs.shape[0]
    obs_dim = obs.shape[1]
    act_dim = act.shape[1] if act.ndim == 2 else 1

    logger.info(
        "[ExpertReplay] Loaded expert data: N=%d, obs_dim=%d, act_dim=%d",
        N, obs_dim, act_dim,
    )

    # done을 bool로 고정
    if done.dtype != np.bool_:
        logger.info(
            "[ExpertReplay] Casting 'done' array from %s to bool",
            done.dtype,
        )
        done = done.astype(np.bool_)

    data = dict(obs=obs, act=act, rew=rew, nobs=nobs, done=done)
    return data


def seed_replay_from_expert(
    replay,
    expert_data: Dict[str, np.ndarray],
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    reset_buffer: bool = True,
    use_tqdm: bool = True,
    desc: str = "[Expert→Replay seed]",
    logger_override: Optional[logging.Logger] = None,
):
    """
    Seed ReplayBuffer with expert transitions.

    Parameters
    ----------
    replay : ReplayBuffer
        Target replay buffer instance.
    expert_data : dict
        Dict with keys 'obs', 'act', 'rew', 'nobs', 'done'.
    max_samples : int, optional
        Maximum number of transitions to load. If None, use all.
        Also clipped by replay.max_size.
    shuffle : bool
        Whether to shuffle expert indices before seeding.
    reset_buffer : bool
        If True, reset replay.ptr and replay.size before seeding.
    use_tqdm : bool
        If True, show tqdm progress bar.
    desc : str
        tqdm description string.
    logger_override : logging.Logger, optional
        If provided, use this logger instead of module-level logger.
    """
    log = logger_override if logger_override is not None else logger

    obs = expert_data["obs"]
    act = expert_data["act"]
    rew = expert_data["rew"]
    nobs = expert_data["nobs"]
    done = expert_data["done"]

    N = obs.shape[0]
    obs_dim = obs.shape[1]
    act_dim_expert = act.shape[1] if act.ndim == 2 else 1

    log.info(
        "[ExpertReplay] Start seeding replay from expert data "
        "(N=%d, obs_dim=%d, act_dim_expert=%d, "
        "replay.obs_dim=%d, replay.act_dim=%d, replay.max_size=%d)",
        N, obs_dim, act_dim_expert,
        replay.obs_dim, replay.act_dim, replay.max_size,
    )

    # Obs dim 체크 (Mismatch면 바로 에러)
    if obs_dim != replay.obs_dim:
        raise ValueError(
            f"[ExpertReplay] obs_dim mismatch: expert={obs_dim}, "
            f"replay={replay.obs_dim}"
        )

    # 사용할 샘플 수 결정
    max_by_buffer = replay.max_size
    if max_samples is None:
        num = min(N, max_by_buffer)
    else:
        num = min(N, max_samples, max_by_buffer)

    if num <= 0:
        log.warning("[ExpertReplay] No samples to seed (num=%d).", num)
        return

    idxs = np.arange(N, dtype=np.int64)
    if shuffle:
        np.random.shuffle(idxs)

    idxs = idxs[:num]

    log.info(
        "[ExpertReplay] Effective seeding samples: %d (of N=%d), "
        "shuffle=%s, reset_buffer=%s",
        num, N, shuffle, reset_buffer,
    )

    # 기존 buffer 초기화 옵션
    if reset_buffer:
        log.info(
            "[ExpertReplay] Resetting replay buffer before seeding "
            "(previous size=%d, ptr=%d)",
            replay.size, replay.ptr,
        )
        replay.ptr = 0
        replay.size = 0

    iterator = idxs
    if use_tqdm:
        iterator = tqdm(idxs, desc=desc, ncols=120)

    # 3D → 4D (gate padding) 자동 처리 여부 로그
    if act_dim_expert == 3 and replay.act_dim == 4:
        log.info(
            "[ExpertReplay] Detected act_dim_expert=3, replay.act_dim=4. "
            "Will automatically pad gate=1.0 → [1.0, dx, dy, dz]."
        )
    elif act_dim_expert != replay.act_dim:
        raise ValueError(
            f"[ExpertReplay] act_dim mismatch not supported: "
            f"expert={act_dim_expert}, replay={replay.act_dim}"
        )

    # 실제 시드 진행
    stored = 0
    for k in iterator:
        obs_i = obs[k]          # (obs_dim,)
        act_i = act[k]          # (expert_act_dim,)
        rew_i = float(rew[k])   # scalar
        nobs_i = nobs[k]        # (obs_dim,)
        done_i = bool(done[k])

        # shape 변환/검증
        if act_dim_expert == replay.act_dim:
            act_use = act_i.astype(np.float32, copy=False)
        else:
            # expert=3, replay=4 → gate=1.0 패딩
            gate = np.ones((1,), dtype=np.float32)
            act_use = np.concatenate(
                [gate, act_i.astype(np.float32, copy=False)],
                axis=-1,
            )

        # ReplayBuffer.store 내부에서 obs_dim 검사/로깅 수행
        replay.store(
            obs_i.astype(np.float32, copy=False),
            act_use,
            rew_i,
            nobs_i.astype(np.float32, copy=False),
            done_i,
        )
        stored += 1

    log.info(
        "[ExpertReplay] Seeding done: stored=%d, replay.size=%d, ptr=%d",
        stored, len(replay), replay.ptr,
    )
# utils/expert_replay.py

import os
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


REQUIRED_KEYS = ("obs", "act", "nobs", "rew", "done")


def load_expert_as_replay(path: str) -> Dict[str, np.ndarray]:
    """
    Expert npz를 ReplayBuffer와 동일한 dict 포맷으로 로딩.

    Parameters
    ----------
    path : str
        npz 파일 경로

    Returns
    -------
    replay : dict
        {
            'obs'  : (K, obs_dim),
            'act'  : (K, act_dim),
            'nobs' : (K, obs_dim),
            'rew'  : (K,),
            'done' : (K,),
        }
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"[expert_replay] file not found: {path}")

    logger.info(f"[expert_replay] Loading expert replay from: {path}")
    data = np.load(path, allow_pickle=False)

    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise KeyError(
            f"[expert_replay] Missing keys in {path}: {missing}. "
            f"Required keys: {REQUIRED_KEYS}"
        )

    replay = {k: data[k] for k in REQUIRED_KEYS}

    K = replay["obs"].shape[0]
    logger.info(
        f"[expert_replay] Loaded expert replay: K={K}, "
        f"obs_dim={replay['obs'].shape[1]}, act_dim={replay['act'].shape[1]}"
    )
    return replay


def save_replay_dict(path: str, replay: Dict[str, Any]) -> None:
    """
    ReplayBuffer 스타일 dict를 npz로 저장.

    Parameters
    ----------
    path : str
        저장할 npz 경로
    replay : dict
        REQUIRED_KEYS가 모두 포함된 dict
    """
    missing = [k for k in REQUIRED_KEYS if k not in replay]
    if missing:
        raise KeyError(
            f"[expert_replay.save_replay_dict] missing keys: {missing}. "
            f"Required: {REQUIRED_KEYS}"
        )

    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    np.savez_compressed(path, **{k: replay[k] for k in REQUIRED_KEYS})
    logger.info(f"[expert_replay] Saved replay dict to: {path}")
