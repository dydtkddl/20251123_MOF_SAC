###############################################################
# MultiAgentReplayBuffer
# -------------------------------------------------------------
# - per-atom transition 저장 (N은 매 episode/step마다 달라짐)
# - obs_dim, act_dim 은 고정 (per-atom feature dimension)
# - PER (Prioritized Experience Replay) + n-step return 지원
# - SAC의 Multi-Agent(원자=에이전트) 설정에 맞게 설계:
#     store_step(obs_atom, actions, reward_scalar, next_obs_atom, done)
#   → 내부에서 n-step 누적 후 per-atom 튜플로 flatten해서 버퍼에 push
###############################################################

import logging
from collections import deque
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None  # torch 없이도 기본 동작은 하도록

logger = logging.getLogger("utils.replay_buffer")


class MultiAgentReplayBuffer:
    """
    Multi-Agent PER Replay Buffer (per-atom transitions).

    - 하나의 step은 구조 전체 (N개 원자)에 대해
        obs_atom: (N, obs_dim)
        actions:  (N, act_dim)
        reward:   scalar (구조 단위 reward)
        next_obs: (N, obs_dim)
        done:     bool
      을 갖고 있음.
    - store_step(...)에서 이를 받아서,
      - n-step이면 내부 n-step 버퍼에 쌓았다가
      - n-step return 계산 후 각 원자별 튜플로 flatten하여 저장.

    저장되는 transition (per-atom):
        (obs_i, act_i, R_nstep, next_obs_i, done_n)

    PER:
        p_i ∝ (|δ_i| + eps)^α
        샘플링 시 중요도 가중치 w_i 사용.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_size: int,
        batch_size: int = 256,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sampling: float = 1e-4,
        n_step: int = 1,
        gamma: float = 0.99,
        seed: int = 42,
        device: Optional[str] = None,
        enable_tqdm: bool = False,
    ):
        """
        Parameters
        ----------
        obs_dim : int
            Per-atom observation dimension.
        act_dim : int
            Per-atom action dimension.
        max_size : int
            Maximum number of per-atom transitions.
        batch_size : int
            Default mini-batch size for sampling.
        alpha : float
            PER exponent (0 → uniform, 1 → full prioritization).
        beta : float
            Importance-sampling exponent.
        beta_increment_per_sampling : float
            Sampling할 때마다 beta를 조금씩 증가시키는 정도.
        n_step : int
            n-step TD. 1이면 일반 1-step.
        gamma : float
            Discount factor.
        seed : int
            난수 seed.
        device : str or None
            "cpu", "cuda" 등. torch 텐서로 변환 시 사용.
        enable_tqdm : bool
            내부 큰 loop에 tqdm progress bar를 활성화할지 여부.
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_inc = float(beta_increment_per_sampling)

        self.n_step = int(max(1, n_step))
        self.gamma = float(gamma)

        self.device = device
        self.enable_tqdm = enable_tqdm

        # 메모리 할당
        self.obs_buf = np.zeros((self.max_size, self.obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.max_size, self.obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((self.max_size, self.act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((self.max_size,), dtype=np.float32)
        self.done_buf = np.zeros((self.max_size,), dtype=np.float32)
        self.priorities = np.zeros((self.max_size,), dtype=np.float32)

        # Circular pointer
        self.ptr = 0
        self.size = 0

        # PER max priority (새로운 튜플의 기본 priority)
        self._eps = 1e-6
        self.max_priority = 1.0

        # n-step 버퍼 (구조 단위 step을 쌓아둔다)
        # 각 element:
        #   {
        #       "obs": (N, obs_dim),
        #       "acts": (N, act_dim),
        #       "reward": float,
        #       "next_obs": (N, obs_dim),
        #       "done": bool
        #   }
        self.nstep_buffer: deque = deque(maxlen=self.n_step)

        # RNG
        self.rng = np.random.RandomState(seed)

        logger.info(
            "[MultiAgentReplayBuffer.__init__] obs_dim=%d, act_dim=%d, max_size=%d, "
            "alpha=%.3f, beta=%.3f, n_step=%d, gamma=%.3f",
            self.obs_dim,
            self.act_dim,
            self.max_size,
            self.alpha,
            self.beta,
            self.n_step,
            self.gamma,
        )

    # ------------------------------------------------------------------
    # 기본 정보
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.size

    @property
    def capacity(self) -> int:
        return self.max_size

    @property
    def filled_ratio(self) -> float:
        return float(self.size) / float(self.max_size + 1e-12)

    # ------------------------------------------------------------------
    # 내부 로우 단위 저장 (per-atom)
    # ------------------------------------------------------------------
    def _store_one(
        self,
        obs_i: np.ndarray,
        act_i: np.ndarray,
        rew: float,
        next_obs_i: np.ndarray,
        done: bool,
    ):
        """
        Per-atom transition 하나를 버퍼에 저장.
        """
        idx = self.ptr

        self.obs_buf[idx] = obs_i
        self.acts_buf[idx] = act_i
        self.rew_buf[idx] = rew
        self.next_obs_buf[idx] = next_obs_i
        self.done_buf[idx] = float(done)

        # 새 transition의 priority는 현재 max_priority로 초기화
        self.priorities[idx] = self.max_priority

        # 포인터 이동
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # ------------------------------------------------------------------
    # n-step 관련 처리
    # ------------------------------------------------------------------
    def _push_nstep_buffer(
        self,
        obs_atom: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs_atom: np.ndarray,
        done: bool,
    ):
        """
        구조 단위 step을 n-step 버퍼에 push.
        """
        step_item = {
            "obs": obs_atom.astype(np.float32),
            "acts": actions.astype(np.float32),
            "reward": float(reward),
            "next_obs": next_obs_atom.astype(np.float32),
            "done": bool(done),
        }
        self.nstep_buffer.append(step_item)

    def _pop_nstep_transition(self) -> Optional[Dict[str, Any]]:
        """
        n-step 버퍼에서 가장 오래된 step을 기준으로 n-step return transition 생성.
        (버퍼 길이가 n_step 이상일 때만 유효)

        Returns
        -------
        dict with keys:
            "obs_0", "acts_0", "R_n", "next_obs_n", "done_n"
        or None
        """
        if len(self.nstep_buffer) < self.n_step:
            return None

        # n-step return 계산
        R = 0.0
        gamma = 1.0
        done_n = False
        next_obs_n = self.nstep_buffer[-1]["next_obs"]

        for k in range(self.n_step):
            item = self.nstep_buffer[k]
            R += gamma * item["reward"]
            gamma *= self.gamma
            if item["done"]:
                done_n = True
                next_obs_n = item["next_obs"]
                break

        first = self.nstep_buffer[0]

        out = {
            "obs_0": first["obs"],       # (N, obs_dim)
            "acts_0": first["acts"],     # (N, act_dim)
            "R_n": float(R),
            "next_obs_n": next_obs_n,    # (N, obs_dim)
            "done_n": bool(done_n),
        }

        # 가장 오래된 것 pop
        self.nstep_buffer.popleft()

        return out

    def _flush_remaining_nstep(self):
        """
        에피소드가 끝나고 남은 n-step 버퍼 내용도
        가능한 만큼 n-step transition으로 변환해 버퍼에 저장.
        """
        if len(self.nstep_buffer) == 0:
            return

        logger.debug(
            "[MultiAgentReplayBuffer._flush_remaining_nstep] flushing %d steps",
            len(self.nstep_buffer),
        )

        while len(self.nstep_buffer) > 0:
            R = 0.0
            gamma = 1.0
            done_n = False
            next_obs_n = self.nstep_buffer[-1]["next_obs"]

            for k, item in enumerate(self.nstep_buffer):
                R += gamma * item["reward"]
                gamma *= self.gamma
                if item["done"]:
                    done_n = True
                    next_obs_n = item["next_obs"]
                    break

            first = self.nstep_buffer[0]
            obs_0 = first["obs"]
            acts_0 = first["acts"]

            self._store_many_per_atom(
                obs_0,
                acts_0,
                float(R),
                next_obs_n,
                done_n,
                desc="Flush n-step buffer",
            )

            self.nstep_buffer.popleft()

    # ------------------------------------------------------------------
    # per-atom으로 flatten해서 여러 개 저장
    # ------------------------------------------------------------------
    def _store_many_per_atom(
        self,
        obs_atom: np.ndarray,
        actions: np.ndarray,
        R_n: float,
        next_obs_atom: np.ndarray,
        done_n: bool,
        desc: str = "Store transitions",
    ):
        """
        (N, obs_dim), (N, act_dim) → per-atom 튜플을 반복 저장.
        loop에 tqdm를 감싸서 진행 상황 확인 가능.
        """
        N = obs_atom.shape[0]
        iterator = range(N)
        if self.enable_tqdm:
            iterator = tqdm(
                iterator,
                desc=desc,
                leave=False,
            )

        for i in iterator:
            self._store_one(
                obs_i=obs_atom[i],
                act_i=actions[i],
                rew=R_n,
                next_obs_i=next_obs_atom[i],
                done=done_n,
            )

    # ------------------------------------------------------------------
    # Public: 매 step마다 호출 (기존 API)
    # ------------------------------------------------------------------
    def store_step(
        self,
        obs_atom: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs_atom: np.ndarray,
        done: bool,
    ):
        """
        환경에서 한 step이 끝날 때마다 호출.

        Parameters
        ----------
        obs_atom      : (N, obs_dim) float32
        actions       : (N, act_dim) float32
        reward        : float (structure-level reward)
        next_obs_atom : (N, obs_dim) float32
        done          : bool (episode 종료 여부)
        """
        if self.n_step == 1:
            # 1-step 모드는 바로 per-atom으로 저장
            self._store_many_per_atom(
                obs_atom=obs_atom,
                actions=actions,
                R_n=float(reward),
                next_obs_atom=next_obs_atom,
                done_n=done,
                desc="Store 1-step transitions",
            )
        else:
            # n-step 모드: 일단 내부 버퍼에 쌓고,
            # 일정 길이가 되면 n-step return으로 flatten
            self._push_nstep_buffer(obs_atom, actions, reward, next_obs_atom, done)
            trans = self._pop_nstep_transition()
            if trans is not None:
                self._store_many_per_atom(
                    obs_atom=trans["obs_0"],
                    actions=trans["acts_0"],
                    R_n=trans["R_n"],
                    next_obs_atom=trans["next_obs_n"],
                    done_n=trans["done_n"],
                    desc="Store n-step transitions",
                )

        # capacity에 대한 milestone 로깅
        fr = self.filled_ratio
        if abs(fr - 0.25) < 1e-3 or abs(fr - 0.5) < 1e-3 or abs(fr - 0.75) < 1e-3:
            logger.info(
                "[MultiAgentReplayBuffer.store_step] buffer filled_ratio=%.3f (size=%d)",
                fr,
                self.size,
            )

    # ------------------------------------------------------------------
    # Public: SAC용 래퍼 메서드 (새 API)
    # ------------------------------------------------------------------
    def store(
        self,
        obs: np.ndarray,
        acts: np.ndarray,
        rews: float,
        next_obs: np.ndarray,
        done: bool,
        atom_type: Optional[np.ndarray] = None,
        next_atom_type: Optional[np.ndarray] = None,
        global_feat: Optional[np.ndarray] = None,
        next_global: Optional[np.ndarray] = None,
    ):
        """
        SAC 쪽에서 일반적으로 사용하는 인터페이스에 맞춘 래퍼.

        Parameters
        ----------
        obs : (N, obs_dim)
        acts : (N, act_dim)
        rews : float
        next_obs : (N, obs_dim)
        done : bool
        atom_type, next_atom_type, global_feat, next_global :
            현재 구현에서는 사용하지 않으며, 향후 확장을 위해 인터페이스만 맞춰둠.
        """
        self.store_step(
            obs_atom=obs,
            actions=acts,
            reward=rews,
            next_obs_atom=next_obs,
            done=done,
        )

    def on_episode_end(self):
        """
        에피소드 종료 시 호출.
        남은 n-step 버퍼를 flush해서 최대한 버리지 않고 저장.
        """
        if self.n_step > 1:
            self._flush_remaining_nstep()
        self.nstep_buffer.clear()

    # ------------------------------------------------------------------
    # PER 샘플링
    # ------------------------------------------------------------------
    def sample(self, batch_size: Optional[int] = None):
        """
        PER 기반 mini-batch 샘플링.

        Returns
        -------
        batch, idxs, weights

        batch : dict
            {
                "obs": np.ndarray (B, obs_dim),
                "acts": np.ndarray (B, act_dim),
                "rews": np.ndarray (B,),
                "next_obs": np.ndarray (B, obs_dim),
                "done": np.ndarray (B,),
                "idxs": np.ndarray (B,),
                "weights": np.ndarray (B,),
                # torch 텐서 (torch가 있다면):
                "obs_t": torch.Tensor (B, obs_dim),
                "acts_t": torch.Tensor (B, act_dim),
                "rews_t": torch.Tensor (B, 1),
                "next_obs_t": torch.Tensor (B, obs_dim),
                "done_t": torch.Tensor (B, 1),
                "weights_t": torch.Tensor (B, 1)
            }

        idxs : np.ndarray (B,)
            선택된 버퍼 인덱스 (int64)

        weights : np.ndarray (B,)
            중요도 가중치 (float32)
        """
        if self.size == 0:
            raise RuntimeError("Replay buffer is empty; cannot sample.")

        if batch_size is None:
            batch_size = self.batch_size

        valid_prios = self.priorities[: self.size]
        if np.all(valid_prios == 0):
            # priority가 아직 안 세팅되어 있다면 uniform
            probs = np.full(self.size, 1.0 / self.size, dtype=np.float32)
        else:
            scaled_prios = valid_prios ** self.alpha
            probs = scaled_prios / np.sum(scaled_prios)

        idxs = self.rng.choice(self.size, size=batch_size, p=probs)

        # 중요도 가중치
        self.beta = min(1.0, self.beta + self.beta_inc)
        weights = (self.size * probs[idxs]) ** (-self.beta)
        weights /= weights.max() + 1e-12
        weights = weights.astype(np.float32)

        batch = {
            "obs": self.obs_buf[idxs].copy(),
            "acts": self.acts_buf[idxs].copy(),
            "rews": self.rew_buf[idxs].copy(),
            "next_obs": self.next_obs_buf[idxs].copy(),
            "done": self.done_buf[idxs].copy(),   # ★ 키 이름: done
            "idxs": idxs.astype(np.int64),
            "weights": weights,
        }

        # torch 텐서로도 반환 (가능한 경우)
        if torch is not None and self.device is not None:
            device = self.device
            batch["obs_t"] = torch.as_tensor(batch["obs"], device=device, dtype=torch.float32)
            batch["acts_t"] = torch.as_tensor(batch["acts"], device=device, dtype=torch.float32)
            batch["rews_t"] = torch.as_tensor(batch["rews"], device=device, dtype=torch.float32).unsqueeze(-1)
            batch["next_obs_t"] = torch.as_tensor(batch["next_obs"], device=device, dtype=torch.float32)
            batch["done_t"] = torch.as_tensor(batch["done"], device=device, dtype=torch.float32).unsqueeze(-1)
            batch["weights_t"] = torch.as_tensor(batch["weights"], device=device, dtype=torch.float32).unsqueeze(-1)

        logger.debug(
            "[MultiAgentReplayBuffer.sample] batch_size=%d, beta=%.3f",
            batch_size,
            self.beta,
        )

        # ★ 기존: return batch
        #    명세에 따라 (batch, idxs, weights) 반환
        return batch, idxs.astype(np.int64), weights

    # ------------------------------------------------------------------
    # PER priority 업데이트
    # ------------------------------------------------------------------
    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray):
        """
        Critic TD-error를 받아서 priority를 업데이트.

        Parameters
        ----------
        idxs : (B,) int
            buffer index들.
        td_errors : (B,) or (B,1)
            TD error.
        """
        idxs = np.asarray(idxs, dtype=np.int64)
        if td_errors.ndim > 1:
            td_errors = td_errors.reshape(-1)
        td_errors = np.abs(td_errors) + self._eps

        for i, idx in enumerate(idxs):
            p = float(td_errors[i])
            self.priorities[idx] = p
            if p > self.max_priority:
                self.max_priority = p

        logger.debug(
            "[MultiAgentReplayBuffer.update_priorities] updated %d priorities, max_priority=%.4f",
            len(idxs),
            self.max_priority,
        )

    # ★ alias 메서드: SAC 코드에서 update_priority(...)를 호출해도 동작하도록
    def update_priority(self, idxs: np.ndarray, td_errors: np.ndarray):
        """
        Alias for update_priorities, to match external SAC interface.
        """
        return self.update_priorities(idxs, td_errors)

    # ------------------------------------------------------------------
    # Debug / Inspect 용 메서드
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """
        버퍼 상태 요약 정보 반환.
        """
        if self.size == 0:
            min_p = 0.0
            max_p = 0.0
            mean_p = 0.0
        else:
            valid_p = self.priorities[: self.size]
            min_p = float(np.min(valid_p))
            max_p = float(np.max(valid_p))
            mean_p = float(np.mean(valid_p))

        out = {
            "size": self.size,
            "capacity": self.max_size,
            "filled_ratio": self.filled_ratio,
            "min_priority": min_p,
            "max_priority": max_p,
            "mean_priority": mean_p,
            "n_step": self.n_step,
            "gamma": self.gamma,
        }
        logger.info(
            "[MultiAgentReplayBuffer.stats] size=%d, capacity=%d, filled_ratio=%.3f, "
            "min_p=%.4e, max_p=%.4e, mean_p=%.4e, n_step=%d, gamma=%.3f",
            out["size"],
            out["capacity"],
            out["filled_ratio"],
            out["min_priority"],
            out["max_priority"],
            out["mean_priority"],
            out["n_step"],
            out["gamma"],
        )
        return out
