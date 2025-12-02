# sac/expert_dataset.py

import os
import glob
import logging
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class ExpertDataset(Dataset):
    """
    BFGS 기반 expert npz 파일들을 묶어서 제공하는 PyTorch Dataset.

    각 npz 파일은 gen_bfgs_trajectories.py에서 생성된 포맷을 따른다고 가정:
        - obs      : (N_samples, obs_dim)
        - disp     : (N_samples, 3)
        - act4     : (N_samples, 4)
        - traj_id  : (N_samples,)
        - step_idx : (N_samples,)
        - atom_idx : (N_samples,)
        - natoms   : scalar
        - obs_dim  : scalar
        - cif_path : (1,) string array

    Parameters
    ----------
    npz_paths : list[str]
        사용할 npz 파일 경로 리스트.
    mode : str
        "disp"  → (obs, disp) 반환 (disp: 3D 변위)
        "act4"  → (obs, act4) 반환 (4D action)
        "both"  → dict 형태로 obs, disp, act4 모두 반환
    in_memory : bool
        True면 모든 npz 내용을 메모리에 적재.
        False면 numpy mmap_mode="r"로 lazy load.
    """

    def __init__(
        self,
        npz_paths: Sequence[str],
        mode: str = "disp",
        in_memory: bool = False,
    ):
        super().__init__()

        assert mode in ("disp", "act4", "both"), \
            f"Invalid mode={mode}, choose from 'disp', 'act4', 'both'."

        self.mode = mode
        self.in_memory = in_memory

        if isinstance(npz_paths, (str, bytes)):
            npz_paths = [npz_paths]

        self._files: List[str] = []
        for path in npz_paths:
            if os.path.isdir(path):
                # 디렉토리면 *.npz 스캔
                files = sorted(glob.glob(os.path.join(path, "*.npz")))
                self._files.extend(files)
            else:
                self._files.append(path)

        if not self._files:
            raise ValueError("No expert npz files found for ExpertDataset.")

        logger.info(f"[ExpertDataset] npz files: {len(self._files)}")
        logger.info(f"[ExpertDataset] mode={self.mode}, in_memory={self.in_memory}")

        # 각 파일에 대한 meta 정보
        self._data_objs = []     # np.load 결과 (in_memory=False면 mmap object)
        self._lengths = []       # 각 파일별 sample 수
        self._cum_lengths = []   # prefix sum
        self._obs_dim = None

        total = 0
        for fpath in tqdm(self._files, desc="Load Expert npz (meta)", ncols=120):
            if self.in_memory:
                data = np.load(fpath, allow_pickle=True)
            else:
                data = np.load(fpath, mmap_mode="r", allow_pickle=True)

            if "obs" not in data:
                raise ValueError(f"File {fpath} has no 'obs' field.")

            n_samples = data["obs"].shape[0]
            self._data_objs.append(data)
            self._lengths.append(n_samples)
            total += n_samples
            self._cum_lengths.append(total)

            if self._obs_dim is None:
                self._obs_dim = int(data["obs"].shape[1])
            else:
                if int(data["obs"].shape[1]) != self._obs_dim:
                    logger.warning(
                        f"[ExpertDataset] obs_dim mismatch: {self._obs_dim} vs "
                        f"{int(data['obs'].shape[1])} in {fpath}"
                    )

            logger.info(
                f"[ExpertDataset] Loaded meta: {fpath}, samples={n_samples}, "
                f"obs_dim={data['obs'].shape[1]}"
            )

        self._total_len = total
        logger.info(
            f"[ExpertDataset] Total samples={self._total_len}, "
            f"files={len(self._files)}, obs_dim={self._obs_dim}"
        )

    # --------------------------------------------------------
    @property
    def obs_dim(self) -> Optional[int]:
        return self._obs_dim

    @property
    def num_files(self) -> int:
        return len(self._files)

    # --------------------------------------------------------
    def __len__(self):
        return self._total_len

    def _locate_index(self, idx: int):
        """
        전체 index → (file_idx, local_idx) 변환.
        """
        if idx < 0:
            idx = self._total_len + idx
        if not (0 <= idx < self._total_len):
            raise IndexError(f"Index {idx} out of range (0..{self._total_len-1})")

        # 선형 탐색도 상관 없지만, 파일이 많아질 수 있으니 이분 탐색
        lo, hi = 0, len(self._cum_lengths) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cum_lengths[mid]:
                hi = mid
            else:
                lo = mid + 1
        file_idx = lo
        prev_cum = 0 if file_idx == 0 else self._cum_lengths[file_idx - 1]
        local_idx = idx - prev_cum
        return file_idx, local_idx

    def __getitem__(self, idx: int):
        file_idx, local_idx = self._locate_index(idx)
        data = self._data_objs[file_idx]

        obs = data["obs"][local_idx]      # (obs_dim,)
        disp = data["disp"][local_idx]    # (3,)
        act4 = data["act4"][local_idx]    # (4,)

        # 추가 meta도 원하면 여기서 꺼낼 수 있음
        traj_id = data["traj_id"][local_idx]
        step_idx = data["step_idx"][local_idx]
        atom_idx = data["atom_idx"][local_idx]
        # natoms, cif_path 등은 파일 전체에 공통 => 필요 시 외부에서 data_obj로 꺼내 쓰면 됨

        if self.mode == "disp":
            sample = {
                "obs": torch.as_tensor(obs, dtype=torch.float32),
                "target": torch.as_tensor(disp, dtype=torch.float32),
                "traj_id": int(traj_id),
                "step_idx": int(step_idx),
                "atom_idx": int(atom_idx),
            }
        elif self.mode == "act4":
            sample = {
                "obs": torch.as_tensor(obs, dtype=torch.float32),
                "target": torch.as_tensor(act4, dtype=torch.float32),
                "traj_id": int(traj_id),
                "step_idx": int(step_idx),
                "atom_idx": int(atom_idx),
            }
        else:  # both
            sample = {
                "obs": torch.as_tensor(obs, dtype=torch.float32),
                "disp": torch.as_tensor(disp, dtype=torch.float32),
                "act4": torch.as_tensor(act4, dtype=torch.float32),
                "traj_id": int(traj_id),
                "step_idx": int(step_idx),
                "atom_idx": int(atom_idx),
            }

        return sample


# ============================================================
# Quick test helper (optional)
# ============================================================

def _quick_test():
    """
    간단한 셀프 테스트용 함수.
    실제 사용 시에는 train 스크립트에서 Dataset을 import해서 사용하면 됨.
    """
    import argparse
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        type=str,
        default="./expert_data_bfgs",
        help="Directory that contains EXP_*.npz files.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="disp",
        choices=["disp", "act4", "both"],
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    args = ap.parse_args()

    ds = ExpertDataset(args.data_dir, mode=args.mode, in_memory=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    logger.info(
        f"[TEST] Dataset length={len(ds)}, obs_dim={ds.obs_dim}, "
        f"num_files={ds.num_files}"
    )

    for i, batch in enumerate(dl):
        if args.mode == "both":
            obs = batch["obs"]
            disp = batch["disp"]
            act4 = batch["act4"]
            logger.info(
                f"[TEST] Batch {i}: obs={obs.shape}, disp={disp.shape}, act4={act4.shape}"
            )
        else:
            obs = batch["obs"]
            target = batch["target"]
            logger.info(
                f"[TEST] Batch {i}: obs={obs.shape}, target={target.shape}"
            )
        if i >= 2:
            break


if __name__ == "__main__":
    _quick_test()
# sac/expert_dataset.py

import os
import logging
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class ExpertDataset(Dataset):
    """
    Expert Dataset for BC / BC+RL
    ------------------------------
    기대 형식(.npz / .npz-like):

        obs      : (K, obs_dim)
        act      : (K, act_dim)           # optional (없을 수도 있음)
        nobs     : (K, obs_dim) optional
        rew      : (K,)        optional
        done     : (K,)        optional

    - 최소 requirement: 'obs'는 반드시 있어야 함.
    - 'act'가 없는 경우, BC pretrain은 못 하고,
      나중에 다른 용도로만 쓸 수 있음.
    """

    def __init__(
        self,
        path: str,
        require_action: bool = True,
        mmap_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.path = os.path.abspath(path)
        self.require_action = require_action

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"[ExpertDataset] file not found: {self.path}")

        logger.info(f"[ExpertDataset] Loading expert data from: {self.path}")
        data = np.load(self.path, allow_pickle=False, mmap_mode=mmap_mode)

        if "obs" not in data:
            raise KeyError(
                f"[ExpertDataset] 'obs' key not found in {self.path}"
            )

        self.obs = data["obs"]
        self.act = data["act"] if "act" in data else None
        self.nobs = data["nobs"] if "nobs" in data else None
        self.rew = data["rew"] if "rew" in data else None
        self.done = data["done"] if "done" in data else None

        if self.require_action and self.act is None:
            raise KeyError(
                f"[ExpertDataset] require_action=True 이지만 "
                f"'act' key가 {self.path}에 없습니다."
            )

        self.length = self.obs.shape[0]

        logger.info(
            f"[ExpertDataset] Loaded: len={self.length}, "
            f"has_act={self.act is not None}, "
            f"has_nobs={self.nobs is not None}, "
            f"has_rew={self.rew is not None}, "
            f"has_done={self.done is not None}"
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32),
        }
        if self.act is not None:
            item["act"] = torch.as_tensor(self.act[idx], dtype=torch.float32)
        if self.nobs is not None:
            item["nobs"] = torch.as_tensor(self.nobs[idx], dtype=torch.float32)
        if self.rew is not None:
            item["rew"] = torch.as_tensor(self.rew[idx], dtype=torch.float32)
        if self.done is not None:
            item["done"] = torch.as_tensor(self.done[idx], dtype=torch.float32)
        return item
