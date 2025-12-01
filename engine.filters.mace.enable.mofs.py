#!/usr/bin/env python3
import os
import shutil
from tqdm import tqdm
from ase.io import read
from ase.data import chemical_symbols
from mace.calculators import MACECalculator
import os
import os

# ============================================
# ★ 절대적으로 필요한 스레드 제한 (모든 백엔드 포함)
# ============================================

POOL_DIR = "/home/yongsang/20251123_MOF_SAC/mofs/train_pool.coremof"
VALID_DIR = "mofs/train_pool_valid.coremof"
INVALID_DIR = "mofs/train_pool_invalid.coremof"

os.makedirs(VALID_DIR, exist_ok=True)
os.makedirs(INVALID_DIR, exist_ok=True)


# -------------------
# Init MACE
# -------------------
calc = MACECalculator(
    model_paths=["mofs_v2.model"],
    head="pbe_d3",
    device="cpu",
    default_dtype="float64"
)

try:
    SUPPORTED_Z = list(calc.model.atomic_numbers)
except:
    SUPPORTED_Z = list(calc.models[0].atomic_numbers)

SUPPORTED_Z = set([int(z) for z in SUPPORTED_Z])


print("Supported:", sorted(SUPPORTED_Z))
print("Total CIF scanning...\n")


# -------------------
# scanning
# -------------------
files = [f for f in os.listdir(POOL_DIR) if f.endswith(".cif")]

valid_cnt = 0
invalid_cnt = 0

for fn in tqdm(files):

    src = os.path.join(POOL_DIR, fn)

    try:
        atoms = read(src)
    except:
        shutil.move(src, os.path.join(INVALID_DIR, fn))
        invalid_cnt += 1
        continue

    zs = atoms.get_atomic_numbers()

    if any(z not in SUPPORTED_Z for z in zs):
        shutil.move(src, os.path.join(INVALID_DIR, fn))
        invalid_cnt += 1
        continue

    shutil.move(src, os.path.join(VALID_DIR, fn))
    valid_cnt += 1



print("\n===================================")
print("DONE")
print(f"VALID   : {valid_cnt}")
print(f"INVALID : {invalid_cnt}")
print("===================================")

