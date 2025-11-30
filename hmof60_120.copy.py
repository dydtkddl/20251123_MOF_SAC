#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import logging
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

##############################################
# Logging
##############################################
logger = logging.getLogger("move_mofs")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler("move_mof_files.log", maxBytes=5_000_000, backupCount=3)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("===== MOF file copy job started =====")

##############################################
# Paths
##############################################
txt_path = "filtered_mofs.60_120.hmof.txt"

SRC_DIR = "/home/yongsang/PSID_SIMULATION_TOOLS/RASPA/share/raspa/structures/cif/HMOF/reduced_WLLFHHS_hMOF/GA_MOFs"
DST_DIR = "/home/yongsang/20251123_MOF_SAC/mofs/train_pool.hmof"

logger.info(f"Source directory: {SRC_DIR}")
logger.info(f"Destination directory: {DST_DIR}")

# 대상 폴더 없으면 생성
os.makedirs(DST_DIR, exist_ok=True)

##############################################
# Load TXT
##############################################
logger.info(f"Reading MOF list from {txt_path}")

with open(txt_path, "r") as f:
    mof_list = [line.strip() for line in f.readlines() if line.strip()]

logger.info(f"Total MOFs to process: {len(mof_list)}")

##############################################
# Copy files
##############################################
missing_files = []

for fname in tqdm(mof_list, desc="Copying MOF files"):
    src = os.path.join(SRC_DIR, fname)
    dst = os.path.join(DST_DIR, fname)

    if not os.path.exists(src):
        logger.warning(f"Missing file: {src}")
        missing_files.append(fname)
        continue

    shutil.copy2(src, dst)  # 타임스탬프까지 그대로
    logger.info(f"Copied {fname}")

##############################################
# Summary
##############################################
logger.info("===== Copy job completed =====")
logger.info(f"Missing files: {len(missing_files)}")

if missing_files:
    logger.info("Missing file list:")
    for mf in missing_files:
        logger.info(f" - {mf}")

print(f"✔ 작업 완료. 복사 실패 파일 수: {len(missing_files)}")

