#!/usr/bin/env python3
import os
import argparse
import csv
import shutil
import logging
from tqdm import tqdm
import pandas as pd

# ---------------------------------------
# 로그 설정
# ---------------------------------------
logging.basicConfig(
    filename="mof_filter_copy.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    parser = argparse.ArgumentParser(description="Filter MOF CIFs by atom count and copy to train_pool.")
    parser.add_argument("--csv_path", type=str, default="relaxed.atom_counts.csv",
                        help="Path to atom count CSV file")
    parser.add_argument("--min_atoms", type=int, required=True, help="Minimum atom count")
    parser.add_argument("--max_atoms", type=int, required=True, help="Maximum atom count")
    parser.add_argument("--source_dir", type=str,
                        default="/home/yongsang/PSID_SIMULATION_TOOLS/RASPA/share/raspa/structures/cif/QMOF/qmof_database/relaxed_structures/",
                        help="Directory where CIF files are located")
    parser.add_argument("--target_dir", type=str, default="mofs/train_pool/",
                        help="Directory to copy selected CIFs")
    parser.add_argument("--output_csv", type=str, default="filtered_mofs.csv",
                        help="Output CSV file listing selected CIFs")

    args = parser.parse_args()

    logging.info("===== MOF Filtering & Copy Engine Started =====")
    logging.info(f"CSV: {args.csv_path}")
    logging.info(f"Atom range: {args.min_atoms} ~ {args.max_atoms}")
    logging.info(f"Source CIF dir: {args.source_dir}")
    logging.info(f"Target CIF dir: {args.target_dir}")

    # ---------------------------------------
    # CSV 로드
    # ---------------------------------------
    df = pd.read_csv(args.csv_path)
    logging.info(f"Loaded CSV with {len(df)} entries")

    # ---------------------------------------
    # atom_count로 필터링
    # ---------------------------------------
    filtered_df = df[(df["atom_count"] >= args.min_atoms) &
                     (df["atom_count"] <= args.max_atoms)]

    logging.info(f"Filtered MOFs: {len(filtered_df)}")

    # ---------------------------------------
    # 타겟 디렉토리 생성
    # ---------------------------------------
    os.makedirs(args.target_dir, exist_ok=True)

    copied_files = []
    missing_files = []

    # ---------------------------------------
    # CIF 파일 복사 (tqdm으로 진행상황 표시)
    # ---------------------------------------
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Copying CIF files"):
        cif_name = row["filename"]
        src = os.path.join(args.source_dir, cif_name)
        dst = os.path.join(args.target_dir, cif_name)

        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied_files.append(cif_name)
        else:
            logging.warning(f"Missing CIF: {cif_name}")
            missing_files.append(cif_name)

    # ---------------------------------------
    # 결과 CSV 저장
    # ---------------------------------------
    filtered_df.to_csv(args.output_csv, index=False)
    logging.info(f"Saved filtered list to {args.output_csv}")
    logging.info(f"Copied {len(copied_files)} CIF files")
    logging.info(f"Missing {len(missing_files)} CIF files")

    print(f"완료! 필터링된 {len(copied_files)}개 CIF가 train_pool에 복사됨.")
    if missing_files:
        print(f"⚠ {len(missing_files)}개 파일이 소스 폴더에서 없음 → log 파일 확인")


if __name__ == "__main__":
    main()

