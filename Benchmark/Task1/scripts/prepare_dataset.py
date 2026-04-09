#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from task1_image_edit.raw_dataset import prepare_split_manifests


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare train/val/test manifests from raw outfit data folders.")
    parser.add_argument("--data-root", required=True, help="Raw dataset root containing the 3 category folders under data/.")
    parser.add_argument("--output-dir", required=True, help="Directory where train/val/test JSONL manifests will be written.")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-ratio", default="7:1:2")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_paths = prepare_split_manifests(
        data_root=args.data_root,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
        seed=args.split_seed,
    )
    for split_name, path in manifest_paths.items():
        print(f"{split_name}\t{path}")


if __name__ == "__main__":
    main()
