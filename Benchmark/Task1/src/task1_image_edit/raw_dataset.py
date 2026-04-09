from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from task1_image_edit.io import write_jsonl
from task1_image_edit.prompts import build_prompt_bundle


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
SPLIT_NAMES = ("train", "val", "test")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw_row in reader:
            row = {}
            for key, value in (raw_row or {}).items():
                normalized_key = (key or "").strip()
                row[normalized_key] = (value or "").strip()
            rows.append(row)
    return rows


def _normalize_photo_list(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    normalized = raw_value.replace("，", ",")
    return [chunk.strip() for chunk in normalized.split(",") if chunk.strip()]


def _resolve_image_path(photo_dir: Path, image_id: str) -> Path:
    candidate = photo_dir / image_id
    if candidate.exists():
        return candidate.resolve()
    for suffix in IMAGE_SUFFIXES:
        candidate = photo_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate.resolve()
    normalized_image_id = image_id.strip().lower()
    if normalized_image_id:
        for candidate in photo_dir.iterdir():
            if not candidate.is_file():
                continue
            suffix = candidate.suffix.lower()
            if suffix and suffix not in IMAGE_SUFFIXES:
                continue
            stem = candidate.stem.lower() if suffix else candidate.name.lower()
            if stem == normalized_image_id:
                return candidate.resolve()
    raise FileNotFoundError(f"Could not find image '{image_id}' under {photo_dir}")


def _load_bid_pid_index(path: Path) -> dict[str, list[str]]:
    raw = np.load(path, allow_pickle=True)
    data = raw.item() if isinstance(raw, np.ndarray) else raw
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict-like bid_pid index in {path}, got {type(data)!r}")
    index: dict[str, list[str]] = {}
    for bid, pids in data.items():
        normalized_bid = str(bid).strip()
        if not normalized_bid:
            continue
        normalized_pids = [str(pid).strip() for pid in pids if str(pid).strip()]
        index[normalized_bid] = normalized_pids
    return index


def _parse_split_ratio(split_ratio: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in split_ratio.split(":")]
    if len(parts) != 3:
        raise ValueError(f"Expected split ratio like '7:1:2', got {split_ratio!r}")
    try:
        values = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"Split ratio must contain integers, got {split_ratio!r}") from exc
    if any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError(f"Split ratio must be non-negative with positive sum, got {split_ratio!r}")
    return values


def _allocate_split_counts(total: int, ratios: tuple[int, int, int]) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    exact = [total * ratio / sum(ratios) for ratio in ratios]
    counts = [int(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda idx: (exact[idx] - counts[idx], ratios[idx], -idx),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts[0], counts[1], counts[2]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return slug.strip("_") or "item"


def _build_prompt_fields(row: dict[str, Any]) -> dict[str, Any]:
    bundle = build_prompt_bundle(
        items=row.get("items"),
        outfit_summary=row.get("outfit_summary"),
        extra_constraints=row.get("extra_constraints"),
    )
    enriched = dict(row)
    enriched.update(
        {
            "qwen_prompt": bundle.qwen_edit_infer,
            "qwen_train_prompt": bundle.qwen_sft,
            "qwen_negative_prompt": bundle.qwen_edit_negative,
            "negative_prompt": bundle.qwen_edit_negative,
            "longcat_prompt": bundle.longcat_edit_infer,
            "longcat_train_prompt": bundle.longcat_sft,
            "longcat_sft_prompt": bundle.longcat_sft,
            "longcat_negative_prompt": bundle.longcat_edit_negative,
            "flux_prompt": bundle.flux_kontext_infer,
            "qwen_sft_prompt": bundle.qwen_sft,
        }
    )
    return enriched


def collect_raw_dataset_rows(data_root: str | Path) -> list[dict[str, Any]]:
    root = Path(data_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    rows: list[dict[str, Any]] = []
    for subset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label_path = subset_dir / "label.csv"
        look_path = subset_dir / "look.csv"
        bid_pid_path = subset_dir / "bid_pid_dict.npy"
        photo_dir = subset_dir / "photos"
        if not (label_path.exists() and look_path.exists() and bid_pid_path.exists() and photo_dir.exists()):
            continue

        label_index = {
            row["photo"]: row for row in _read_csv_rows(label_path) if row.get("photo")
        }
        look_index = {
            row["bandle"]: row for row in _read_csv_rows(look_path) if row.get("bandle")
        }
        bid_pid_index = _load_bid_pid_index(bid_pid_path)

        for bid in sorted(bid_pid_index):
            look_row = look_index.get(bid, {})
            source_image = str(_resolve_image_path(photo_dir, bid))
            outfit_id = f"{subset_dir.name}_{bid}"
            look_photo_ids = _normalize_photo_list(look_row.get("photos"))
            expected_pids = bid_pid_index[bid]
            expected_pid_set = set(expected_pids)
            ordered_pids = []
            seen_pids = set()
            for pid in look_photo_ids + expected_pids:
                if pid not in expected_pid_set or pid in seen_pids:
                    continue
                ordered_pids.append(pid)
                seen_pids.add(pid)

            category_counts: dict[str, int] = defaultdict(int)
            for pid in ordered_pids:
                label_row = label_index.get(pid)
                if not label_row:
                    raise KeyError(f"Missing label row for pid '{pid}' in {label_path}")
                raw_category = label_row.get("category") or "garment"
                normalized_category = raw_category.strip().lower()
                category_counts[normalized_category] += 1
                filename = f"{category_counts[normalized_category]:02d}_{_slugify(normalized_category)}_{pid}.png"

                rows.append(
                    _build_prompt_fields(
                        {
                            "sample_id": f"{outfit_id}_{pid}",
                            "outfit_id": outfit_id,
                            "source_subset": subset_dir.name,
                            "bid": bid,
                            "pid": pid,
                            "category": normalized_category,
                            "source_image": source_image,
                            "target_image": str(_resolve_image_path(photo_dir, pid)),
                            "items": [normalized_category],
                            "outfit_summary": "",
                            "extra_constraints": "",
                            "output_subpath": f"{outfit_id}/{filename}",
                        }
                    )
                )
    if not rows:
        raise ValueError(f"No dataset subsets found under {root}")
    return rows


def split_raw_dataset_rows(
    rows: list[dict[str, Any]],
    split_ratio: str = "7:1:2",
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    ratios = _parse_split_ratio(split_ratio)
    rows_by_subset_and_outfit: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        subset = str(row["source_subset"])
        outfit_id = str(row["outfit_id"])
        rows_by_subset_and_outfit[subset][outfit_id].append(row)

    split_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for subset in sorted(rows_by_subset_and_outfit):
        outfit_ids = sorted(rows_by_subset_and_outfit[subset])
        rng = random.Random(f"{seed}:{subset}")
        rng.shuffle(outfit_ids)
        train_count, val_count, test_count = _allocate_split_counts(len(outfit_ids), ratios)
        boundaries = {
            "train": outfit_ids[:train_count],
            "val": outfit_ids[train_count : train_count + val_count],
            "test": outfit_ids[train_count + val_count : train_count + val_count + test_count],
        }
        for split_name, split_outfit_ids in boundaries.items():
            for outfit_id in split_outfit_ids:
                for row in rows_by_subset_and_outfit[subset][outfit_id]:
                    split_row = dict(row)
                    split_row["split"] = split_name
                    split_rows[split_name].append(split_row)
    return split_rows


def prepare_split_manifests(
    data_root: str | Path,
    output_dir: str | Path,
    split_ratio: str = "7:1:2",
    seed: int = 42,
) -> dict[str, str]:
    manifest_dir = Path(output_dir).resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    split_rows = split_raw_dataset_rows(
        collect_raw_dataset_rows(data_root),
        split_ratio=split_ratio,
        seed=seed,
    )
    manifest_paths: dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        output_path = manifest_dir / f"{split_name}.jsonl"
        ordered_rows = sorted(
            split_rows[split_name],
            key=lambda row: (str(row["source_subset"]), str(row["bid"]), str(row["pid"])),
        )
        write_jsonl(output_path, ordered_rows)
        manifest_paths[split_name] = str(output_path)
    return manifest_paths


def prepare_split_manifest(
    data_root: str | Path,
    output_dir: str | Path,
    split: str,
    split_ratio: str = "7:1:2",
    seed: int = 42,
) -> str:
    normalized_split = split.strip().lower()
    if normalized_split not in SPLIT_NAMES:
        raise ValueError(f"Unsupported split '{split}'. Expected one of {', '.join(SPLIT_NAMES)}")
    return prepare_split_manifests(
        data_root=data_root,
        output_dir=output_dir,
        split_ratio=split_ratio,
        seed=seed,
    )[normalized_split]
