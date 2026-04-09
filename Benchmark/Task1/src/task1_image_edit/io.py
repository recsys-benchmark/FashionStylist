from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path).resolve()
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {manifest_path}") from exc
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_items(items: Any) -> list[str]:
    if items is None:
        return []
    if isinstance(items, list):
        return [str(item).strip() for item in items if str(item).strip()]
    if isinstance(items, str):
        return [chunk.strip() for chunk in items.split(",") if chunk.strip()]
    return [str(items).strip()]


def resolve_path(path_like: str, manifest_path: str | Path) -> str:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return str(candidate)
    return str((Path(manifest_path).resolve().parent / candidate).resolve())


def resolve_manifest_row(row: dict[str, Any], manifest_path: str | Path) -> dict[str, Any]:
    resolved = dict(row)
    for key in ("source_image", "target_image"):
        if key in resolved and resolved[key]:
            resolved[key] = resolve_path(str(resolved[key]), manifest_path)
    resolved["items"] = _normalize_items(resolved.get("items"))
    if "sample_id" not in resolved or not str(resolved["sample_id"]).strip():
        stem = Path(resolved.get("source_image") or resolved.get("target_image") or "sample").stem
        resolved["sample_id"] = stem
    return resolved


def prompt_context_from_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    items = _normalize_items(row.get("items"))
    bid = str(row.get("bid") or "").strip()
    pid = str(row.get("pid") or "").strip()
    if bid and pid and items == [pid]:
        items = []
    return {
        "items": items,
        "outfit_summary": row.get("outfit_summary"),
        "extra_constraints": row.get("extra_constraints"),
    }


def _infer_bid_pid_image_root(index_path: Path) -> Path:
    image_root_override = os.environ.get("GARMENT_DATASET_IMAGE_DIR")
    if image_root_override:
        return Path(image_root_override).resolve()

    candidate_dirs = sorted(path for path in index_path.parent.iterdir() if path.is_dir() and path.name.startswith("photos"))
    if len(candidate_dirs) == 1:
        return candidate_dirs[0].resolve()
    if candidate_dirs:
        raise ValueError(
            f"Multiple candidate image directories found next to {index_path}: "
            + ", ".join(str(path) for path in candidate_dirs)
            + ". Please set GARMENT_DATASET_IMAGE_DIR explicitly."
        )
    raise ValueError(
        f"Could not infer image directory for {index_path}. "
        "Please place the image folder next to the index file or set GARMENT_DATASET_IMAGE_DIR."
    )


def _resolve_image_path(image_root: Path, image_id: str) -> Path:
    candidate = image_root / image_id
    if candidate.exists():
        return candidate.resolve()

    for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidate = image_root / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not find image for id '{image_id}' under {image_root}")


def _load_bid_pid_manifest(path: str | Path) -> list[dict[str, Any]]:
    index_path = Path(path).resolve()
    raw = np.load(index_path, allow_pickle=True)
    data = raw.item() if isinstance(raw, np.ndarray) else raw
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict-like .npy index at {index_path}, but got {type(data)!r}")

    image_root = _infer_bid_pid_image_root(index_path)
    rows = []
    for bid, pids in data.items():
        source_id = str(bid).strip()
        if not source_id:
            continue
        source_image = str(_resolve_image_path(image_root, source_id))
        for pid in _normalize_items(pids):
            rows.append(
                {
                    "sample_id": f"{source_id}_{pid}",
                    "source_image": source_image,
                    "target_image": str(_resolve_image_path(image_root, pid)),
                    "items": [pid],
                    "bid": source_id,
                    "pid": pid,
                }
            )
    return rows


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path).resolve()
    if manifest_path.suffix.lower() == ".npy":
        rows = _load_bid_pid_manifest(manifest_path)
    else:
        rows = read_jsonl(manifest_path)
    return [resolve_manifest_row(row, path) for row in rows]
