from __future__ import annotations

from pathlib import Path

from PIL import Image

from task1_image_edit.io import load_manifest, prompt_context_from_manifest_row, write_jsonl
from task1_image_edit.prompts import build_training_prompt


def _resolve_image_size(row: dict) -> tuple[int, int]:
    width = row.get("width") or row.get("target_width") or row.get("image_width")
    height = row.get("height") or row.get("target_height") or row.get("image_height")
    if width and height:
        return int(width), int(height)
    with Image.open(row["target_image"]) as image:
        return image.size


def export_manifest_to_longcat_edit(
    manifest_path: str | Path,
    output_path: str | Path,
) -> str:
    rows = load_manifest(manifest_path)
    exported_rows = []
    for row in rows:
        prompt_context = prompt_context_from_manifest_row(row)
        prompt = (
            row.get("longcat_train_prompt")
            or row.get("longcat_sft_prompt")
            or row.get("longcat_prompt")
            or build_training_prompt(
                "longcat_edit_turbo",
                items=prompt_context["items"],
                outfit_summary=prompt_context["outfit_summary"],
                extra_constraints=prompt_context["extra_constraints"],
            )
        )
        width, height = _resolve_image_size(row)
        exported_rows.append(
            {
                "img_path": row["target_image"],
                "ref_img_path": row["source_image"],
                "prompt": prompt,
                "width": width,
                "height": height,
                "sample_id": str(row["sample_id"]),
            }
        )
    write_jsonl(output_path, exported_rows)
    return str(Path(output_path).resolve())
