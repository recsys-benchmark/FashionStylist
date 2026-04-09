from __future__ import annotations

from pathlib import Path

from task1_image_edit.io import load_manifest, prompt_context_from_manifest_row, write_jsonl
from task1_image_edit.prompts import build_training_prompt


def export_manifest_to_diffsynth_qwen_edit(
    manifest_path: str | Path,
    output_path: str | Path,
) -> str:
    rows = load_manifest(manifest_path)
    exported_rows = []
    for row in rows:
        prompt_context = prompt_context_from_manifest_row(row)
        prompt = row.get("qwen_train_prompt") or row.get("qwen_sft_prompt") or row.get("qwen_prompt") or build_training_prompt(
            "qwen_edit",
            items=prompt_context["items"],
            outfit_summary=prompt_context["outfit_summary"],
            extra_constraints=prompt_context["extra_constraints"],
        )
        exported_rows.append(
            {
                "prompt": prompt,
                "image": row["target_image"],
                "edit_image": row["source_image"],
                "sample_id": str(row["sample_id"]),
            }
        )
    write_jsonl(output_path, exported_rows)
    return str(Path(output_path).resolve())
