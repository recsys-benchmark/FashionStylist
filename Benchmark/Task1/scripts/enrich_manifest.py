#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from task1_image_edit.io import load_manifest, prompt_context_from_manifest_row, write_jsonl
from task1_image_edit.prompts import build_prompt_bundle


def parse_args():
    parser = argparse.ArgumentParser(description="Populate model-specific prompt fields in a training manifest.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input manifest path (.jsonl or bid->pid .npy index resolved by task1_image_edit.io.load_manifest).",
    )
    parser.add_argument("--output", required=True, help="Output manifest JSONL.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_manifest(args.input)
    enriched = []
    for row in rows:
        prompt_context = prompt_context_from_manifest_row(row)
        bundle = build_prompt_bundle(
            items=prompt_context["items"],
            outfit_summary=prompt_context["outfit_summary"],
            extra_constraints=prompt_context["extra_constraints"],
        )
        merged = dict(row)
        defaults = {
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
        for key, value in defaults.items():
            if args.overwrite or not merged.get(key):
                merged[key] = value
        enriched.append(merged)
    write_jsonl(args.output, enriched)


if __name__ == "__main__":
    main()
