from __future__ import annotations

import os

import datasets

from task1_image_edit.io import load_manifest, prompt_context_from_manifest_row
from task1_image_edit.prompts import build_training_prompt


class FluxKontextManifest(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "sample_id": datasets.Value("string"),
                    "source_image": datasets.Image(),
                    "target_image": datasets.Image(),
                    "flux_prompt": datasets.Value("string"),
                    "items": datasets.Sequence(datasets.Value("string")),
                    "outfit_summary": datasets.Value("string"),
                }
            )
        )

    def _split_generators(self, dl_manager):
        manifest_path = os.environ.get("GARMENT_DATASET_MANIFEST")
        if not manifest_path:
            raise ValueError("GARMENT_DATASET_MANIFEST is not set")
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"manifest_path": manifest_path})]

    def _generate_examples(self, manifest_path: str):
        for idx, row in enumerate(load_manifest(manifest_path)):
            prompt_context = prompt_context_from_manifest_row(row)
            prompt = row.get("flux_prompt") or build_training_prompt(
                "flux_kontext",
                items=prompt_context["items"],
                outfit_summary=prompt_context["outfit_summary"],
                extra_constraints=prompt_context["extra_constraints"],
            )
            yield idx, {
                "sample_id": str(row["sample_id"]),
                "source_image": row["source_image"],
                "target_image": row["target_image"],
                "flux_prompt": prompt,
                "items": row.get("items", []),
                "outfit_summary": row.get("outfit_summary", "") or "",
            }
