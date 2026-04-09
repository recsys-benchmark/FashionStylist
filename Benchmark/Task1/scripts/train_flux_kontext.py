#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from task1_image_edit.io import project_root
from task1_image_edit.raw_dataset import prepare_split_manifest
from task1_image_edit.trainers.common import build_training_env, ensure_diffusers_script, run_command


def ensure_compatible_datasets_version() -> None:
    try:
        installed_version = version("datasets")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Missing `datasets` package. Install training dependencies first, for example "
            "`pip install -r requirements/base.txt -r requirements/train.txt`."
        ) from exc

    try:
        major_version = int(installed_version.split(".", 1)[0])
    except ValueError:
        return

    if major_version >= 4:
        raise RuntimeError(
            f"Incompatible `datasets` version {installed_version}. "
            "This training entrypoint passes a local dataset script to diffusers, but `datasets>=4` "
            "removed support for dataset scripts. Please install `datasets<4`, for example "
            "`pip install 'datasets>=3,<4'`."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Launch official diffusers FLUX Kontext LoRA training.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Training manifest path (.jsonl or bid->pid .npy index resolved by task1_image_edit.io.load_manifest).",
    )
    parser.add_argument("--data-root", default=None, help="Raw dataset root containing the 3 category folders under data/.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-ratio", default="7:1:2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--diffusers-root",
        default=os.environ.get("DIFFUSERS_DIR", str(project_root() / "external" / "diffusers")),
    )
    parser.add_argument("--pretrained-model-name-or-path", default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--accelerate-config", default=None)
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr-warmup-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-image", default=None)
    parser.add_argument("--validation-prompt", default=None)
    parser.add_argument("--num-validation-images", type=int, default=2)
    parser.add_argument("--aspect-ratio-buckets", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--cache-latents", action="store_true")
    parser.add_argument("--use-8bit-adam", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if bool(args.manifest) == bool(args.data_root):
        raise ValueError("Specify exactly one of --manifest or --data-root")
    return args


def main():
    ensure_compatible_datasets_version()
    args = parse_args()
    manifest_path = args.manifest
    if args.data_root:
        manifest_path = prepare_split_manifest(
            data_root=args.data_root,
            output_dir=Path(args.output_dir).resolve() / "_prepared_manifests",
            split=args.split,
            split_ratio=args.split_ratio,
            seed=args.split_seed,
        )
    script_path = ensure_diffusers_script(
        args.diffusers_root,
        "examples/dreambooth/train_dreambooth_lora_flux_kontext.py",
    )
    dataset_script = project_root() / "hf_datasets" / "flux_kontext_manifest" / "flux_kontext_manifest.py"
    command = ["accelerate", "launch"]
    if args.accelerate_config:
        command.extend(["--config_file", args.accelerate_config])
    command.extend(
        [
            str(script_path),
            "--pretrained_model_name_or_path",
            args.pretrained_model_name_or_path,
            "--dataset_name",
            str(dataset_script),
            "--image_column",
            "target_image",
            "--cond_image_column",
            "source_image",
            "--caption_column",
            "flux_prompt",
            "--output_dir",
            str(Path(args.output_dir).resolve()),
            "--mixed_precision",
            args.mixed_precision,
            "--resolution",
            str(args.resolution),
            "--train_batch_size",
            str(args.train_batch_size),
            "--guidance_scale",
            "1",
            "--gradient_accumulation_steps",
            str(args.gradient_accumulation_steps),
            "--optimizer",
            "adamw",
            "--learning_rate",
            str(args.learning_rate),
            "--lr_scheduler",
            "constant",
            "--lr_warmup_steps",
            str(args.lr_warmup_steps),
            "--max_train_steps",
            str(args.max_train_steps),
            "--rank",
            str(args.rank),
            "--seed",
            str(args.seed),
        ]
    )
    if args.gradient_checkpointing:
        command.append("--gradient_checkpointing")
    if args.cache_latents:
        command.append("--cache_latents")
    if args.use_8bit_adam:
        command.append("--use_8bit_adam")
    if args.validation_image:
        command.extend(["--validation_image", args.validation_image])
    if args.validation_prompt:
        command.extend(["--validation_prompt", args.validation_prompt])
        command.extend(["--num_validation_images", str(args.num_validation_images)])
    if args.aspect_ratio_buckets:
        command.extend(["--aspect_ratio_buckets", args.aspect_ratio_buckets])

    env = build_training_env(manifest_path)
    env["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    run_command(command, env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
