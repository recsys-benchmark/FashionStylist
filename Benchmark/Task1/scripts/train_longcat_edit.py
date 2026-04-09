#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from task1_image_edit.datasets.longcat_edit import export_manifest_to_longcat_edit
from task1_image_edit.io import project_root
from task1_image_edit.raw_dataset import prepare_split_manifest
from task1_image_edit.trainers.common import build_training_env, run_command


def ensure_longcat_path(longcat_root: str | Path, relative_path: str) -> Path:
    root = Path(longcat_root).resolve()
    target = root / relative_path
    if not target.exists():
        raise FileNotFoundError(f"Missing LongCat-Image file: {target}")
    return target


def resolve_local_path(path_like: str | None) -> str | None:
    if path_like in (None, "", "latest"):
        return path_like
    candidate = Path(path_like)
    if candidate.exists():
        return str(candidate.resolve())
    return path_like


def parse_args():
    parser = argparse.ArgumentParser(description="Launch official LongCat image-edit LoRA/SFT training.")
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
        "--longcat-root",
        default=os.environ.get("LONGCAT_DIR", str(project_root() / "external" / "LongCat-Image")),
    )
    parser.add_argument("--pretrained-model-name-or-path", default="meituan-longcat/LongCat-Image-Edit")
    parser.add_argument("--training-mode", choices=["lora"], default="lora")
    parser.add_argument("--accelerate-config", default=None)
    parser.add_argument("--diffusion-pretrain-weight", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--aspect-ratio-type", choices=["mar_256", "mar_512", "mar_1024"], default=None)
    parser.add_argument("--null-text-ratio", type=float, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--adam-weight-decay", type=float, default=None)
    parser.add_argument("--adam-epsilon", type=float, default=None)
    parser.add_argument("--adam-beta1", type=float, default=None)
    parser.add_argument("--adam-beta2", type=float, default=None)
    parser.add_argument("--lr-num-cycles", type=int, default=None)
    parser.add_argument("--lr-power", type=float, default=None)
    parser.add_argument("--lr-scheduler", default=None)
    parser.add_argument("--lr-warmup-steps", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ema-rate", type=float, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--save-model-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--use-8bit-adam", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if bool(args.manifest) == bool(args.data_root):
        raise ValueError("Specify exactly one of --manifest or --data-root")
    return args


def load_base_config(longcat_root: str | Path, training_mode: str) -> tuple[Path, dict]:
    config_path = ensure_longcat_path(longcat_root, "train_examples/edit_lora/train_config.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected dict config in {config_path}, got {type(config)!r}")
    return config_path, config


def main():
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
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "longcat_edit_train_data.txt"
    export_manifest_to_longcat_edit(manifest_path, metadata_path)

    _, config = load_base_config(args.longcat_root, args.training_mode)
    config_updates = {
        "data_txt_root": str(metadata_path),
        "pretrained_model_name_or_path": resolve_local_path(args.pretrained_model_name_or_path),
        "work_dir": str(output_dir),
        "diffusion_pretrain_weight": resolve_local_path(args.diffusion_pretrain_weight),
        "resolution": args.resolution,
        "aspect_ratio_type": args.aspect_ratio_type,
        "null_text_ratio": args.null_text_ratio,
        "dataloader_num_workers": args.dataloader_num_workers,
        "train_batch_size": args.train_batch_size,
        "repeats": args.repeats,
        "mixed_precision": args.mixed_precision,
        "max_train_steps": args.max_train_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_clip": args.gradient_clip,
        "learning_rate": args.learning_rate,
        "adam_weight_decay": args.adam_weight_decay,
        "adam_epsilon": args.adam_epsilon,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "lr_num_cycles": args.lr_num_cycles,
        "lr_power": args.lr_power,
        "lr_scheduler": args.lr_scheduler,
        "lr_warmup_steps": args.lr_warmup_steps,
        "use_ema": args.use_ema,
        "ema_rate": args.ema_rate,
        "resume_from_checkpoint": resolve_local_path(args.resume_from_checkpoint),
        "log_interval": args.log_interval,
        "save_model_steps": args.save_model_steps,
        "seed": args.seed,
    }
    if args.training_mode == "lora":
        config_updates["lora_rank"] = args.lora_rank
    for key, value in config_updates.items():
        if value is not None:
            config[key] = value

    generated_config_path = output_dir / f"longcat_edit_{args.training_mode}_config.yaml"
    with generated_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    train_script = ensure_longcat_path(args.longcat_root, "train_examples/edit_lora/train_edit_lora.py")

    command = ["accelerate", "launch"]
    if args.accelerate_config:
        command.extend(["--config_file", args.accelerate_config])
    if config.get("mixed_precision"):
        command.extend(["--mixed_precision", str(config["mixed_precision"])])
    command.extend([str(train_script), "--config", str(generated_config_path)])
    if args.report_to:
        command.extend(["--report_to", args.report_to])
    if args.allow_tf32:
        command.append("--allow_tf32")
    if args.use_8bit_adam:
        command.append("--use_8bit_adam")

    env = build_training_env(manifest_path)
    longcat_root = str(Path(args.longcat_root).resolve())
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = longcat_root if not current_pythonpath else f"{longcat_root}:{current_pythonpath}"
    env.setdefault("TOKENIZERS_PARALLELISM", "False")
    env.setdefault("NCCL_DEBUG", "INFO")
    env.setdefault("NCCL_TIMEOUT", "12000")
    run_command(command, env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
