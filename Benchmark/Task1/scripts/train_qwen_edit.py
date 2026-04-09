#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from task1_image_edit.datasets.qwen_edit_diffsynth import export_manifest_to_diffsynth_qwen_edit
from task1_image_edit.io import project_root
from task1_image_edit.raw_dataset import prepare_split_manifest
from task1_image_edit.trainers.common import build_training_env, ensure_diffsynth_script, run_command


DEFAULT_LORA_TARGET_MODULES = (
    "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,"
    "img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
)
QWEN_IMAGE_REPO_ID = "Qwen/Qwen-Image"


def parse_args():
    parser = argparse.ArgumentParser(description="Launch DiffSynth-Studio Qwen-Image-Edit training.")
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
        "--diffsynth-root",
        default=os.environ.get("DIFFSYNTH_DIR", str(project_root() / "external" / "DiffSynth-Studio")),
    )
    parser.add_argument("--pretrained-model-name-or-path", default="Qwen/Qwen-Image-Edit")
    parser.add_argument("--training-mode", choices=["lora"], default="lora")
    parser.add_argument("--accelerate-config", default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--dataset-repeat", type=int, default=50)
    parser.add_argument("--dataset-num-workers", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--use-gradient-checkpointing", action="store_true")
    parser.add_argument("--find-unused-parameters", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-target-modules", default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument("--zero-cond-t", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if bool(args.manifest) == bool(args.data_root):
        raise ValueError("Specify exactly one of --manifest or --data-root")
    return args


def resolve_local_path(path_like: str | None) -> str | None:
    if path_like in (None, "", "latest"):
        return path_like
    candidate = Path(path_like).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return path_like


def local_component_exists(local_root: Path | None, relative_path: str) -> bool:
    if local_root is None:
        return False
    normalized = relative_path.rstrip("/")
    if "*" in normalized or "?" in normalized or "[" in normalized:
        return any(local_root.glob(normalized))
    return (local_root / normalized).exists()


def build_remote_model_id_with_origin_paths(model_name_or_path: str) -> str:
    resolved_model_name_or_path = resolve_local_path(model_name_or_path) or model_name_or_path
    local_root = Path(resolved_model_name_or_path) if Path(resolved_model_name_or_path).exists() else None
    text_encoder_source = resolved_model_name_or_path if local_component_exists(local_root, "text_encoder/model*.safetensors") else QWEN_IMAGE_REPO_ID
    vae_source = resolved_model_name_or_path if local_component_exists(local_root, "vae/diffusion_pytorch_model.safetensors") else QWEN_IMAGE_REPO_ID
    return (
        f"{resolved_model_name_or_path}:transformer/diffusion_pytorch_model*.safetensors,"
        f"{text_encoder_source}:text_encoder/model*.safetensors,"
        f"{vae_source}:vae/diffusion_pytorch_model.safetensors"
    )


def list_local_component_paths(local_root: Path, relative_pattern: str) -> list[str]:
    return [str(path.resolve()) for path in sorted(local_root.glob(relative_pattern)) if path.is_file()]


def pack_component_paths(paths: list[str]) -> str | list[str] | None:
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    return paths


def build_model_loading_config(model_name_or_path: str) -> tuple[list[str | list[str]], list[str], str | None, str | None]:
    resolved_model_name_or_path = resolve_local_path(model_name_or_path) or model_name_or_path
    local_root = Path(resolved_model_name_or_path) if Path(resolved_model_name_or_path).exists() else None
    if local_root is None:
        return [], [build_remote_model_id_with_origin_paths(resolved_model_name_or_path)], None, None

    model_paths: list[str | list[str]] = []
    remote_model_ids: list[str] = []

    transformer_paths = list_local_component_paths(local_root, "transformer/diffusion_pytorch_model*.safetensors")
    if not transformer_paths:
        raise FileNotFoundError(
            f"Local pretrained model directory exists but transformer weights were not found under "
            f"`{local_root / 'transformer'}`."
        )
    transformer_component = pack_component_paths(transformer_paths)
    if transformer_component is not None:
        model_paths.append(transformer_component)

    text_encoder_paths = list_local_component_paths(local_root, "text_encoder/model*.safetensors")
    if text_encoder_paths:
        text_encoder_component = pack_component_paths(text_encoder_paths)
        if text_encoder_component is not None:
            model_paths.append(text_encoder_component)
    else:
        remote_model_ids.append(f"{QWEN_IMAGE_REPO_ID}:text_encoder/model*.safetensors")

    vae_paths = list_local_component_paths(local_root, "vae/diffusion_pytorch_model.safetensors")
    if vae_paths:
        vae_component = pack_component_paths(vae_paths)
        if vae_component is not None:
            model_paths.append(vae_component)
    else:
        remote_model_ids.append(f"{QWEN_IMAGE_REPO_ID}:vae/diffusion_pytorch_model.safetensors")

    tokenizer_path = str((local_root / "tokenizer").resolve()) if (local_root / "tokenizer").exists() else None
    processor_path = str((local_root / "processor").resolve()) if (local_root / "processor").exists() else None
    return model_paths, remote_model_ids, tokenizer_path, processor_path


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
    diffsynth_root = Path(args.diffsynth_root).resolve()
    train_script = ensure_diffsynth_script(
        diffsynth_root,
        "examples/qwen_image/model_training/train.py",
    )
    metadata_path = Path(args.output_dir).resolve() / "diffsynth_metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    export_manifest_to_diffsynth_qwen_edit(manifest_path, metadata_path)
    model_paths, remote_model_ids, tokenizer_path, processor_path = build_model_loading_config(
        args.pretrained_model_name_or_path
    )

    command = ["accelerate", "launch"]
    if args.accelerate_config:
        command.extend(["--config_file", args.accelerate_config])
    if args.gradient_accumulation_steps > 1:
        # DiffSynth consumes this through the script parser, not the accelerate launcher.
        pass

    command.extend(
        [
            str(train_script),
            "--dataset_base_path",
            "",
            "--dataset_metadata_path",
            str(metadata_path),
            "--data_file_keys",
            "image,edit_image",
            "--extra_inputs",
            "edit_image",
            "--max_pixels",
            str(args.max_pixels),
            "--dataset_repeat",
            str(args.dataset_repeat),
            "--dataset_num_workers",
            str(args.dataset_num_workers),
            "--learning_rate",
            str(args.learning_rate),
            "--num_epochs",
            str(args.num_epochs),
            "--gradient_accumulation_steps",
            str(args.gradient_accumulation_steps),
            "--remove_prefix_in_ckpt",
            "pipe.dit.",
            "--output_path",
            str(Path(args.output_dir).resolve()),
        ]
    )
    if model_paths:
        command.extend(["--model_paths", json.dumps(model_paths)])
    if remote_model_ids:
        command.extend(["--model_id_with_origin_paths", ",".join(remote_model_ids)])
    if tokenizer_path:
        command.extend(["--tokenizer_path", tokenizer_path])
    if processor_path:
        command.extend(["--processor_path", processor_path])
    command.extend(
        [
            "--lora_base_model",
            "dit",
            "--lora_target_modules",
            args.lora_target_modules,
            "--lora_rank",
            str(args.lora_rank),
        ]
    )
    if args.use_gradient_checkpointing:
        command.append("--use_gradient_checkpointing")
    if args.find_unused_parameters:
        command.append("--find_unused_parameters")
    if args.zero_cond_t or "2511" in args.pretrained_model_name_or_path:
        command.append("--zero_cond_t")

    env = build_training_env(manifest_path, extra_pythonpaths=[diffsynth_root])
    env["DIFFSYNTH_DIR"] = str(diffsynth_root)
    run_command(command, env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
