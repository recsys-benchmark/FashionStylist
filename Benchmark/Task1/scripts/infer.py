#!/usr/bin/env python3
from __future__ import annotations

import argparse
import site
import sys
from pathlib import Path


def _remove_user_site_packages() -> None:
    try:
        user_site = site.getusersitepackages()
    except Exception:
        return
    user_sites = {user_site} if isinstance(user_site, str) else set(user_site)
    sys.path[:] = [path for path in sys.path if path not in user_sites]


_remove_user_site_packages()

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from task1_image_edit.io import load_manifest, prompt_context_from_manifest_row, project_root
from task1_image_edit.models.base import InferenceRequest
from task1_image_edit.prompts import canonical_model_name, select_inference_prompt
from task1_image_edit.raw_dataset import prepare_split_manifest
from task1_image_edit.runtime import ensure_diffsynth_available


def _pick_prompt_context(cli_value, row_context, key):
    if cli_value is not None:
        return cli_value
    if row_context:
        value = row_context.get(key)
        if value not in (None, ""):
            return value
    return None


def _provided_option_names(argv: list[str]) -> set[str]:
    provided: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        if "=" in token:
            provided.add(token.split("=", 1)[0])
        else:
            provided.add(token)
    return provided


def _validate_qwen_inputs(args, provided_options: set[str]) -> None:
    unsupported = []
    if "--guidance-scale" in provided_options:
        unsupported.append("--guidance-scale")
    if "--image-guidance-scale" in provided_options:
        unsupported.append("--image-guidance-scale")
    if "--max-sequence-length" in provided_options:
        unsupported.append("--max-sequence-length")
    if unsupported:
        raise ValueError(
            "qwen_edit does not use "
            + ", ".join(unsupported)
            + ". Use --true-cfg-scale to control Qwen guidance."
        )
    if args.manifest and Path(args.manifest).suffix.lower() == ".npy":
        raise ValueError(
            "qwen_edit does not support --manifest with a bid->pid .npy index because that path lacks "
            "the category prompt context Qwen inference expects. Use --data-root to auto-build a JSONL "
            "manifest or pass an enriched .jsonl manifest instead."
        )


def build_runner(args):
    canonical = canonical_model_name(args.model)
    if canonical == "qwen_edit":
        ensure_diffsynth_available(args.diffsynth_root)
        try:
            from task1_image_edit.models.qwen_edit import QwenEditRunner
        except ModuleNotFoundError as exc:
            if exc.name == "diffsynth":
                raise
            raise
        return QwenEditRunner(
            model_name_or_path=args.model_name_or_path,
            device=args.device,
            dtype=args.dtype,
            lora_path=args.lora_path,
            offload=args.offload,
            low_vram=args.low_vram,
            zero_cond_t=args.zero_cond_t,
        )
    if canonical == "longcat_edit_turbo":
        try:
            from task1_image_edit.models.longcat_edit import LongCatEditRunner
        except ModuleNotFoundError as exc:
            if exc.name in {"diffusers", "peft"}:
                raise ModuleNotFoundError(
                    "longcat_edit_turbo requires the 'diffusers' and 'peft' packages in the same Python environment that runs scripts/infer.py"
                ) from exc
            raise
        return LongCatEditRunner(
            model_name_or_path=args.model_name_or_path,
            device=args.device,
            dtype=args.dtype,
            lora_path=args.lora_path,
            offload=args.offload,
        )
    common = {
        "model_name_or_path": args.model_name_or_path,
        "device": args.device,
        "dtype": args.dtype,
        "lora_path": args.lora_path,
        "offload": args.offload,
    }
    if canonical == "flux_kontext":
        try:
            from task1_image_edit.models.flux_kontext import FluxKontextRunner
        except ModuleNotFoundError as exc:
            if exc.name == "diffusers":
                raise ModuleNotFoundError(
                    "flux_kontext requires the 'diffusers' package in the same Python environment that runs scripts/infer.py"
                ) from exc
            raise
        return FluxKontextRunner(**common)
    raise ValueError(f"Unsupported model: {args.model}")


def build_prompt(args, row):
    field_map = {
        "qwen_edit": "qwen_prompt",
        "longcat_edit_turbo": "longcat_prompt",
        "flux_kontext": "flux_prompt",
    }
    negative_field_map = {
        "qwen_edit": ("qwen_negative_prompt", "negative_prompt"),
        "longcat_edit_turbo": ("longcat_negative_prompt", "negative_prompt"),
    }
    canonical = canonical_model_name(args.model)
    if args.prompt:
        return args.prompt, args.negative_prompt
    row_context = prompt_context_from_manifest_row(row) if row else None
    has_context_override = any(
        value is not None for value in (args.items, args.outfit_summary, args.extra_constraints)
    )
    if row and row.get(field_map[canonical]) and not has_context_override:
        row_negative_prompt = None
        for key in negative_field_map.get(canonical, ()):
            value = row.get(key)
            if value:
                row_negative_prompt = value
                break
        return row[field_map[canonical]], args.negative_prompt or row_negative_prompt
    items = _pick_prompt_context(args.items, row_context, "items")
    summary = _pick_prompt_context(args.outfit_summary, row_context, "outfit_summary")
    constraints = _pick_prompt_context(args.extra_constraints, row_context, "extra_constraints")
    prompt, negative_prompt = select_inference_prompt(
        canonical,
        items=items,
        outfit_summary=summary,
        extra_constraints=constraints,
    )
    return prompt, args.negative_prompt or negative_prompt


def run_single(runner, args):
    prompt, negative_prompt = build_prompt(args, None)
    request = InferenceRequest(
        input_image=args.input_image,
        output_path=args.output,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        true_cfg_scale=args.true_cfg_scale,
        max_sequence_length=args.max_sequence_length,
        edit_image_auto_resize=not args.disable_edit_image_auto_resize,
        zero_cond_t=args.zero_cond_t,
    )
    saved_path = runner.run(request)
    print(f"saved={saved_path}")
    print(f"prompt={prompt}")
    if negative_prompt:
        print(f"negative_prompt={negative_prompt}")


def run_batch(runner, args):
    rows = load_manifest(args.manifest)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        prompt, negative_prompt = build_prompt(args, row)
        output_subpath = row.get("output_subpath") or f"{row['sample_id']}.png"
        output_path = output_dir / output_subpath
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request = InferenceRequest(
            input_image=row["source_image"],
            output_path=str(output_path),
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=args.seed,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            true_cfg_scale=args.true_cfg_scale,
            max_sequence_length=args.max_sequence_length,
            edit_image_auto_resize=not args.disable_edit_image_auto_resize,
            zero_cond_t=args.zero_cond_t,
        )
        saved_path = runner.run(request)
        print(f"{row['sample_id']}\t{saved_path}")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Unified inference entry for Task1 image-edit baselines.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument(
        "--diffsynth-root",
        default=None,
        help="Optional local DiffSynth-Studio checkout for qwen-edit if 'diffsynth' is not installed.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--low-vram", action="store_true", help="DiffSynth low-VRAM mode for qwen-edit.")
    parser.add_argument("--zero-cond-t", action="store_true", help="Enable Qwen-Image-Edit-2511 special flag.")
    parser.add_argument("--disable-edit-image-auto-resize", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--items", nargs="*", default=None)
    parser.add_argument("--outfit-summary", default=None)
    parser.add_argument("--extra-constraints", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--image-guidance-scale", type=float, default=None)
    parser.add_argument("--true-cfg-scale", type=float, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--input-image", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest", default=None, help="Batch manifest path (.jsonl or bid->pid .npy index).")
    parser.add_argument("--data-root", default=None, help="Raw dataset root containing the 3 category folders under data/.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-ratio", default="7:1:2")
    parser.add_argument("--output-dir", default=None)
    argv = list(sys.argv[1:] if argv is None else argv)
    provided_options = _provided_option_names(argv)
    args = parser.parse_args(argv)

    canonical = canonical_model_name(args.model)
    if canonical == "qwen_edit":
        _validate_qwen_inputs(args, provided_options)
    defaults = {
        "qwen_edit": {
            "model_name_or_path": "Qwen/Qwen-Image-Edit",
            "dtype": "bf16",
            "steps": 28,
            "guidance_scale": 3.5,
            "image_guidance_scale": 1.5,
            "true_cfg_scale": 4.0,
        },
        "longcat_edit_turbo": {
            "model_name_or_path": "meituan-longcat/LongCat-Image-Edit-Turbo",
            "dtype": "bf16",
            "steps": None,
            "guidance_scale": None,
            "image_guidance_scale": 1.5,
            "true_cfg_scale": 4.0,
        },
        "flux_kontext": {
            "model_name_or_path": "black-forest-labs/FLUX.1-Kontext-dev",
            "dtype": "bf16",
            "steps": 28,
            "guidance_scale": 3.5,
            "image_guidance_scale": 1.5,
            "true_cfg_scale": 4.0,
        },
    }
    model_defaults = defaults[canonical]
    if canonical == "longcat_edit_turbo":
        if not args.model_name_or_path:
            args.model_name_or_path = (
                "meituan-longcat/LongCat-Image-Edit" if args.lora_path else "meituan-longcat/LongCat-Image-Edit-Turbo"
            )
        longcat_is_turbo = "turbo" in args.model_name_or_path.lower()
        if model_defaults["steps"] is None:
            model_defaults["steps"] = 8 if longcat_is_turbo else 50
        if model_defaults["guidance_scale"] is None:
            model_defaults["guidance_scale"] = 1.0 if longcat_is_turbo else 4.5
    elif not args.model_name_or_path:
        args.model_name_or_path = model_defaults["model_name_or_path"]
    if not args.dtype:
        args.dtype = model_defaults["dtype"]
    if args.steps is None:
        args.steps = model_defaults["steps"]
    if args.guidance_scale is None:
        args.guidance_scale = model_defaults["guidance_scale"]
    if args.image_guidance_scale is None:
        args.image_guidance_scale = model_defaults["image_guidance_scale"]
    if args.true_cfg_scale is None:
        args.true_cfg_scale = model_defaults["true_cfg_scale"]
    batch_inputs = [value for value in (args.manifest, args.data_root) if value]
    if len(batch_inputs) > 1:
        raise ValueError("Specify at most one of --manifest or --data-root")
    if bool(batch_inputs) == bool(args.input_image):
        raise ValueError("Specify exactly one of --input-image or a batch source (--manifest / --data-root)")
    if args.input_image and not args.output:
        raise ValueError("--output is required with --input-image")
    if batch_inputs and not args.output_dir:
        raise ValueError("--output-dir is required with batch inference")
    if args.data_root:
        manifest_dir = project_root() / "output" / "manifests"
        args.manifest = prepare_split_manifest(
            data_root=args.data_root,
            output_dir=manifest_dir,
            split=args.split,
            split_ratio=args.split_ratio,
            seed=args.split_seed,
        )
    args.model = canonical
    return args


def main():
    args = parse_args()
    runner = build_runner(args)
    if args.manifest:
        run_batch(runner, args)
    else:
        run_single(runner, args)


if __name__ == "__main__":
    main()
