#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover - torch is provided in the training environment
    torch = None

    class TorchDataset:  # type: ignore[override]
        pass

try:
    import Benchamark.Task3.mllm_eval as eval_utils
except ModuleNotFoundError:
    import mllm_eval as eval_utils

try:
    from Benchamark.Task3.task3_dataset import OutfitNegativeSampleDataset
except ModuleNotFoundError:
    from task3_dataset import OutfitNegativeSampleDataset

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DATASET_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "qwen25vl-7b")
DEFAULT_MODELS_DIR = os.environ.get("VLLM_MODELS_DIR", str(SCRIPT_DIR / "models"))
DEFAULT_RESULTS_DIR = os.environ.get("SFT_RESULTS_DIR", str(SCRIPT_DIR / "SFT_results"))
DEFAULT_DATASET_ROOT = os.environ.get(
    "FASHION_STYLIST_DATA_ROOT",
    os.environ.get("TASK3_DATASET_ROOT", str(SHARED_DATASET_ROOT)),
)
DEFAULT_PROMPT_FILE = os.environ.get("TASK3_PROMPT_FILE", str(SCRIPT_DIR / "template" / "prompt.txt"))
DEFAULT_TRAIN_SPLIT = os.environ.get("SFT_TRAIN_SPLIT", "train").strip().lower() or "train"
DEFAULT_TEST_SPLIT = os.environ.get("SFT_TEST_SPLIT", "test").strip().lower() or "test"
DEFAULT_SPLIT_SEED = int(os.environ.get("VLLM_EVAL_SPLIT_SEED", "42"))
DEFAULT_SAMPLE_MODE = os.environ.get("SFT_SAMPLE_MODE", "both")
DEFAULT_SELECTION_SPLIT = os.environ.get("SFT_CHECKPOINT_SELECTION_SPLIT", "val").strip().lower() or "val"
DEFAULT_SELECTION_METRIC = (
    os.environ.get("SFT_CHECKPOINT_SELECTION_METRIC", "mod_index_accuracy_on_modified_only").strip()
    or "mod_index_accuracy_on_modified_only"
)
DEFAULT_MAX_SEQ_LENGTH = int(os.environ.get("SFT_MAX_SEQ_LENGTH", "8192"))
DEFAULT_EVAL_MAX_NEW_TOKENS = int(os.environ.get("SFT_EVAL_MAX_NEW_TOKENS", "1024"))
DEFAULT_EVAL_TEMPERATURE = getattr(
    eval_utils,
    "DEFAULT_TEMPERATURE",
    float(os.environ.get("VLLM_TEMPERATURE", "0.3")),
)
DEFAULT_EVAL_TOP_K = getattr(
    eval_utils,
    "DEFAULT_TOP_K",
    int(os.environ.get("VLLM_TOP_K", "50")),
)
DEFAULT_BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "1"))
DEFAULT_GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", "4"))
DEFAULT_EPOCHS = float(os.environ.get("SFT_NUM_TRAIN_EPOCHS", "2"))
DEFAULT_LEARNING_RATE = float(os.environ.get("SFT_LEARNING_RATE", "2e-4"))
DEFAULT_LORA_R = int(os.environ.get("SFT_LORA_R", "16"))
DEFAULT_LORA_ALPHA = int(os.environ.get("SFT_LORA_ALPHA", "16"))
DEFAULT_LORA_DROPOUT = float(os.environ.get("SFT_LORA_DROPOUT", "0.0"))
DEFAULT_LOGGING_STEPS = int(os.environ.get("SFT_LOGGING_STEPS", "10"))
DEFAULT_SAVE_STEPS = int(os.environ.get("SFT_SAVE_STEPS", "100"))
DEFAULT_SAVE_TOTAL_LIMIT = int(os.environ.get("SFT_SAVE_TOTAL_LIMIT", "0"))
DEFAULT_SEED = int(os.environ.get("SFT_SEED", "3407"))
DEFAULT_WARMUP_RATIO = float(os.environ.get("SFT_WARMUP_RATIO", "0.03"))
DEFAULT_WARMUP_STEPS = int(os.environ.get("SFT_WARMUP_STEPS", "0"))
DEFAULT_VISION_IMAGE_SIZE = int(os.environ.get("SFT_VISION_IMAGE_SIZE", "512"))
DEFAULT_SWEEP_OUTPUT_DIR = os.environ.get("SFT_CHECKPOINT_SWEEP_OUTPUT_DIR", str(SCRIPT_DIR / "SFT_checkpoint_sweeps"))
MIN_REASONABLE_IMAGE_EDGE = 128
MAX_REASONABLE_IMAGE_EDGE = 2048
MIN_VLM_MAX_SEQ_LENGTH = 8192
TRAINING_THINKING_MODE = "hidden"
SELECTION_METRIC_CHOICES = (
    "json_valid_rate",
    "style_accuracy",
    "season_accuracy",
    "occasion_accuracy",
    "need_to_modify_accuracy",
    "mod_index_accuracy",
    "mod_index_accuracy_on_modified_only",
    "strict_accuracy",
)


def should_use_relaxed_visible_reasoning(model_name: str, model_family: str, thinking_mode: str) -> bool:
    helper = getattr(eval_utils, "should_use_relaxed_visible_reasoning", None)
    if callable(helper):
        return bool(helper(model_name, model_family, thinking_mode))
    if thinking_mode != "visible" or model_family != "qwen_vl":
        return False
    normalized = model_name.strip().lower().replace("_", "-")
    return "qwen3-vl-8b" in normalized or "qwen3vl-8b" in normalized


def merge_system_prompt_into_user_text(system_prompt: str, user_text: str) -> str:
    helper = getattr(eval_utils, "merge_system_prompt_into_user_text", None)
    if callable(helper):
        return helper(system_prompt, user_text)
    if not system_prompt:
        return user_text
    if not user_text:
        return system_prompt
    return f"{system_prompt}\n\n{user_text}"


def import_unsloth_components() -> tuple[Any, Any, Any, Any, Any]:
    start = time.perf_counter()
    print("[bootstrap] importing unsloth + trl", flush=True)
    try:
        from unsloth import FastVisionModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import unsloth.FastVisionModel: {exc.__class__.__name__}: {exc}") from exc

    try:
        from unsloth import is_bfloat16_supported
    except Exception:  # noqa: BLE001
        try:
            from unsloth import is_bf16_supported as is_bfloat16_supported
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to import a bf16 capability helper from unsloth: {exc.__class__.__name__}: {exc}"
            ) from exc

    try:
        from unsloth.trainer import UnslothVisionDataCollator
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to import UnslothVisionDataCollator: {exc.__class__.__name__}: {exc}"
        ) from exc

    try:
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import trl SFT classes: {exc.__class__.__name__}: {exc}") from exc

    print(f"[bootstrap] unsloth + trl imported in {time.perf_counter() - start:.1f}s", flush=True)
    return (
        FastVisionModel,
        UnslothVisionDataCollator,
        SFTTrainer,
        SFTConfig,
        is_bfloat16_supported,
    )


def validate_requested_model_for_sft(parser: argparse.ArgumentParser, model_name: str, model_family: str) -> None:
    eval_utils.validate_model_name(parser, model_name)
    if model_family in {"qwen_vl", "gemma_vl"}:
        return
    parser.error(
        f"Unsupported model family for this Unsloth SFT script: {model_family!r}. "
        "Supported families here are qwen_vl and gemma_vl."
    )


def resolve_training_output_paths(results_dir: Path, resolved_model_name: str) -> tuple[Path, Path, Path]:
    model_token = eval_utils.sanitize_path_token(resolved_model_name)
    adapter_dir = results_dir / f"{model_token}_lora"
    predictions_path = results_dir / f"{model_token}_sft_test_results.json"
    metrics_path = eval_utils.build_metrics_path(predictions_path)
    return adapter_dir, predictions_path, metrics_path


def import_peft_model() -> Any:
    try:
        from peft import PeftModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to import peft.PeftModel: {exc.__class__.__name__}: {exc}"
        ) from exc
    return PeftModel


def checkpoint_step(path: Path, adapter_dir: Path) -> int:
    if path.resolve() == adapter_dir.resolve():
        return 10**18
    suffix = path.name.removeprefix("checkpoint-")
    return int(suffix) if suffix.isdigit() else -1


def discover_adapter_candidates(adapter_dir: Path, include_final_adapter: bool) -> list[Path]:
    candidates = sorted(
        (path for path in adapter_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: checkpoint_step(path, adapter_dir),
    )
    if include_final_adapter and adapter_dir.is_dir():
        candidates.append(adapter_dir)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint candidates found under: {adapter_dir}")
    return candidates


def candidate_label(path: Path, adapter_dir: Path) -> str:
    if path.resolve() == adapter_dir.resolve():
        return "final"
    return path.name


def metric_score(metrics: dict[str, Any], metric_name: str) -> float:
    value = metrics.get(metric_name)
    if value is None:
        return float("-inf")
    return float(value)


def copy_if_different(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def resolve_model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not resolve model device: {exc}") from exc


def unload_model(model: Any) -> None:
    try:
        del model
    finally:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_system_prompt(
    prompt_template: str,
    thinking_mode: str,
    model_name: str,
    model_family: str,
) -> str:
    relaxed_visible_reasoning = should_use_relaxed_visible_reasoning(
        model_name,
        model_family,
        thinking_mode,
    )
    # Share the evaluator's prompt normalization so SFT training and SFT test
    # inference stay aligned with zero-shot prompting semantics.
    return eval_utils.build_effective_prompt_template(
        prompt_template,
        thinking_mode,
        relaxed_visible_reasoning=relaxed_visible_reasoning,
    )


def build_messages(
    prompt_template: str,
    prompt_input: dict[str, Any],
    images: list[Any],
    thinking_mode: str,
    model_name: str,
    model_family: str,
    assistant_text: str | None = None,
    include_image_objects: bool = True,
) -> list[dict[str, Any]]:
    relaxed_visible_reasoning = should_use_relaxed_visible_reasoning(
        model_name,
        model_family,
        thinking_mode,
    )
    user_prefix = (
        "/think\n"
        if eval_utils.should_emit_thinking(thinking_mode) and not relaxed_visible_reasoning
        else ""
    )
    system_prompt = build_system_prompt(
        prompt_template,
        thinking_mode,
        model_name=model_name,
        model_family=model_family,
    )
    user_text = eval_utils.build_structured_user_prompt(prompt_input, user_prefix=user_prefix)
    if include_image_objects:
        user_content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    else:
        user_content = [{"type": "image"} for _ in images]
    if model_family == "gemma_vl":
        user_text = merge_system_prompt_into_user_text(system_prompt, user_text)
    user_content.append({"type": "text", "text": user_text})

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    if model_family != "gemma_vl":
        messages.insert(0, {"role": "system", "content": system_prompt})
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
    return messages


class VisionSFTConversationDataset(TorchDataset):
    def __init__(
        self,
        base_dataset: OutfitNegativeSampleDataset,
        prompt_template: str,
        processor: Any,
        model_name: str,
        model_family: str,
    ) -> None:
        self.base_dataset = base_dataset
        self.prompt_template = prompt_template
        self.processor = processor
        self.model_name = model_name
        self.model_family = model_family

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _build_training_messages(self, index: int, include_image_objects: bool) -> list[dict[str, Any]]:
        sample = eval_utils.build_sample_record(self.base_dataset, index)
        images = [eval_utils.open_rgb_image(path) for path in sample["image_paths"]]
        assistant_text = json_dumps_compact(sample["gold"])
        return build_messages(
            prompt_template=self.prompt_template,
            prompt_input=sample["prompt_input"],
            images=images,
            thinking_mode=TRAINING_THINKING_MODE,
            model_name=self.model_name,
            model_family=self.model_family,
            assistant_text=assistant_text,
            include_image_objects=include_image_objects,
        )

    def _build_training_input_ids(self, index: int) -> list[int]:
        messages = self._build_training_messages(index, include_image_objects=False)
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        encoded = tokenizer(input_text, add_special_tokens=False)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        return list(input_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"messages": self._build_training_messages(index, include_image_objects=True)}

    def map(
        self,
        function: Any,
        batched: bool = False,
        batch_size: int | None = None,
        **_: Any,
    ) -> "VisionSFTConversationDataset":
        effective_batch_size = len(self) if not batch_size or batch_size <= 0 else batch_size
        if batched:
            for start in range(0, len(self), effective_batch_size):
                batch_input_ids = [
                    self._build_training_input_ids(index)
                    for index in range(start, min(len(self), start + effective_batch_size))
                ]
                function({"input_ids": batch_input_ids})
        else:
            for index in range(len(self)):
                function({"input_ids": self._build_training_input_ids(index)})
        return self


def json_dumps_compact(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_image_size_candidate(value: Any) -> tuple[int, int] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return (
            (value, value)
            if MIN_REASONABLE_IMAGE_EDGE <= value <= MAX_REASONABLE_IMAGE_EDGE
            else None
        )
    if isinstance(value, float):
        as_int = int(value)
        return (
            (as_int, as_int)
            if MIN_REASONABLE_IMAGE_EDGE <= as_int <= MAX_REASONABLE_IMAGE_EDGE
            else None
        )
    if isinstance(value, dict):
        if "height" in value and "width" in value:
            height = int(value["height"])
            width = int(value["width"])
            return (
                (height, width)
                if MIN_REASONABLE_IMAGE_EDGE <= height <= MAX_REASONABLE_IMAGE_EDGE
                and MIN_REASONABLE_IMAGE_EDGE <= width <= MAX_REASONABLE_IMAGE_EDGE
                else None
            )
        if "size" in value:
            return normalize_image_size_candidate(value["size"])
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        height = int(value[0])
        width = int(value[1])
        return (
            (height, width)
            if MIN_REASONABLE_IMAGE_EDGE <= height <= MAX_REASONABLE_IMAGE_EDGE
            and MIN_REASONABLE_IMAGE_EDGE <= width <= MAX_REASONABLE_IMAGE_EDGE
            else None
        )
    return None


def resolve_vision_resize(processor: Any, model: Any) -> tuple[tuple[int, int], str]:
    image_processor = getattr(processor, "image_processor", None)
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    candidates = (
        ("processor.image_processor.size", getattr(image_processor, "size", None)),
        ("processor.image_processor.crop_size", getattr(image_processor, "crop_size", None)),
        ("processor.size", getattr(processor, "size", None)),
        ("processor.crop_size", getattr(processor, "crop_size", None)),
        ("model.config.vision_config.image_size", getattr(vision_config, "image_size", None)),
        ("model.config.image_size", getattr(getattr(model, "config", None), "image_size", None)),
    )
    for source, candidate in candidates:
        normalized = normalize_image_size_candidate(candidate)
        if normalized is not None:
            return normalized, source
    fallback = (DEFAULT_VISION_IMAGE_SIZE, DEFAULT_VISION_IMAGE_SIZE)
    return fallback, "fallback_default"


def get_world_size() -> int:
    raw = os.environ.get("WORLD_SIZE", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 1


def resolve_warmup_steps(
    train_example_count: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    warmup_steps: int,
    warmup_ratio: float,
) -> int:
    if warmup_steps > 0:
        return warmup_steps
    if warmup_ratio <= 0 or train_example_count <= 0:
        return 0
    world_size = get_world_size()
    global_batch_size = max(1, per_device_train_batch_size * gradient_accumulation_steps * world_size)
    steps_per_epoch = max(1, math.ceil(train_example_count / global_batch_size))
    total_optimizer_steps = max(1, math.ceil(steps_per_epoch * num_train_epochs))
    return max(1, math.ceil(total_optimizer_steps * warmup_ratio))


def callable_accepts_keyword(function: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return False
    if keyword in signature.parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def lora_config_supports_ensure_weight_tying() -> bool:
    try:
        from peft import LoraConfig
    except Exception:  # noqa: BLE001
        return False
    return callable_accepts_keyword(LoraConfig, "ensure_weight_tying")


def should_enable_ensure_weight_tying(
    peft_model_getter: Any,
    model: Any,
    modules_to_save: list[str],
) -> bool:
    config = getattr(model, "config", None)
    if not bool(getattr(config, "tie_word_embeddings", False)):
        return False
    if not any(module in {"lm_head", "embed_tokens"} for module in modules_to_save):
        return False
    if not callable_accepts_keyword(peft_model_getter, "ensure_weight_tying"):
        return False
    return lora_config_supports_ensure_weight_tying()


def resolve_effective_max_seq_length(max_seq_length: int, model_family: str) -> int:
    if model_family in {"qwen_vl", "gemma_vl"}:
        return max(max_seq_length, MIN_VLM_MAX_SEQ_LENGTH)
    return max_seq_length


def prepare_processor_inputs(processor: Any, images: list[Any], input_text: str) -> Any:
    try:
        return processor(images, input_text, add_special_tokens=False, return_tensors="pt")
    except TypeError:
        return processor(images=images, text=input_text, add_special_tokens=False, return_tensors="pt")


def decode_generated_text(processor: Any, outputs: Any, prompt_length: int) -> str:
    generated_tokens = outputs[:, prompt_length:]
    if hasattr(processor, "batch_decode"):
        decoded = processor.batch_decode(generated_tokens, skip_special_tokens=True)
        if decoded:
            return decoded[0]
    if hasattr(processor, "decode"):
        return processor.decode(generated_tokens[0], skip_special_tokens=True)
    raise ValueError("Processor does not support decoding generated tokens.")


def build_inference_efficiency(
    prompt_length: int,
    completion_length: int,
    image_count: int,
    load_seconds: float,
    prompt_seconds: float,
    generation_seconds: float,
    parse_seconds: float,
    total_seconds: float,
) -> dict[str, Any]:
    return {
        "prompt_tokens": int(prompt_length),
        "completion_tokens": int(completion_length),
        "total_tokens": int(prompt_length + completion_length),
        "image_count": int(image_count),
        "load_seconds": round(load_seconds, 4),
        "prompt_seconds": round(prompt_seconds, 4),
        "generation_seconds": round(generation_seconds, 4),
        "parse_seconds": round(parse_seconds, 4),
        "total_seconds": round(total_seconds, 4),
    }


def summarize_efficiency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    efficiencies = [row.get("efficiency") for row in rows if isinstance(row.get("efficiency"), dict)]
    if not efficiencies:
        return {
            "rows_with_efficiency": 0,
            "total_seconds_sum": 0.0,
            "total_seconds_avg": None,
            "total_seconds_max": None,
            "generation_seconds_sum": 0.0,
            "generation_seconds_avg": None,
            "generation_seconds_max": None,
            "prompt_tokens_sum": 0,
            "completion_tokens_sum": 0,
            "total_tokens_sum": 0,
        }

    def float_values(key: str) -> list[float]:
        values: list[float] = []
        for item in efficiencies:
            value = item.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    def int_values(key: str) -> list[int]:
        values: list[int] = []
        for item in efficiencies:
            value = item.get(key)
            if isinstance(value, int):
                values.append(value)
        return values

    total_seconds_values = float_values("total_seconds")
    generation_seconds_values = float_values("generation_seconds")
    prompt_token_values = int_values("prompt_tokens")
    completion_token_values = int_values("completion_tokens")
    total_token_values = int_values("total_tokens")
    return {
        "rows_with_efficiency": len(efficiencies),
        "total_seconds_sum": round(sum(total_seconds_values), 4),
        "total_seconds_avg": round(sum(total_seconds_values) / len(total_seconds_values), 4)
        if total_seconds_values
        else None,
        "total_seconds_max": round(max(total_seconds_values), 4) if total_seconds_values else None,
        "generation_seconds_sum": round(sum(generation_seconds_values), 4),
        "generation_seconds_avg": round(sum(generation_seconds_values) / len(generation_seconds_values), 4)
        if generation_seconds_values
        else None,
        "generation_seconds_max": round(max(generation_seconds_values), 4) if generation_seconds_values else None,
        "prompt_tokens_sum": sum(prompt_token_values),
        "completion_tokens_sum": sum(completion_token_values),
        "total_tokens_sum": sum(total_token_values),
        "avg_total_tokens_per_sample": round(sum(total_token_values) / len(total_token_values), 4)
        if total_token_values
        else None,
    }


def load_model_for_checkpoint(
    resolved_model: str,
    adapter_path: Path,
    max_seq_length: int,
    load_in_4bit: bool,
) -> tuple[Any, Any]:
    FastVisionModel, _, _, _, _ = import_unsloth_components()
    PeftModel = import_peft_model()
    model, processor = FastVisionModel.from_pretrained(
        model_name=resolved_model,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
        max_seq_length=max_seq_length,
    )
    FastVisionModel.for_inference(model)
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    if hasattr(model, "eval"):
        model.eval()
    return model, processor


def evaluate_loaded_model(
    model: Any,
    processor: Any,
    dataset: OutfitNegativeSampleDataset,
    prompt_template: str,
    resolved_model_name: str,
    model_family: str,
    split_name: str,
    output_path: Path,
    thinking_mode: str,
    eval_max_new_tokens: int,
    temperature: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is not available in the current environment.")

    rows: list[dict[str, Any]] = []
    total_samples = len(dataset)
    do_sample = temperature > 0.0
    inference_started = time.perf_counter()
    model_device = resolve_model_device(model)

    for index in range(total_samples):
        progress = index + 1
        stage_prefix = f"[{split_name} {progress}/{total_samples}]"
        sample = eval_utils.build_sample_record(dataset, index)
        row = eval_utils.build_result_row(sample)
        rows.append(row)
        sample_started = time.perf_counter()
        load_seconds = 0.0
        prompt_seconds = 0.0
        generation_seconds = 0.0
        parse_seconds = 0.0
        prompt_length = 0
        completion_length = 0

        try:
            row["stage"] = "load_images"
            print(f"{stage_prefix} outfit_id={sample['outfit_id']} stage=load_images", flush=True)
            load_started = time.perf_counter()
            images = [eval_utils.open_rgb_image(path) for path in sample["image_paths"]]
            load_seconds = time.perf_counter() - load_started

            row["stage"] = "build_prompt"
            print(f"{stage_prefix} outfit_id={sample['outfit_id']} stage=build_prompt", flush=True)
            prompt_started = time.perf_counter()
            input_text = build_inference_prompt_text(
                prompt_template=prompt_template,
                prompt_input=sample["prompt_input"],
                image_count=len(images),
                thinking_mode=thinking_mode,
                model_name=resolved_model_name,
                model_family=model_family,
            )
            inputs = prepare_processor_inputs(processor, images, input_text)
            inputs = inputs.to(model_device)
            prompt_length = int(inputs["input_ids"].shape[-1])
            prompt_seconds = time.perf_counter() - prompt_started

            generation_kwargs = {
                "max_new_tokens": eval_utils.resolve_effective_max_tokens(
                    eval_max_new_tokens,
                    thinking_mode,
                    model_family,
                ),
                "use_cache": True,
                "do_sample": do_sample,
                "top_p": 1.0,
                "top_k": DEFAULT_EVAL_TOP_K,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature

            row["stage"] = "generate"
            print(f"{stage_prefix} outfit_id={sample['outfit_id']} stage=generate", flush=True)
            generation_started = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(**inputs, **generation_kwargs)
            generation_seconds = time.perf_counter() - generation_started

            row["stage"] = "parse_output"
            print(f"{stage_prefix} outfit_id={sample['outfit_id']} stage=parse_output", flush=True)
            parse_started = time.perf_counter()
            raw_text = decode_generated_text(processor, outputs, prompt_length)
            completion_length = max(0, int(outputs.shape[-1]) - prompt_length)
            row["stage"] = "score"
            print(f"{stage_prefix} outfit_id={sample['outfit_id']} stage=score", flush=True)
            eval_utils.finalize_row_from_raw_text(row, sample, raw_text)
            parse_seconds = time.perf_counter() - parse_started
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{exc.__class__.__name__}: {exc}"
        finally:
            row["efficiency"] = build_inference_efficiency(
                prompt_length=prompt_length,
                completion_length=completion_length,
                image_count=len(sample["image_paths"]),
                load_seconds=load_seconds,
                prompt_seconds=prompt_seconds,
                generation_seconds=generation_seconds,
                parse_seconds=parse_seconds,
                total_seconds=time.perf_counter() - sample_started,
            )

        print(
            f"{stage_prefix} outfit_id={sample['outfit_id']} stage={row['stage']} "
            f"json_valid={row['json_valid']} error={row['error']}",
            flush=True,
        )

    eval_utils.write_json(output_path, rows)
    metrics = eval_utils.summarize_metrics(rows)
    metrics["efficiency"] = summarize_efficiency(rows)
    metrics["inference_seconds"] = round(time.perf_counter() - inference_started, 4)
    metrics["split"] = split_name
    metrics_path = eval_utils.build_metrics_path(output_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, metrics


def build_inference_prompt_text(
    prompt_template: str,
    prompt_input: dict[str, Any],
    image_count: int,
    thinking_mode: str,
    model_name: str,
    model_family: str,
) -> str:
    # Reuse the evaluator's prompt renderer so zero-shot and SFT test inference
    # compare the same prompt text for the same sample.
    return eval_utils.build_model_prompt(
        prompt_template=prompt_template,
        prompt_input=prompt_input,
        image_count=image_count,
        model_name=model_name,
        model_family=model_family,
        thinking_mode=thinking_mode,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Qwen or Gemma vision models with Unsloth LoRA, "
            "select the best checkpoint on the validation split, and evaluate that checkpoint on the test split."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"Model alias, local path, or Hugging Face repo id. Default: {DEFAULT_MODEL}. "
            "Supported here: qwen25vl-7b, qwen3vl-4b, qwen3vl-8b, unsloth-gemma3-4b."
        ),
    )
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help=f"Directory containing downloaded models. Default: {DEFAULT_MODELS_DIR}",
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root directory containing the Female, Male, and Child folders, each with look*.csv, label*.csv, and photos*.",
    )
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE, help="Prompt template file.")
    parser.add_argument("--sample-mode", choices=("original", "modified", "both"), default=DEFAULT_SAMPLE_MODE)
    parser.add_argument(
        "--selection-split",
        choices=eval_utils.SPLIT_CHOICES,
        default=DEFAULT_SELECTION_SPLIT if DEFAULT_SELECTION_SPLIT in eval_utils.SPLIT_CHOICES else "val",
        help="Split used to select the best checkpoint. Default: val.",
    )
    parser.add_argument("--train-split", choices=eval_utils.SPLIT_CHOICES, default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--test-split", choices=eval_utils.SPLIT_CHOICES, default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help=f"Output directory. Default: {DEFAULT_RESULTS_DIR}")
    parser.add_argument(
        "--sweep-output-dir",
        default=DEFAULT_SWEEP_OUTPUT_DIR,
        help=(
            "Directory used to write checkpoint sweep outputs. "
            f"Default: {DEFAULT_SWEEP_OUTPUT_DIR}"
        ),
    )
    parser.add_argument("--num-train-epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--per-device-train-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help="Explicit warmup steps. If > 0, overrides --warmup-ratio.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=DEFAULT_SAVE_TOTAL_LIMIT,
        help=(
            "Maximum number of trainer checkpoints to keep. "
            "Use 0 or a negative value to keep all checkpoints for validation-based checkpoint selection. "
            f"Default: {DEFAULT_SAVE_TOTAL_LIMIT}"
        ),
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=(
            "Model context length passed to FastVisionModel.from_pretrained. "
            "The SFT trainer itself disables truncation for VLM batches to avoid dropping image tokens."
        ),
    )
    parser.add_argument("--eval-max-new-tokens", type=int, default=DEFAULT_EVAL_MAX_NEW_TOKENS)
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_EVAL_TEMPERATURE,
        help=(
            "Generation temperature for test prediction. "
            f"Default matches evaluate.py: {DEFAULT_EVAL_TEMPERATURE}."
        ),
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("hidden", "visible"),
        default="hidden",
        help=(
            "Test-time reasoning mode. Training targets supervise only the final JSON, so hidden is the safer default."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRIC_CHOICES,
        default=DEFAULT_SELECTION_METRIC if DEFAULT_SELECTION_METRIC in SELECTION_METRIC_CHOICES else "mod_index_accuracy_on_modified_only",
        help=(
            "Metric used to select the best checkpoint on the validation split. "
            "The default follows the paper setting and selects checkpoints by "
            "modified-item index accuracy on modified samples. "
            "Default: mod_index_accuracy_on_modified_only."
        ),
    )
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--include-final-adapter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the final adapter directory itself as a checkpoint candidate. Default: true.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the base model in 4-bit before LoRA fine-tuning. Default: false",
    )
    parser.add_argument(
        "--allow-hf-download",
        action="store_true",
        help="Allow loading from Hugging Face when a local model directory is not found.",
    )
    parser.add_argument(
        "--skip-test-inference",
        action="store_true",
        help="Train and save checkpoints/adapters, but skip validation-based checkpoint selection and test inference.",
    )
    return parser


def main() -> int:
    run_started = time.perf_counter()
    print("[step 1/8] parsing arguments", flush=True)
    parser = build_argument_parser()
    args = parser.parse_args()
    if torch is None:
        raise RuntimeError("PyTorch is not available. Please run this script inside your training environment.")
    if args.per_device_train_batch_size < 1:
        parser.error("--per-device-train-batch-size must be at least 1.")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be at least 1.")
    if args.train_split != "train":
        parser.error("This SFT entrypoint expects --train-split train.")
    if args.test_split != "test":
        parser.error("This SFT entrypoint expects --test-split test.")
    if args.selection_split == args.train_split:
        parser.error("--selection-split must differ from --train-split.")

    eval_utils.normalize_cuda_visible_devices(parser)
    models_dir = Path(args.models_dir).expanduser()
    resolved_model, resolved_model_name, model_family = eval_utils.resolve_requested_model(
        parser=parser,
        requested_model=args.model,
        models_dir=models_dir,
        allow_hf_download=args.allow_hf_download,
    )
    validate_requested_model_for_sft(parser, args.model, model_family)
    requested_max_seq_length = args.max_seq_length
    args.max_seq_length = resolve_effective_max_seq_length(args.max_seq_length, model_family)

    results_dir = Path(args.results_dir).expanduser()
    adapter_dir, predictions_path, metrics_path = resolve_training_output_paths(results_dir, resolved_model_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    sweep_output_root = Path(args.sweep_output_dir).expanduser()
    sweep_dir = sweep_output_root / f"{eval_utils.sanitize_path_token(resolved_model_name)}_checkpoint_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "requested_model": args.model,
        "resolved_model": resolved_model,
        "resolved_model_name": resolved_model_name,
        "model_family": model_family,
        "root": args.root,
        "prompt_file": args.prompt_file,
        "sample_mode": args.sample_mode,
        "selection_split": args.selection_split,
        "selection_metric": args.selection_metric,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "split_seed": args.split_seed,
        "results_dir": str(results_dir),
        "sweep_output_dir": str(sweep_output_root),
        "sweep_dir": str(sweep_dir),
        "adapter_dir": str(adapter_dir),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "requested_warmup_steps": args.warmup_steps,
        "save_total_limit": args.save_total_limit,
        "max_seq_length": args.max_seq_length,
        "eval_max_new_tokens": args.eval_max_new_tokens,
        "thinking_mode": args.thinking_mode,
        "include_final_adapter": args.include_final_adapter,
        "load_in_4bit": args.load_in_4bit,
        "temperature": args.temperature,
        "top_k": DEFAULT_EVAL_TOP_K,
        "skip_test_inference": args.skip_test_inference,
    }
    print("[config] SFT configuration:", flush=True)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    print("[step 2/8] importing datasets + prompt", flush=True)
    prompt_template = eval_utils.load_prompt(Path(args.prompt_file))
    train_dataset_base = OutfitNegativeSampleDataset(
        root=args.root,
        transform=None,
        deterministic=True,
        sample_mode=args.sample_mode,
        split=args.train_split,
        split_seed=args.split_seed,
    )
    selection_dataset_base = OutfitNegativeSampleDataset(
        root=args.root,
        transform=None,
        deterministic=True,
        sample_mode=args.sample_mode,
        split=args.selection_split,
        split_seed=args.split_seed,
    )
    test_dataset_base = OutfitNegativeSampleDataset(
        root=args.root,
        transform=None,
        deterministic=True,
        sample_mode=args.sample_mode,
        split=args.test_split,
        split_seed=args.split_seed,
    )
    print(
        f"[step 2/8] train_dataset: outfits={len(train_dataset_base.outfits)} samples={len(train_dataset_base)} "
        f"split_counts={train_dataset_base.split_outfit_counts_by_segment}",
        flush=True,
    )
    print(
        f"[step 2/8] selection_dataset: split={args.selection_split} "
        f"outfits={len(selection_dataset_base.outfits)} samples={len(selection_dataset_base)} "
        f"split_counts={selection_dataset_base.split_outfit_counts_by_segment}",
        flush=True,
    )
    print(
        f"[step 2/8] test_dataset: outfits={len(test_dataset_base.outfits)} samples={len(test_dataset_base)} "
        f"split_counts={test_dataset_base.split_outfit_counts_by_segment}",
        flush=True,
    )
    warmup_steps = resolve_warmup_steps(
        train_example_count=len(train_dataset_base),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
    )
    max_images_per_outfit = max(
        eval_utils.get_max_image_count(train_dataset_base),
        eval_utils.get_max_image_count(test_dataset_base),
    )
    if args.max_seq_length != requested_max_seq_length:
        print(
            f"[config] increased max_seq_length from {requested_max_seq_length} to {args.max_seq_length} "
            f"for model_family={model_family} max_images_per_outfit={max_images_per_outfit}",
            flush=True,
        )

    print("[step 3/8] importing unsloth", flush=True)
    (
        FastVisionModel,
        UnslothVisionDataCollator,
        SFTTrainer,
        SFTConfig,
        is_bfloat16_supported,
    ) = import_unsloth_components()

    print("[step 4/8] loading base model", flush=True)
    model, processor = FastVisionModel.from_pretrained(
        model_name=resolved_model,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
        max_seq_length=args.max_seq_length,
    )
    modules_to_save: list[str] = ["lm_head", "embed_tokens"]
    if model_family == "gemma_vl" and args.load_in_4bit:
        # Gemma 3 + 4-bit can upcast lm_head/embed_tokens via modules_to_save,
        # then hit Half vs Float mismatches during the first training step.
        modules_to_save = []
        print(
            "[config] disabling modules_to_save for 4-bit Gemma to avoid lm_head/embed_tokens dtype mismatches",
            flush=True,
        )
    peft_model_kwargs: dict[str, Any] = {}
    if should_enable_ensure_weight_tying(FastVisionModel.get_peft_model, model, modules_to_save):
        peft_model_kwargs["ensure_weight_tying"] = True
        print("[config] enabling ensure_weight_tying=True for tied embeddings", flush=True)
    elif bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False)):
        print(
            "[config] tied embeddings detected, but current unsloth/peft stack does not expose ensure_weight_tying",
            flush=True,
        )
    if modules_to_save:
        peft_model_kwargs["modules_to_save"] = modules_to_save
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
        **peft_model_kwargs,
    )
    vision_resize, vision_resize_source = resolve_vision_resize(processor, model)
    print(
        f"[config] using UnslothVisionDataCollator resize={vision_resize} source={vision_resize_source}",
        flush=True,
    )
    print(
        f"[config] warmup_steps={warmup_steps} "
        f"(requested_warmup_steps={args.warmup_steps} warmup_ratio={args.warmup_ratio})",
        flush=True,
    )

    print("[step 5/8] preparing SFT trainer", flush=True)
    FastVisionModel.for_training(model)
    train_dataset = VisionSFTConversationDataset(
        train_dataset_base,
        prompt_template,
        processor,
        model_name=resolved_model_name,
        model_family=model_family,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=UnslothVisionDataCollator(model, processor, resize=vision_resize),
        train_dataset=train_dataset,
        args=SFTConfig(
            output_dir=str(adapter_dir),
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            warmup_steps=warmup_steps,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=None if args.save_total_limit <= 0 else args.save_total_limit,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            remove_unused_columns=False,
            report_to="none",
            seed=args.seed,
            max_seq_length=args.max_seq_length,
            dataloader_pin_memory=False,
        ),
    )

    print("[step 6/8] starting LoRA fine-tuning", flush=True)
    training_started = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - training_started
    trainer.save_model(str(adapter_dir))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(adapter_dir))
    print(f"[step 6/8] adapter saved to {adapter_dir} training_seconds={training_seconds:.4f}", flush=True)

    if args.skip_test_inference:
        print(
            "[step 7/9] skipping validation-based checkpoint selection and test inference "
            "because --skip-test-inference was set",
            flush=True,
        )
        print(f"\nAdapter saved to: {adapter_dir}")
        return 0

    print("[step 7/9] selecting the best checkpoint on the validation split", flush=True)
    del trainer
    unload_model(model)
    model = None
    processor = None

    candidates = discover_adapter_candidates(adapter_dir, include_final_adapter=args.include_final_adapter)
    candidate_summaries: list[dict[str, Any]] = []
    best_summary: dict[str, Any] | None = None
    best_results_path: Path | None = None
    best_metrics_path: Path | None = None

    for candidate_path in candidates:
        label = candidate_label(candidate_path, adapter_dir)
        label_token = eval_utils.sanitize_path_token(label)
        selection_results_path = sweep_dir / f"{label_token}_{args.selection_split}_results.json"
        selection_metrics_path = eval_utils.build_metrics_path(selection_results_path)
        print(
            f"[candidate] evaluating {label} on split={args.selection_split} selection_metric={args.selection_metric}",
            flush=True,
        )
        candidate_model = candidate_processor = None
        try:
            candidate_model, candidate_processor = load_model_for_checkpoint(
                resolved_model=resolved_model,
                adapter_path=candidate_path,
                max_seq_length=args.max_seq_length,
                load_in_4bit=args.load_in_4bit,
            )
            _, selection_metrics = evaluate_loaded_model(
                model=candidate_model,
                processor=candidate_processor,
                dataset=selection_dataset_base,
                prompt_template=prompt_template,
                resolved_model_name=resolved_model_name,
                model_family=model_family,
                split_name=args.selection_split,
                output_path=selection_results_path,
                thinking_mode=args.thinking_mode,
                eval_max_new_tokens=args.eval_max_new_tokens,
                temperature=args.temperature,
            )
        finally:
            if candidate_model is not None:
                unload_model(candidate_model)

        summary = {
            "label": label,
            "path": str(candidate_path),
            "selection_results_path": str(selection_results_path),
            "selection_metrics_path": str(selection_metrics_path),
            "selection_metrics": selection_metrics,
            "selection_metric_value": selection_metrics.get(args.selection_metric),
            "step": checkpoint_step(candidate_path, adapter_dir),
        }
        candidate_summaries.append(summary)

        if best_summary is None:
            best_summary = summary
            best_results_path = selection_results_path
            best_metrics_path = selection_metrics_path
            continue

        current_score = metric_score(selection_metrics, args.selection_metric)
        best_score = metric_score(best_summary["selection_metrics"], args.selection_metric)
        if current_score > best_score or (
            current_score == best_score and summary["step"] > int(best_summary["step"])
        ):
            best_summary = summary
            best_results_path = selection_results_path
            best_metrics_path = selection_metrics_path

    if best_summary is None or best_results_path is None or best_metrics_path is None:
        raise RuntimeError("Checkpoint selection did not produce any validation results.")

    copy_if_different(best_results_path, sweep_dir / f"best_{args.selection_split}_results.json")
    copy_if_different(best_metrics_path, sweep_dir / f"best_{args.selection_split}_eval.json")

    print("[step 8/9] running test split inference with the selected checkpoint", flush=True)
    best_candidate_path = Path(best_summary["path"])
    final_results_path = sweep_dir / f"best_{args.test_split}_results.json"
    final_metrics_path = eval_utils.build_metrics_path(final_results_path)
    best_model = best_processor = None
    try:
        best_model, best_processor = load_model_for_checkpoint(
            resolved_model=resolved_model,
            adapter_path=best_candidate_path,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
        )
        _, final_metrics = evaluate_loaded_model(
            model=best_model,
            processor=best_processor,
            dataset=test_dataset_base,
            prompt_template=prompt_template,
            resolved_model_name=resolved_model_name,
            model_family=model_family,
            split_name=args.test_split,
            output_path=final_results_path,
            thinking_mode=args.thinking_mode,
            eval_max_new_tokens=args.eval_max_new_tokens,
            temperature=args.temperature,
        )
    finally:
        if best_model is not None:
            unload_model(best_model)

    final_metrics["training_seconds"] = round(training_seconds, 4)
    final_metrics["run_seconds"] = round(time.perf_counter() - run_started, 4)
    final_metrics_path.write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    copy_if_different(final_results_path, predictions_path)
    copy_if_different(final_metrics_path, metrics_path)

    final_evaluation = {
        "split": args.test_split,
        "results_path": str(final_results_path),
        "metrics_path": str(final_metrics_path),
        "metrics": final_metrics,
    }

    print("[step 9/9] writing checkpoint selection summary", flush=True)
    summary_path = sweep_dir / "checkpoint_sweep_summary.json"
    summary_payload = {
        "requested_model": args.model,
        "resolved_model_name": resolved_model_name,
        "adapter_dir": str(adapter_dir),
        "selection_split": args.selection_split,
        "final_split": args.test_split,
        "selection_metric": args.selection_metric,
        "thinking_mode": args.thinking_mode,
        "temperature": args.temperature,
        "best_checkpoint": {
            "label": best_summary["label"],
            "path": best_summary["path"],
            "selection_metric_value": best_summary["selection_metric_value"],
            "selection_results_path": best_summary["selection_results_path"],
            "selection_metrics_path": best_summary["selection_metrics_path"],
        },
        "final_evaluation": final_evaluation,
        "candidates": candidate_summaries,
        "run_seconds": round(time.perf_counter() - run_started, 4),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nBest checkpoint:")
    print(json.dumps(summary_payload["best_checkpoint"], ensure_ascii=False, indent=2))
    print("\nMetrics:")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    print(f"\nAdapter saved to: {adapter_dir}")
    print(f"Predictions written to: {predictions_path}")
    print(f"Metrics written to: {metrics_path}")
    print(f"Checkpoint sweep summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
