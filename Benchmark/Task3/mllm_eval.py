#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DATASET_ROOT = SCRIPT_DIR.parent.parent
SPLIT_CHOICES = ("all", "train", "val", "test")
QWEN_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
XML_TAG_LINE_RE = re.compile(r"^<[^>\n]+>$")
MARKDOWN_FENCE_LINE_RE = re.compile(r"^```(?:[\w-]+)?$")
INLINE_TOOL_TAG_RE = re.compile(
    r"</?(?:tool_call|tool_response|function_call|function_response|response)\b[^>]*>",
    re.IGNORECASE,
)
KEY_VALUE_LINE_RE = re.compile(
    r'^\s*"?(?P<key>outfit_summary|outfit_style|season|occasion|need_to_modify|mod_index)"?\s*[:=]\s*(?P<value>.*?)\s*,?\s*$',
    re.IGNORECASE,
)
MODEL_SPECS = {
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "local_dir_name": "Qwen2.5-VL-7B-Instruct",
        "aliases": ("qwen25vl-7b", "qwen2.5-vl-7b", "qwen25-7b"),
        "family": "qwen_vl",
    },
    "Qwen/Qwen3-VL-4B-Instruct": {
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "local_dir_name": "Qwen3-VL-4B-Instruct",
        "aliases": ("qwen3vl-4b", "qwen3-vl-4b", "qwen3-4b"),
        "family": "qwen_vl",
    },
    "Qwen/Qwen3-VL-4B-Thinking": {
        "repo_id": "Qwen/Qwen3-VL-4B-Thinking",
        "local_dir_name": "Qwen3-VL-4B-Thinking",
        "aliases": ("qwen3vl-4b-thinking", "qwen3-vl-4b-thinking", "qwen3-4b-thinking"),
        "family": "qwen_vl",
    },
    "Qwen/Qwen3-VL-8B-Instruct": {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
        "local_dir_name": "Qwen3-VL-8B-Instruct",
        "aliases": ("qwen3vl-8b", "qwen3-vl-8b", "qwen3-8b"),
        "family": "qwen_vl",
    },
    "Qwen/Qwen3-VL-8B-Thinking": {
        "repo_id": "Qwen/Qwen3-VL-8B-Thinking",
        "local_dir_name": "Qwen3-VL-8B-Thinking",
        "aliases": ("qwen3vl-8b-thinking", "qwen3-vl-8b-thinking", "qwen3-8b-thinking"),
        "family": "qwen_vl",
    },
    "unsloth/gemma-3-4b-it": {
        "repo_id": "unsloth/gemma-3-4b-it",
        "local_dir_name": "unsloth-gemma-3-4b-it",
        "aliases": ("unsloth-gemma3-4b", "unsloth-gemma-3-4b", "unsloth-gemma-3-4b-it"),
        "family": "gemma_vl",
    },
}
INVALID_MODEL_HINTS = {
    "Qwen/Qwen2.5-VL-8B-Instruct": (
        "Qwen2.5-VL does not have an official 8B checkpoint. "
        "Use Qwen/Qwen2.5-VL-7B-Instruct or switch to Qwen/Qwen3-VL-8B-Instruct."
    ),
}
MODEL_MIN_MAX_MODEL_LEN = {}
VISIBLE_REASONING_MIN_MAX_TOKENS = {
    "qwen_vl": 12288,
}
VISIBLE_REASONING_MIN_MAX_MODEL_LEN = {
    "qwen_vl": 32768,
}
MODE_SENSITIVE_PROMPT_LINE_PREFIXES = (
    "Keep this reasoning hidden.",
    "Do not output your reasoning process, chain-of-thought, scratch work, bullet analysis, or `<think>` content.",
    "- Do not output reasoning, explanations, or any extra text outside the required JSON.",
    "Do not output reasoning, explanations, or any extra text outside the required JSON.",
    "- Do not output `<think>`, `<analysis>`, or any hidden-reasoning markers.",
    "Do not output `<think>`, `<analysis>`, or any hidden-reasoning markers.",
    "Only output `<think>...</think>` when the evaluator explicitly asks for visible reasoning.",
    "If the evaluator asks for visible reasoning, place that reasoning before the final JSON.",
    "If the evaluator asks for visible reasoning, place that reasoning before the final JSON and",
    "If the evaluator asks for visible reasoning, keep it extremely short:",
    "- Perform the reasoning internally first, then output only the final JSON.",
    "Perform the reasoning internally first, then output only the final JSON.",
    "- If output length becomes tight, stop the reasoning immediately and output the final JSON object.",
    "If output length becomes tight, stop the reasoning immediately and output the final JSON object.",
    "The first non-whitespace character of the response must be `{` and the last must be `}`.",
    "Return JSON only.",
)


@dataclass(frozen=True)
class RuntimeDefaults:
    model: str
    output: str | None
    models_dir: str
    dataset_root: str
    prompt_file: str
    results_dir: Path
    temperature: float
    top_k: int
    thinking_mode: str
    limit: int
    batch_size: int
    max_tokens: int
    max_model_len: int
    split: str
    split_seed: int
    sample_mode: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    dtype: str
    setting: str


def load_runtime_defaults() -> RuntimeDefaults:
    return RuntimeDefaults(
        model=os.environ.get("VLLM_MODEL", "qwen25vl-7b"),
        output=os.environ.get("VLLM_EVAL_OUTPUT") or None,
        models_dir=os.environ.get("VLLM_MODELS_DIR", str(SCRIPT_DIR / "models")),
        dataset_root=os.environ.get(
            "FASHION_STYLIST_DATA_ROOT",
            os.environ.get("TASK3_DATASET_ROOT", str(SHARED_DATASET_ROOT)),
        ),
        prompt_file=os.environ.get("TASK3_PROMPT_FILE", str(SCRIPT_DIR / "template" / "prompt.txt")),
        results_dir=Path(os.environ.get("TASK3_RESULTS_DIR", str(SCRIPT_DIR / "results"))),
        temperature=float(os.environ.get("VLLM_TEMPERATURE", "0.3")),
        top_k=int(os.environ.get("VLLM_TOP_K", "50")),
        thinking_mode=os.environ.get("VLLM_THINKING_MODE", "visible").strip().lower() or "visible",
        limit=int(os.environ.get("VLLM_EVAL_LIMIT", "0")),
        batch_size=int(os.environ.get("VLLM_EVAL_BATCH_SIZE", "4")),
        max_tokens=int(os.environ.get("VLLM_MAX_TOKENS", "1024")),
        max_model_len=int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384")),
        split=os.environ.get("VLLM_EVAL_SPLIT", "test").strip().lower() or "test",
        split_seed=int(os.environ.get("VLLM_EVAL_SPLIT_SEED", "42")),
        sample_mode="both",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="auto",
        setting="zero-shot",
    )


def should_use_relaxed_visible_reasoning(model_name: str, model_family: str, thinking_mode: str) -> bool:
    if not should_emit_thinking(thinking_mode):
        return False
    if model_family != "qwen_vl":
        return False
    normalized = model_name.strip().lower().replace("_", "-")
    return "qwen3-vl-8b" in normalized or "qwen3vl-8b" in normalized


def merge_system_prompt_into_user_text(system_prompt: str, user_text: str) -> str:
    if not system_prompt:
        return user_text
    if not user_text:
        return system_prompt
    return f"{system_prompt}\n\n{user_text}"


def import_dataset_class() -> Any:
    start = time.perf_counter()
    print("[bootstrap] importing task3_dataset", flush=True)
    try:
        from task3_dataset import OutfitNegativeSampleDataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to import task3_dataset: {exc.__class__.__name__}: {exc}"
        ) from exc
    print(
        f"[bootstrap] task3_dataset imported in {time.perf_counter() - start:.1f}s",
        flush=True,
    )
    return OutfitNegativeSampleDataset


def import_torch_module() -> Any:
    start = time.perf_counter()
    print("[bootstrap] importing torch", flush=True)
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import torch: {exc.__class__.__name__}: {exc}") from exc
    print(
        f"[bootstrap] torch imported in {time.perf_counter() - start:.1f}s "
        f"version={getattr(torch, '__version__', 'unknown')}",
        flush=True,
    )
    return torch


def import_vllm_components() -> tuple[Any, Any]:
    start = time.perf_counter()
    print("[bootstrap] importing vllm", flush=True)
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import vllm: {exc.__class__.__name__}: {exc}") from exc
    print(
        f"[bootstrap] vllm imported in {time.perf_counter() - start:.1f}s",
        flush=True,
    )
    return LLM, SamplingParams


def import_transformers_auto_processor() -> Any:
    start = time.perf_counter()
    print("[bootstrap] importing transformers.AutoProcessor", flush=True)
    try:
        from transformers import AutoProcessor
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to import transformers.AutoProcessor: {exc.__class__.__name__}: {exc}"
        ) from exc
    print(
        f"[bootstrap] transformers.AutoProcessor imported in {time.perf_counter() - start:.1f}s",
        flush=True,
    )
    return AutoProcessor


def format_mod_index(value: Any) -> Any:
    return "NONE" if value is None else value


def configure_vllm_worker_multiproc_method() -> None:
    existing = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if existing:
        print(
            f"[step 4/7] using preconfigured VLLM_WORKER_MULTIPROC_METHOD={existing!r}",
            flush=True,
        )
        return
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    print(
        "[step 4/7] set VLLM_WORKER_MULTIPROC_METHOD='spawn' before torch.cuda/vllm init",
        flush=True,
    )


def load_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()


def should_emit_thinking(thinking_mode: str) -> bool:
    return thinking_mode == "visible"


def strip_mode_sensitive_prompt_lines(prompt_template: str) -> str:
    cleaned_lines: list[str] = []
    previous_blank = False
    for raw_line in prompt_template.strip().splitlines():
        stripped = raw_line.strip()
        if stripped and any(stripped.startswith(prefix) for prefix in MODE_SENSITIVE_PROMPT_LINE_PREFIXES):
            continue
        if not stripped:
            if previous_blank:
                continue
            previous_blank = True
            cleaned_lines.append("")
            continue
        previous_blank = False
        cleaned_lines.append(raw_line.rstrip())
    return "\n".join(cleaned_lines).strip()


def build_reasoning_mode_suffix(thinking_mode: str, relaxed_visible_reasoning: bool = False) -> str:
    if should_emit_thinking(thinking_mode):
        if relaxed_visible_reasoning:
            return (
                "Reasoning mode:\n"
                "- Make this reasoning visible before the final JSON.\n"
                "- Output visible reasoning before the final JSON.\n"
                "- Do not use hidden-reasoning markers or tool wrappers.\n"
                "- Think as fully as needed before finalizing the answer.\n"
                "- After the reasoning preamble, output only the final JSON object.\n"
                "- If the reasoning is getting long, stop it immediately and output the JSON object.\n"
                "- The final non-whitespace character of the full response must be `}`.\n"
                "- Return visible reasoning first, then the final JSON."
            )
        return (
            "Reasoning mode:\n"
            "- Make this reasoning visible in a single `<think>...</think>` block before the final JSON.\n"
            "- You must output your reasoning process in a single `<think>...</think>` block, then output the final JSON only.\n"
            "- Output reasoning only inside a single `<think>...</think>` block before the final JSON. Outside `<think>`, output only the final JSON.\n"
            "- You must output exactly one `<think>...</think>` block before the final JSON. Do not use any other reasoning markers.\n"
            "- Begin with `<think>` immediately and think as fully as needed before giving the final JSON.\n"
            "- If the `<think>` block is getting long, stop it immediately and output the JSON object.\n"
            "- The first non-whitespace character after `</think>` must be `{` and the last non-whitespace character of the full response must be `}`.\n"
            "- Return a single `<think>...</think>` block followed by JSON."
        )

    return (
        "Reasoning mode:\n"
        "- Keep this reasoning hidden.\n"
        "- Perform the reasoning internally first, then output only the final JSON.\n"
        "- Do not output your reasoning process, chain-of-thought, scratch work, bullet analysis, or `<think>` content. Return the final JSON only.\n"
        "- Do not output reasoning, explanations, or any extra text outside the required JSON.\n"
        "- Do not output `<think>`, `<analysis>`, or any hidden-reasoning markers.\n"
        "- If output length becomes tight, stop the reasoning immediately and output the final JSON object.\n"
        "- The first non-whitespace character of the response must be `{` and the last must be `}`.\n"
        "- Return JSON only."
    )


def build_visible_generation_prefix(thinking_mode: str, relaxed_visible_reasoning: bool = False) -> str:
    if not should_emit_thinking(thinking_mode):
        return ""
    if relaxed_visible_reasoning:
        return ""
    return "<think>\n"


def build_effective_prompt_template(
    prompt_template: str,
    thinking_mode: str,
    relaxed_visible_reasoning: bool = False,
) -> str:
    rewritten = strip_mode_sensitive_prompt_lines(prompt_template)
    mode_suffix = build_reasoning_mode_suffix(
        thinking_mode,
        relaxed_visible_reasoning=relaxed_visible_reasoning,
    )
    if not rewritten:
        return mode_suffix
    return f"{rewritten}\n\n{mode_suffix}"


def build_structured_user_prompt(prompt_input: dict[str, Any], user_prefix: str = "") -> str:
    return f"{user_prefix}Sample input JSON:\n{json.dumps(prompt_input, ensure_ascii=False, indent=2)}"


def open_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def build_sample_record(dataset: Any, index: int) -> dict[str, Any]:
    outfit, need_to_modify = dataset._resolve_sample(index)
    original_items = list(outfit.item_records)

    if need_to_modify:
        rng = dataset._build_rng(index)
        negative_index, _, negative_item, _ = dataset._select_negative_replacement(
            source_outfit=outfit,
            original_items=original_items,
            rng=rng,
        )
        input_items = list(original_items)
        input_items[negative_index] = negative_item
        mod_index = negative_index
    else:
        input_items = original_items
        mod_index = None

    image_paths = [item.image_path.resolve() for item in input_items]
    prompt_input = {
        "images": [f"image_{item_index}" for item_index in range(len(image_paths))],
        "candidate_style": dataset.outfit_style_candidates,
        "candidate_season": dataset.outfit_season_candidates,
        "candidate_occasion": dataset.outfit_occasion_candidates,
    }
    gold = {
        "outfit_summary": outfit.outfit_summary,
        "outfit_style": outfit.outfit_style,
        "season": outfit.season,
        "occasion": outfit.occasion,
        "need_to_modify": int(need_to_modify),
        "mod_index": format_mod_index(mod_index),
    }
    return {
        "index": index,
        "outfit_id": outfit.outfit_id,
        "prompt_input": prompt_input,
        "image_paths": image_paths,
        "gold": gold,
    }


def build_qwen_prompt(
    prompt_template: str,
    prompt_input: dict[str, Any],
    image_count: int,
    thinking_mode: str,
    model_name: str,
    model_family: str,
) -> str:
    emit_thinking = should_emit_thinking(thinking_mode)
    relaxed_visible_reasoning = should_use_relaxed_visible_reasoning(model_name, model_family, thinking_mode)
    generation_prefix = build_visible_generation_prefix(
        thinking_mode,
        relaxed_visible_reasoning=relaxed_visible_reasoning,
    )
    system_prompt = build_effective_prompt_template(
        prompt_template,
        thinking_mode,
        relaxed_visible_reasoning=relaxed_visible_reasoning,
    )
    user_prefix = "/think\n" if emit_thinking and not relaxed_visible_reasoning else ""
    if emit_thinking:
        if relaxed_visible_reasoning:
            system_prompt = (
                f"{system_prompt}\n\n"
                "Show a short visible reasoning preamble before the final JSON object. "
                "Do not use `<tool_call>` or any other tool wrapper. "
                "You may use plain text or a single `<think>...</think>` block for the reasoning, "
                "but `<think>` is not required. "
                "Do not omit the final JSON object."
            )
        else:
            system_prompt = (
                f"{system_prompt}\n\n"
                "You must first output a single `<think>...</think>` block containing your reasoning, "
                "and then output the final JSON object. "
                "Do not omit the JSON object. Do not place the JSON inside `<think>`."
            )
    user_prompt = build_structured_user_prompt(prompt_input, user_prefix=user_prefix)
    image_tokens = "\n".join(QWEN_IMAGE_PLACEHOLDER for _ in range(image_count))
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{image_tokens}\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{generation_prefix}"
    )


def render_processor_chat_template(
    processor: Any,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = True,
) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
    except TypeError:
        return processor.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)


def build_gemma_messages(
    prompt_template: str,
    prompt_input: dict[str, Any],
    image_count: int,
    thinking_mode: str,
) -> list[dict[str, Any]]:
    system_prompt = build_effective_prompt_template(prompt_template, thinking_mode)
    user_prompt = build_structured_user_prompt(prompt_input)
    merged_user_text = merge_system_prompt_into_user_text(system_prompt, user_prompt)
    user_content = [{"type": "image"} for _ in range(image_count)]
    user_content.append({"type": "text", "text": merged_user_text})
    return [{"role": "user", "content": user_content}]


def build_gemma_prompt(
    prompt_template: str,
    prompt_input: dict[str, Any],
    image_count: int,
    thinking_mode: str,
    prompt_processor: Any | None = None,
) -> str:
    generation_prefix = build_visible_generation_prefix(thinking_mode)
    messages = build_gemma_messages(prompt_template, prompt_input, image_count, thinking_mode)
    if prompt_processor is not None:
        rendered = render_processor_chat_template(prompt_processor, messages, add_generation_prompt=True)
        return f"{rendered}{generation_prefix}"

    user_text = messages[0]["content"][-1]["text"]
    image_tokens = "\n".join("<start_of_image>" for _ in range(image_count))
    if image_tokens:
        user_text = f"{image_tokens}\n{user_text}"
    return (
        f"<start_of_turn>user\n{user_text}<end_of_turn>\n"
        f"<start_of_turn>model\n{generation_prefix}"
    )


def build_model_prompt(
    prompt_template: str,
    prompt_input: dict[str, Any],
    image_count: int,
    model_name: str,
    model_family: str,
    thinking_mode: str,
    prompt_processor: Any | None = None,
) -> str:
    if model_family == "gemma_vl":
        return build_gemma_prompt(
            prompt_template,
            prompt_input,
            image_count,
            thinking_mode,
            prompt_processor=prompt_processor,
        )
    return build_qwen_prompt(prompt_template, prompt_input, image_count, thinking_mode, model_name, model_family)


def build_text_only_model_prompt(system_prompt: str, user_prompt: str, model_family: str) -> str:
    if model_family == "gemma_vl":
        merged_user_text = merge_system_prompt_into_user_text(system_prompt, user_prompt)
        return (
            f"<start_of_turn>user\n{merged_user_text}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_json_repair_prompt(
    raw_text: str,
    prompt_input: dict[str, Any],
    model_family: str,
) -> str:
    system_prompt = (
        "You are repairing a previous draft response for a fashion outfit evaluation task. "
        "Return exactly one complete JSON object and nothing else. "
        "Do not output `<think>`, markdown, tool wrappers, or explanations."
    )
    user_prompt = (
        "Extract the final answer from the previous draft.\n"
        "Rules:\n"
        "- Output exactly one JSON object with keys `outfit_summary`, `outfit_style`, `season`, `occasion`, `need_to_modify`, `mod_index`.\n"
        "- `outfit_style` must be chosen from `candidate_style` exactly.\n"
        "- `season` must be chosen from `candidate_season` exactly.\n"
        "- `occasion` must be chosen from `candidate_occasion` exactly.\n"
        "- If the draft is ambiguous, follow the last explicit final decision stated in the draft.\n"
        "- If the draft says there is no obvious mismatch, use `need_to_modify = 0` and `mod_index = \"NONE\"`.\n"
        "- If the draft identifies a mismatched item by ordinal wording such as `the third item`, convert it to a 0-based index.\n"
        "- Write `outfit_summary` in English as one short sentence ending with a period.\n\n"
        f"candidate_style = {json.dumps(prompt_input.get('candidate_style', []), ensure_ascii=False)}\n\n"
        f"candidate_season = {json.dumps(prompt_input.get('candidate_season', []), ensure_ascii=False)}\n\n"
        f"candidate_occasion = {json.dumps(prompt_input.get('candidate_occasion', []), ensure_ascii=False)}\n\n"
        "Previous draft:\n"
        "```text\n"
        f"{raw_text}\n"
        "```"
    )
    return build_text_only_model_prompt(system_prompt, user_prompt, model_family)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def extract_reasoning_block(text: str) -> str | None:
    matches = [match.strip() for match in THINK_BLOCK_RE.findall(text) if match.strip()]
    if matches:
        return "\n\n".join(matches)
    stripped = text.strip()
    start_tag = "<think>"
    end_tag = "</think>"
    start_index = stripped.lower().find(start_tag)
    if start_index == -1:
        return None
    end_index = stripped.lower().find(end_tag, start_index + len(start_tag))
    if end_index == -1:
        return stripped[start_index + len(start_tag) :].strip() or None
    return stripped[start_index + len(start_tag) : end_index].strip() or None


def extract_prefinal_reasoning_text(text: str) -> str | None:
    cleaned = strip_code_fences(text)
    cleaned = INLINE_TOOL_TAG_RE.sub("\n", cleaned)
    cleaned = cleaned.replace("```", "\n")
    cleaned = strip_tool_markup(cleaned)
    if not cleaned:
        return None

    json_candidate = extract_first_json_object_text(cleaned)
    if json_candidate is not None:
        prefix = cleaned[: cleaned.find(json_candidate)].strip()
        return prefix or None

    lines = [
        line.strip()
        for line in cleaned.splitlines()
        if line.strip() and line.strip() not in {"{", "}", "[", "]"}
    ]
    if not lines:
        return None

    for index, line in enumerate(lines):
        if KEY_VALUE_LINE_RE.match(line):
            prefix = "\n".join(lines[:index]).strip()
            return prefix or None
    return None


def strip_reasoning_block(text: str) -> str:
    stripped = THINK_BLOCK_RE.sub("", text).strip()
    if stripped.lower().startswith("<think>"):
        closing_tag = "</think>"
        closing_index = stripped.lower().find(closing_tag)
        if closing_index != -1:
            stripped = stripped[closing_index + len(closing_tag) :].strip()
    return stripped


def strip_tool_markup(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if XML_TAG_LINE_RE.match(stripped):
            continue
        if MARKDOWN_FENCE_LINE_RE.match(stripped):
            continue
        if stripped.lower() == "json":
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def strip_outer_quotes(value: str) -> str:
    stripped = value.strip()
    if stripped.endswith(","):
        stripped = stripped[:-1].rstrip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1].strip()
    return stripped


def choose_summary_line(summary_lines: list[str]) -> str:
    if not summary_lines:
        return ""
    for line in reversed(summary_lines):
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"[A-Za-z]", stripped) and re.search(r"[.!?]$", stripped):
            return stripped
    for line in reversed(summary_lines):
        stripped = line.strip()
        if stripped and re.search(r"[A-Za-z]", stripped):
            return stripped
    return summary_lines[-1].strip()


def clean_model_output_text(text: str) -> str:
    cleaned = strip_code_fences(text)
    cleaned = strip_reasoning_block(cleaned)
    cleaned = INLINE_TOOL_TAG_RE.sub("\n", cleaned)
    cleaned = cleaned.replace("```", "\n")
    return strip_tool_markup(cleaned)


def extract_first_json_object_text(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_line_structured_output(text: str) -> dict[str, Any] | None:
    cleaned = clean_model_output_text(text)
    lines = [
        line.strip()
        for line in cleaned.splitlines()
        if line.strip() and line.strip() not in {"{", "}", "[", "]"}
    ]
    if len(lines) < 6:
        return None

    mod_index_line = strip_outer_quotes(lines[-1])
    need_to_modify_line = strip_outer_quotes(lines[-2])
    occasion_line = strip_outer_quotes(lines[-3])
    season_line = strip_outer_quotes(lines[-4])
    outfit_style_line = strip_outer_quotes(lines[-5])
    outfit_summary_line = " ".join(line.strip() for line in lines[:-5]).strip()
    outfit_summary_line = strip_outer_quotes(outfit_summary_line)

    if not outfit_summary_line or not outfit_style_line or not season_line or not occasion_line:
        return None
    if need_to_modify_line not in {"0", "1"}:
        return None

    return {
        "outfit_summary": outfit_summary_line,
        "outfit_style": outfit_style_line,
        "season": season_line,
        "occasion": occasion_line,
        "need_to_modify": int(need_to_modify_line),
        "mod_index": mod_index_line,
    }


def parse_key_value_structured_output(text: str) -> dict[str, Any] | None:
    cleaned = clean_model_output_text(text)
    return parse_key_value_structured_text(cleaned)


def parse_key_value_structured_text(cleaned: str) -> dict[str, Any] | None:
    parsed: dict[str, Any] = {}
    summary_lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped or stripped in {"{", "}", "[", "]"}:
            continue
        match = KEY_VALUE_LINE_RE.match(stripped)
        if match is None:
            summary_lines.append(stripped)
            continue
        key = match.group("key").lower()
        value = strip_outer_quotes(match.group("value"))
        parsed[key] = value

    if "outfit_summary" not in parsed and summary_lines:
        parsed["outfit_summary"] = choose_summary_line(summary_lines)

    if not {"outfit_summary", "outfit_style", "season", "occasion", "need_to_modify"} <= parsed.keys():
        return None

    need_to_modify = str(parsed["need_to_modify"]).strip()
    if need_to_modify not in {"0", "1"}:
        return None
    parsed["need_to_modify"] = int(need_to_modify)

    if "mod_index" not in parsed or not str(parsed["mod_index"]).strip():
        if parsed["need_to_modify"] == 0:
            parsed["mod_index"] = "NONE"
        else:
            return None
    return parsed

def extract_json_dict(text: str, prompt_input: dict[str, Any] | None = None) -> dict[str, Any]:
    stripped = clean_model_output_text(text)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        json_candidate = extract_first_json_object_text(stripped)
        if json_candidate is not None:
            parsed = json.loads(json_candidate)
        else:
            fallback = parse_key_value_structured_output(stripped)
            if fallback is not None:
                parsed = fallback
            else:
                fallback = parse_line_structured_output(stripped)
                if fallback is None:
                    raise
                parsed = fallback

    if not isinstance(parsed, dict):
        raise ValueError("Model output is not a JSON object.")
    return parsed


def normalize_mod_index(value: Any) -> Any:
    if value is None:
        return "NONE"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.upper() == "NONE":
            return "NONE"
        if stripped.isdigit():
            return int(stripped)
        return stripped
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return value


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "outfit_summary": str(prediction.get("outfit_summary", "")).strip(),
        "outfit_style": str(prediction.get("outfit_style", "")).strip(),
        "season": str(prediction.get("season", "")).strip(),
        "occasion": str(prediction.get("occasion", "")).strip(),
        "need_to_modify": int(prediction.get("need_to_modify", -1)),
        "mod_index": normalize_mod_index(prediction.get("mod_index")),
    }


def score_prediction(prediction: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    style_correct = prediction["outfit_style"] == gold["outfit_style"]
    season_correct = prediction["season"] == gold["season"]
    occasion_correct = prediction["occasion"] == gold["occasion"]
    need_correct = prediction["need_to_modify"] == gold["need_to_modify"]
    mod_correct = prediction["mod_index"] == gold["mod_index"]
    return {
        "style_correct": style_correct,
        "season_correct": season_correct,
        "occasion_correct": occasion_correct,
        "need_to_modify_correct": need_correct,
        "mod_index_correct": mod_correct,
        "strict_correct": style_correct and season_correct and occasion_correct and need_correct and mod_correct,
        "summary_non_empty": bool(prediction["outfit_summary"]),
    }


def build_result_row(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": sample["index"],
        "outfit_id": sample["outfit_id"],
        "prompt_input": sample["prompt_input"],
        "image_paths": [str(path) for path in sample["image_paths"]],
        "gold": sample["gold"],
        "json_valid": False,
        "prediction": None,
        "metrics": {},
        "reasoning": None,
        "raw_response": None,
        "error": None,
        "stage": "build_prompt",
    }


def build_efficiency_state(prompt_text: str, image_count: int) -> dict[str, Any]:
    return {
        "attempt_count": 0,
        "repair_attempted": False,
        "image_count": int(image_count),
        "prompt_characters": len(prompt_text),
        "response_characters": 0,
        "load_seconds": 0.0,
        "generation_seconds": 0.0,
        "generation_seconds_estimate": 0.0,
        "parse_seconds": 0.0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "token_count_complete": True,
        "token_count_sources": [],
    }


def count_sequence_items(value: Any) -> int | None:
    if isinstance(value, (str, bytes, bytearray)):
        return None
    if isinstance(value, Sequence):
        return len(value)
    try:
        return len(value)
    except Exception:  # noqa: BLE001
        return None


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def resolve_prompt_tokenizer(llm: Any, prompt_processor: Any | None = None) -> Any | None:
    if prompt_processor is not None:
        tokenizer = getattr(prompt_processor, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        return prompt_processor
    getter = getattr(llm, "get_tokenizer", None)
    if callable(getter):
        try:
            return getter()
        except Exception:  # noqa: BLE001
            return None
    return None


def tokenize_text_length(tokenizer: Any | None, text: str) -> int | None:
    if tokenizer is None or not text:
        return None
    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except TypeError:
        try:
            encoded = tokenizer(text)
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None

    if isinstance(encoded, dict):
        token_ids = encoded.get("input_ids")
    else:
        token_ids = getattr(encoded, "input_ids", None)
    token_count = count_sequence_items(token_ids)
    if token_count == 0:
        return 0
    if isinstance(token_ids, Sequence):
        try:
            first_item = token_ids[0]
        except Exception:  # noqa: BLE001
            first_item = None
        if first_item is not None and not isinstance(first_item, int):
            token_ids = first_item
    return count_sequence_items(token_ids)


def extract_prompt_token_count(output: Any) -> int | None:
    for attribute in ("prompt_token_ids", "encoder_prompt_token_ids"):
        count = count_sequence_items(getattr(output, attribute, None))
        if count is not None:
            return count
    for attribute in ("num_prompt_tokens", "prompt_tokens", "prompt_token_count"):
        value = coerce_int(getattr(output, attribute, None))
        if value is not None:
            return value
    metrics = getattr(output, "metrics", None)
    if metrics is not None:
        for attribute in ("num_prompt_tokens", "prompt_tokens", "prompt_token_count"):
            value = coerce_int(getattr(metrics, attribute, None))
            if value is not None:
                return value
            if isinstance(metrics, dict):
                value = coerce_int(metrics.get(attribute))
                if value is not None:
                    return value
    return None


def extract_completion_token_count(output: Any) -> int | None:
    candidates = getattr(output, "outputs", None)
    if not candidates:
        return None
    candidate = candidates[0]
    for attribute in ("token_ids", "output_token_ids"):
        count = count_sequence_items(getattr(candidate, attribute, None))
        if count is not None:
            return count
    for attribute in ("num_output_tokens", "completion_tokens", "output_token_count"):
        value = coerce_int(getattr(candidate, attribute, None))
        if value is not None:
            return value
    return None


def merge_optional_token_count(
    efficiency_state: dict[str, Any],
    key: str,
    value: int | None,
) -> None:
    if value is None:
        efficiency_state["token_count_complete"] = False
        return
    existing = efficiency_state.get(key)
    if existing is None:
        efficiency_state[key] = int(value)
    else:
        efficiency_state[key] = int(existing) + int(value)


def append_token_count_source(efficiency_state: dict[str, Any], source: str) -> None:
    sources = efficiency_state.setdefault("token_count_sources", [])
    if source not in sources:
        sources.append(source)


def record_generation_attempt(
    efficiency_state: dict[str, Any],
    output: Any,
    prompt_text: str,
    raw_text: str,
    tokenizer: Any | None,
    generation_seconds: float,
    batch_request_count: int,
) -> None:
    efficiency_state["attempt_count"] += 1
    efficiency_state["generation_seconds"] += generation_seconds
    effective_batch_size = max(1, batch_request_count)
    efficiency_state["generation_seconds_estimate"] += generation_seconds / effective_batch_size
    efficiency_state["response_characters"] += len(raw_text)

    prompt_tokens = extract_prompt_token_count(output)
    completion_tokens = extract_completion_token_count(output)
    token_source = None
    if prompt_tokens is None:
        prompt_tokens = tokenize_text_length(tokenizer, prompt_text)
        if prompt_tokens is not None:
            token_source = "tokenizer_estimate"
    if completion_tokens is None:
        completion_tokens = tokenize_text_length(tokenizer, raw_text)
        if completion_tokens is not None:
            token_source = "tokenizer_estimate"
    if token_source is None and (prompt_tokens is not None or completion_tokens is not None):
        token_source = "request_output"
    if token_source is not None:
        append_token_count_source(efficiency_state, token_source)

    merge_optional_token_count(efficiency_state, "prompt_tokens", prompt_tokens)
    merge_optional_token_count(efficiency_state, "completion_tokens", completion_tokens)
    prompt_total = efficiency_state.get("prompt_tokens")
    completion_total = efficiency_state.get("completion_tokens")
    if prompt_total is not None and completion_total is not None:
        efficiency_state["total_tokens"] = int(prompt_total) + int(completion_total)
    else:
        efficiency_state["total_tokens"] = None


def finalize_efficiency_state(efficiency_state: dict[str, Any], total_seconds: float) -> dict[str, Any]:
    finalized = dict(efficiency_state)
    finalized["load_seconds"] = round(float(finalized.get("load_seconds", 0.0)), 4)
    finalized["generation_seconds"] = round(float(finalized.get("generation_seconds", 0.0)), 4)
    finalized["generation_seconds_estimate"] = round(float(finalized.get("generation_seconds_estimate", 0.0)), 4)
    finalized["parse_seconds"] = round(float(finalized.get("parse_seconds", 0.0)), 4)
    finalized["total_seconds"] = round(float(total_seconds), 4)
    finalized["total_seconds_estimate"] = round(
        finalized["load_seconds"] + finalized["generation_seconds_estimate"] + finalized["parse_seconds"],
        4,
    )
    finalized["token_count_sources"] = list(finalized.get("token_count_sources", []))
    return finalized


def finalize_row_from_raw_text(row: dict[str, Any], sample: dict[str, Any], raw_text: str) -> None:
    row["stage"] = "parse_output"
    row["raw_response"] = raw_text
    row["reasoning"] = extract_reasoning_block(raw_text) or extract_prefinal_reasoning_text(raw_text)
    parsed = extract_json_dict(raw_text, sample["prompt_input"])
    prediction = normalize_prediction(parsed)
    row["prediction"] = prediction
    row["json_valid"] = True
    row["stage"] = "score"
    row["metrics"] = score_prediction(prediction, sample["gold"])
    row["stage"] = "completed"


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize_path_token(value: str) -> str:
    sanitized = value.strip().replace("\\", "/").rstrip("/")
    if "/" in sanitized:
        sanitized = sanitized.split("/")[-1]
    sanitized = sanitized.replace(" ", "_")
    sanitized = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in sanitized)
    return sanitized or "unknown"


def resolve_model_path(model_name: str) -> Path:
    return Path(model_name).expanduser()


def is_local_model_path(model_name: str) -> bool:
    return resolve_model_path(model_name).exists()


def build_alias_lookup() -> dict[str, dict[str, str]]:
    alias_lookup: dict[str, dict[str, str]] = {}
    for spec in MODEL_SPECS.values():
        keys = (spec["repo_id"], spec["local_dir_name"], *spec["aliases"])
        for key in keys:
            alias_lookup[key] = spec
    return alias_lookup


MODEL_ALIAS_LOOKUP = build_alias_lookup()


def infer_model_family(model_name: str) -> str:
    spec = MODEL_ALIAS_LOOKUP.get(model_name)
    if spec is not None:
        return spec["family"]

    normalized = model_name.strip().lower()
    if "gemma-3-4b" in normalized or "gemma3-4b" in normalized:
        return "gemma_vl"
    if "qwen" in normalized:
        return "qwen_vl"
    return "unknown"


def resolve_requested_model(
    parser: argparse.ArgumentParser,
    requested_model: str,
    models_dir: Path,
    allow_hf_download: bool,
) -> tuple[str, str, str]:
    spec = MODEL_ALIAS_LOOKUP.get(requested_model)
    if spec is not None:
        local_path = (models_dir / spec["local_dir_name"]).expanduser()
        if local_path.exists():
            return str(local_path.resolve()), spec["local_dir_name"], spec["family"]
        if allow_hf_download:
            return spec["repo_id"], spec["local_dir_name"], spec["family"]
        parser.error(
            f"Local model directory not found: {local_path}. "
            "Run `python3 download.py` first, or pass --allow-hf-download to let vLLM fetch from Hugging Face."
        )

    model_path = resolve_model_path(requested_model)
    if model_path.exists():
        resolved_path = model_path.resolve()
        return str(resolved_path), resolved_path.name, infer_model_family(resolved_path.name)

    if allow_hf_download:
        return requested_model, sanitize_path_token(requested_model), infer_model_family(requested_model)

    parser.error(
        f"Model path does not exist: {model_path}. "
        "Pass a local directory under models/, or use --allow-hf-download."
    )


def build_metrics_path(results_path: Path) -> Path:
    if results_path.suffix == ".json" and results_path.stem.endswith("_results"):
        return results_path.with_name(f"{results_path.stem[:-8]}_eval.json")
    if results_path.suffix == ".json":
        return results_path.with_name(f"{results_path.stem}_eval.json")
    return results_path.with_name(f"{results_path.name}_eval.json")


def resolve_output_path(
    results_dir: Path,
    model_name: str,
    setting: str,
    split: str,
    output: str | None,
) -> Path:
    if output:
        return Path(output)
    model_token = sanitize_path_token(model_name)
    setting_token = sanitize_path_token(setting)
    split_token = sanitize_path_token(split)
    return results_dir / f"{model_token}_{setting_token}_{split_token}_results.json"


def parse_cuda_visible_devices() -> list[str]:
    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw_value:
        return []
    return [token.strip() for token in raw_value.split(",") if token.strip()]


def query_gpu_uuid_to_index_map() -> dict[str, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    mapping: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        if len(parts) != 2:
            continue
        index, uuid = parts
        mapping[uuid] = index
    return mapping


def normalize_cuda_visible_devices(parser: argparse.ArgumentParser) -> None:
    tokens = parse_cuda_visible_devices()
    if not tokens:
        return
    if not any(token.startswith(("GPU-", "MIG-")) for token in tokens):
        return
    if any(token.startswith("MIG-") for token in tokens):
        parser.error(
            "CUDA_VISIBLE_DEVICES is using MIG UUIDs. This vLLM environment expects numeric GPU indices here. "
            "Please convert it to numeric ids manually before running evaluate.py."
        )

    try:
        uuid_to_index = query_gpu_uuid_to_index_map()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        parser.error(
            f"Failed to query GPU index mapping with nvidia-smi while normalizing CUDA_VISIBLE_DEVICES: {exc}"
        )

    try:
        normalized = ",".join(uuid_to_index[token] if token.startswith("GPU-") else token for token in tokens)
    except KeyError as exc:
        parser.error(
            f"Could not map GPU UUID {exc.args[0]!r} to a numeric index. "
            "Please run `nvidia-smi --query-gpu=index,uuid --format=csv,noheader` and set CUDA_VISIBLE_DEVICES manually."
        )

    original = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = normalized
    print(
        f"[step 1/7] normalized CUDA_VISIBLE_DEVICES from {original!r} to {normalized!r}",
        flush=True,
    )


def print_run_config(
    args: argparse.Namespace,
    output_path: Path,
    metrics_path: Path,
    resolved_model: str,
    resolved_model_name: str,
    model_family: str,
) -> None:
    config = {
        "requested_model": args.model,
        "resolved_model": resolved_model,
        "resolved_model_name": resolved_model_name,
        "model_family": model_family,
        "model_is_local_path": Path(resolved_model).exists(),
        "models_dir": str(Path(args.models_dir).expanduser()),
        "root": args.root,
        "prompt_file": args.prompt_file,
        "sample_mode": args.sample_mode,
        "split": args.split,
        "split_seed": args.split_seed,
        "setting": args.setting,
        "allow_hf_download": args.allow_hf_download,
        "start_index": args.start_index,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "thinking_mode": args.thinking_mode,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "output_path": str(output_path),
        "metrics_path": str(metrics_path),
    }
    print("[config] evaluation configuration:", flush=True)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


def validate_model_name(parser: argparse.ArgumentParser, model_name: str) -> None:
    message = INVALID_MODEL_HINTS.get(model_name)
    if message:
        parser.error(message)


def validate_model_runtime_requirements(
    parser: argparse.ArgumentParser,
    model_family: str,
    trust_remote_code: bool,
) -> None:
    _ = parser
    _ = model_family
    _ = trust_remote_code


def validate_setting_split(parser: argparse.ArgumentParser, setting: str, split: str) -> None:
    if setting == "zero-shot" and split != "test":
        parser.error("Zero-shot evaluation must run on the test split. Use `--split test`.")


def resolve_effective_max_tokens(max_tokens: int, thinking_mode: str, model_family: str = "unknown") -> int:
    if not should_emit_thinking(thinking_mode):
        return max_tokens
    min_visible_max_tokens = VISIBLE_REASONING_MIN_MAX_TOKENS.get(model_family)
    if min_visible_max_tokens is not None and max_tokens < min_visible_max_tokens:
        return min_visible_max_tokens
    if max_tokens < 1024:
        return 1024
    return max_tokens


def resolve_effective_max_model_len(max_model_len: int, model_family: str, thinking_mode: str = "hidden") -> int:
    resolved = max(max_model_len, MODEL_MIN_MAX_MODEL_LEN.get(model_family, max_model_len))
    if should_emit_thinking(thinking_mode):
        resolved = max(resolved, VISIBLE_REASONING_MIN_MAX_MODEL_LEN.get(model_family, resolved))
    return resolved


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "modified_total": 0,
            "json_valid_rate": 0.0,
            "style_accuracy": 0.0,
            "season_accuracy": 0.0,
            "occasion_accuracy": 0.0,
            "need_to_modify_accuracy": 0.0,
            "mod_index_accuracy": 0.0,
            "mod_index_accuracy_on_modified_only": None,
            "strict_accuracy": 0.0,
        }

    def mean(key: str) -> float:
        return sum(int(row["metrics"].get(key, False)) for row in rows) / total

    modified_rows = [row for row in rows if row.get("gold", {}).get("need_to_modify") == 1]
    modified_total = len(modified_rows)
    mod_index_accuracy_on_modified_only = (
        sum(int(row["metrics"].get("mod_index_correct", False)) for row in modified_rows) / modified_total
        if modified_total > 0
        else None
    )
    json_valid_rate = sum(int(row["json_valid"]) for row in rows) / total
    return {
        "total": total,
        "modified_total": modified_total,
        "json_valid_rate": json_valid_rate,
        "style_accuracy": mean("style_correct"),
        "season_accuracy": mean("season_correct"),
        "occasion_accuracy": mean("occasion_correct"),
        "need_to_modify_accuracy": mean("need_to_modify_correct"),
        "mod_index_accuracy": mean("mod_index_correct"),
        "mod_index_accuracy_on_modified_only": mod_index_accuracy_on_modified_only,
        "strict_accuracy": mean("strict_correct"),
    }


def summarize_efficiency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    efficiencies = [row.get("efficiency") for row in rows if isinstance(row.get("efficiency"), dict)]
    if not efficiencies:
        return {
            "rows_with_efficiency": 0,
            "rows_with_token_usage": 0,
            "total_seconds_sum": 0.0,
            "total_seconds_avg": None,
            "total_seconds_max": None,
            "total_seconds_estimate_sum": 0.0,
            "total_seconds_estimate_avg": None,
            "generation_seconds_sum": 0.0,
            "generation_seconds_avg": None,
            "generation_seconds_estimate_sum": 0.0,
            "generation_seconds_estimate_avg": None,
            "prompt_tokens_sum": None,
            "completion_tokens_sum": None,
            "total_tokens_sum": None,
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
    total_seconds_estimate_values = float_values("total_seconds_estimate")
    generation_seconds_values = float_values("generation_seconds")
    generation_seconds_estimate_values = float_values("generation_seconds_estimate")
    prompt_token_values = int_values("prompt_tokens")
    completion_token_values = int_values("completion_tokens")
    total_token_values = int_values("total_tokens")
    token_usage_rows = sum(
        1
        for item in efficiencies
        if any(isinstance(item.get(key), int) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
    )
    return {
        "rows_with_efficiency": len(efficiencies),
        "rows_with_token_usage": token_usage_rows,
        "total_seconds_sum": round(sum(total_seconds_values), 4),
        "total_seconds_avg": round(sum(total_seconds_values) / len(total_seconds_values), 4) if total_seconds_values else None,
        "total_seconds_max": round(max(total_seconds_values), 4) if total_seconds_values else None,
        "total_seconds_estimate_sum": round(sum(total_seconds_estimate_values), 4),
        "total_seconds_estimate_avg": round(sum(total_seconds_estimate_values) / len(total_seconds_estimate_values), 4)
        if total_seconds_estimate_values
        else None,
        "generation_seconds_sum": round(sum(generation_seconds_values), 4),
        "generation_seconds_avg": round(sum(generation_seconds_values) / len(generation_seconds_values), 4)
        if generation_seconds_values
        else None,
        "generation_seconds_estimate_sum": round(sum(generation_seconds_estimate_values), 4),
        "generation_seconds_estimate_avg": round(
            sum(generation_seconds_estimate_values) / len(generation_seconds_estimate_values),
            4,
        )
        if generation_seconds_estimate_values
        else None,
        "prompt_tokens_sum": sum(prompt_token_values) if prompt_token_values else None,
        "completion_tokens_sum": sum(completion_token_values) if completion_token_values else None,
        "total_tokens_sum": sum(total_token_values) if total_token_values else None,
        "avg_total_tokens_per_token_usage_row": round(sum(total_token_values) / len(total_token_values), 4)
        if total_token_values
        else None,
    }


def get_max_image_count(dataset: Any) -> int:
    return max(len(outfit.item_records) for outfit in dataset.outfits)


def validate_torch_cuda_compatibility(parser: argparse.ArgumentParser, torch_module: Any) -> None:
    if not torch_module.cuda.is_available():
        parser.error("CUDA is not available in the current PyTorch environment.")

    device_index = torch_module.cuda.current_device()
    device_name = torch_module.cuda.get_device_name(device_index)
    capability = torch_module.cuda.get_device_capability(device_index)
    capability_token = f"sm_{capability[0]}{capability[1]}"
    compute_token = f"compute_{capability[0]}{capability[1]}"
    arch_list = torch_module.cuda.get_arch_list()

    print(
        "[step 4/7] torch cuda preflight: "
        f"device={device_name} capability={capability[0]}.{capability[1]} "
        f"torch={torch_module.__version__} cuda={torch_module.version.cuda} "
        f"supported_arches={arch_list}",
        flush=True,
    )

    if capability_token in arch_list or compute_token in arch_list:
        return

    parser.error(
        "Current PyTorch CUDA build does not include kernels for this GPU. "
        f"Detected device={device_name} capability={capability[0]}.{capability[1]}, "
        f"but torch.cuda.get_arch_list()={arch_list}. "
        "On V100 (sm_70), use a PyTorch build that includes sm_70 and rebuild/reinstall vLLM against it, "
        "or move to a newer GPU such as T4/A100/H100."
    )


def extract_generation_text(output: Any) -> str:
    candidates = getattr(output, "outputs", None)
    if not candidates:
        raise ValueError("vLLM returned no candidate outputs.")
    text = getattr(candidates[0], "text", None)
    if not isinstance(text, str):
        raise ValueError("vLLM output text is missing.")
    return text


def should_retry_with_json_repair_prompt(
    raw_text: str,
    thinking_mode: str,
    model_family: str,
) -> bool:
    if thinking_mode != "visible":
        return False
    if model_family not in {"qwen_vl", "gemma_vl"}:
        return False
    if not raw_text or not raw_text.strip():
        return False
    cleaned = clean_model_output_text(raw_text)
    if not cleaned:
        return True
    return extract_first_json_object_text(cleaned) is None


def retry_row_with_json_repair_prompt(
    llm: Any,
    sampling_params: Any,
    sample: dict[str, Any],
    row: dict[str, Any],
    stage_prefix: str,
    model_family: str,
    tokenizer: Any | None,
    efficiency_state: dict[str, Any],
) -> None:
    row["raw_response_before_retry"] = row.get("raw_response")
    row["stage"] = "repair_generate"
    print(f"{stage_prefix} outfit_id={sample['outfit_id']} stage=repair_generate", flush=True)
    repair_prompt = build_json_repair_prompt(row["raw_response"], sample["prompt_input"], model_family)
    repair_request = {"prompt": repair_prompt}
    repair_generation_started = time.perf_counter()
    repair_output = llm.generate([repair_request], sampling_params=sampling_params)[0]
    repair_generation_seconds = time.perf_counter() - repair_generation_started
    repair_raw_text = extract_generation_text(repair_output)
    efficiency_state["repair_attempted"] = True
    record_generation_attempt(
        efficiency_state,
        repair_output,
        repair_prompt,
        repair_raw_text,
        tokenizer,
        repair_generation_seconds,
        batch_request_count=1,
    )
    row["stage"] = "repair_parse_output"
    print(f"{stage_prefix} outfit_id={sample['outfit_id']} stage=repair_parse_output", flush=True)
    parse_started = time.perf_counter()
    finalize_row_from_raw_text(row, sample, repair_raw_text)
    efficiency_state["parse_seconds"] += time.perf_counter() - parse_started
    row["repair_strategy"] = "text_json_repair_retry"


def build_argument_parser(defaults: RuntimeDefaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-VL, Qwen3-VL, and Gemma 3 zero-shot performance with the local vLLM Python package."
    )
    parser.add_argument(
        "--model",
        default=defaults.model,
        help=(
            f"Model alias, local path, or Hugging Face repo id. Default: {defaults.model}. "
            "Known aliases: qwen25vl-7b, qwen3vl-4b, qwen3vl-4b-thinking, "
            "qwen3vl-8b, qwen3vl-8b-thinking, unsloth-gemma3-4b."
        ),
    )
    parser.add_argument(
        "--models-dir",
        default=defaults.models_dir,
        help=f"Directory containing downloaded models. Default: {defaults.models_dir}",
    )
    parser.add_argument(
        "--root",
        default=defaults.dataset_root,
        help="Dataset root directory containing the Female, Male, and Child folders, each with look*.csv, label*.csv, and photos*.",
    )
    parser.add_argument("--prompt-file", default=defaults.prompt_file, help="Prompt template file.")
    parser.add_argument("--sample-mode", choices=("original", "modified", "both"), default=defaults.sample_mode)
    parser.add_argument(
        "--split",
        choices=SPLIT_CHOICES,
        default=defaults.split if defaults.split in SPLIT_CHOICES else "test",
        help="Dataset split to evaluate. Default: test",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=defaults.split_seed,
        help=f"Random seed used for the per-segment train/val/test split. Default: {defaults.split_seed}",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Starting sample index.")
    parser.add_argument(
        "--limit",
        type=int,
        default=defaults.limit,
        help=(
            "Number of samples to evaluate. "
            f"Default: {defaults.limit} (0 means evaluate all samples from start_index to the end)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.batch_size,
        help=f"Number of samples to send in one vLLM generate call. Default: {defaults.batch_size}.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=defaults.max_tokens,
        help=(
            f"Max output tokens per sample. Default: {defaults.max_tokens}. "
            "Visible thinking mode may automatically raise this to at least 12288 "
            "for Qwen-VL."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=defaults.temperature,
        help=f"Sampling temperature. Default: {defaults.temperature}. Recommended for Instruct evaluation: 0.3-0.7.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=defaults.top_k,
        help=f"Sampling top_k. Default: {defaults.top_k}.",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("hidden", "visible"),
        default=defaults.thinking_mode if defaults.thinking_mode in {"hidden", "visible"} else "visible",
        help=(
            "How to handle reasoning. "
            "Use `hidden` to keep reasoning internal and output JSON only; "
            "use `visible` to ask the model to output `<think>...</think>` before the final JSON."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=defaults.tensor_parallel_size,
        help=f"vLLM tensor_parallel_size. Default: {defaults.tensor_parallel_size}.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=defaults.gpu_memory_utilization,
        help=f"vLLM gpu_memory_utilization. Default: {defaults.gpu_memory_utilization}.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=defaults.max_model_len,
        help=(
            f"vLLM max_model_len. Default: {defaults.max_model_len}. "
            "Some multimodal models may automatically raise this to satisfy their visual token budget. "
            "Visible reasoning for Qwen-VL may automatically raise this to at least 32768."
        ),
    )
    parser.add_argument("--dtype", default=defaults.dtype, help=f"vLLM dtype, for example auto, bfloat16, float16. Default: {defaults.dtype}.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to vLLM.")
    parser.add_argument(
        "--allow-hf-download",
        action="store_true",
        help="Allow vLLM to download from Hugging Face when a local model directory is not found.",
    )
    parser.add_argument(
        "--setting",
        choices=("zero-shot", "few-shot", "lora", "fullft"),
        default=defaults.setting,
        help="Evaluation setting name used in the default results directory.",
    )
    parser.add_argument(
        "--output",
        default=defaults.output,
        help=(
            "Path to write prediction JSON. "
            f"Default: {defaults.results_dir}/<model_name>_<setting>_<split>_results.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run_started = time.perf_counter()
    print("[step 1/7] parsing arguments", flush=True)
    defaults = load_runtime_defaults()
    parser = build_argument_parser(defaults)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    validate_setting_split(parser, args.setting, args.split)
    normalize_cuda_visible_devices(parser)
    validate_model_name(parser, args.model)
    models_dir = Path(args.models_dir).expanduser()
    resolved_model, resolved_model_name, model_family = resolve_requested_model(
        parser=parser,
        requested_model=args.model,
        models_dir=models_dir,
        allow_hf_download=args.allow_hf_download,
    )
    requested_max_tokens = args.max_tokens
    args.max_tokens = resolve_effective_max_tokens(args.max_tokens, args.thinking_mode, model_family)
    requested_max_model_len = args.max_model_len
    args.max_model_len = resolve_effective_max_model_len(args.max_model_len, model_family, args.thinking_mode)
    validate_model_runtime_requirements(parser, model_family, args.trust_remote_code)
    output_path = resolve_output_path(defaults.results_dir, resolved_model_name, args.setting, args.split, args.output)
    metrics_path = build_metrics_path(output_path)
    print_run_config(args, output_path, metrics_path, resolved_model, resolved_model_name, model_family)
    if args.max_tokens != requested_max_tokens:
        print(
            f"[config] increased max_tokens from {requested_max_tokens} to {args.max_tokens} "
            f"for thinking_mode={args.thinking_mode} model_family={model_family}",
            flush=True,
        )
    if args.max_model_len != requested_max_model_len:
        print(
            f"[config] increased max_model_len from {requested_max_model_len} to {args.max_model_len} "
            f"for model_family={model_family}",
            flush=True,
        )
    if args.thinking_mode == "visible" and model_family in {"qwen_vl", "gemma_vl"}:
        print(
            "[config] visible reasoning outputs that miss or corrupt the final JSON "
            "will be retried once with a text-only JSON repair prompt.",
            flush=True,
        )

    OutfitNegativeSampleDataset = import_dataset_class()
    print("[step 2/7] loading dataset", flush=True)
    dataset = OutfitNegativeSampleDataset(
        root=args.root,
        transform=None,
        deterministic=True,
        sample_mode=args.sample_mode,
        split=args.split,
        split_seed=args.split_seed,
    )
    print(
        f"[step 2/7] dataset loaded: split={args.split} outfit_count={len(dataset.outfits)} "
        f"sample_count={len(dataset)} split_counts={dataset.split_outfit_counts_by_segment} "
        f"max_images_per_outfit={get_max_image_count(dataset)}",
        flush=True,
    )
    print("[step 3/7] loading prompt template", flush=True)
    prompt_template = load_prompt(Path(args.prompt_file))
    print(f"[step 3/7] prompt loaded: characters={len(prompt_template)}", flush=True)
    prompt_processor = None
    if model_family == "gemma_vl":
        print("[step 3/7] loading Gemma processor for prompt rendering", flush=True)
        AutoProcessor = import_transformers_auto_processor()
        prompt_processor = AutoProcessor.from_pretrained(
            resolved_model,
            trust_remote_code=args.trust_remote_code,
        )
        print("[step 3/7] Gemma processor ready", flush=True)
    requested_limit = len(dataset) - args.start_index if args.limit <= 0 else args.limit
    end_index = min(len(dataset), args.start_index + requested_limit)

    if end_index <= args.start_index:
        print("[step 4/7] no samples selected, writing empty outputs", flush=True)
        rows: list[dict[str, Any]] = []
        write_json(output_path, rows)
        metrics = summarize_metrics(rows)
        metrics["efficiency"] = summarize_efficiency(rows)
        metrics["run_seconds"] = round(time.perf_counter() - run_started, 4)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nMetrics:")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"\nPredictions written to: {output_path}")
        print(f"Metrics written to: {metrics_path}")
        return 0

    configure_vllm_worker_multiproc_method()
    torch_module = import_torch_module()
    validate_torch_cuda_compatibility(parser, torch_module)
    LLM, SamplingParams = import_vllm_components()
    print("[step 4/7] initializing vLLM engine", flush=True)
    llm = LLM(
        model=resolved_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        limit_mm_per_prompt={"image": get_max_image_count(dataset)},
    )
    print("[step 4/7] vLLM engine ready", flush=True)
    prompt_tokenizer = resolve_prompt_tokenizer(llm, prompt_processor)
    print("[step 5/7] building sampling params", flush=True)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    print("[step 5/7] sampling params ready", flush=True)
    print(
        f"[step 6/7] starting evaluation loop with model={resolved_model} batch_size={args.batch_size}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    total_samples = end_index - args.start_index
    batch_count = (total_samples + args.batch_size - 1) // args.batch_size
    indices = list(range(args.start_index, end_index))
    evaluation_started = time.perf_counter()
    for batch_number, batch_start in enumerate(range(0, total_samples, args.batch_size), start=1):
        batch_indices = indices[batch_start : batch_start + args.batch_size]
        batch_label = f"[batch {batch_number}/{batch_count}]"
        print(f"{batch_label} stage=build_requests size={len(batch_indices)}", flush=True)

        batch_rows: list[dict[str, Any]] = []
        batch_ready_requests: list[dict[str, Any]] = []
        batch_request_metas: list[dict[str, Any]] = []

        for index in batch_indices:
            sample_started = time.perf_counter()
            progress = index - args.start_index + 1
            stage_prefix = f"[sample {progress}/{total_samples}]"
            print(f"{stage_prefix} stage=build_sample", flush=True)
            sample = build_sample_record(dataset, index)
            print(
                f"{stage_prefix} outfit_id={sample['outfit_id']} stage=build_prompt image_count={len(sample['image_paths'])}",
                flush=True,
            )
            prompt = build_model_prompt(
                prompt_template=prompt_template,
                prompt_input=sample["prompt_input"],
                image_count=len(sample["image_paths"]),
                model_name=resolved_model_name,
                model_family=model_family,
                thinking_mode=args.thinking_mode,
                prompt_processor=prompt_processor,
            )

            row = build_result_row(sample)
            efficiency_state = build_efficiency_state(prompt, len(sample["image_paths"]))
            batch_rows.append(row)

            try:
                row["stage"] = "load_images"
                print(f"{stage_prefix} outfit_id={sample['outfit_id']} stage=load_images", flush=True)
                load_started = time.perf_counter()
                images = [open_rgb_image(path) for path in sample["image_paths"]]
                efficiency_state["load_seconds"] += time.perf_counter() - load_started
                row["stage"] = "generate"
                batch_ready_requests.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": images},
                    }
                )
                batch_request_metas.append(
                    {
                        "sample": sample,
                        "row": row,
                        "stage_prefix": stage_prefix,
                        "prompt": prompt,
                        "efficiency_state": efficiency_state,
                        "sample_started": sample_started,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{exc.__class__.__name__}: {exc}"
                row["efficiency"] = finalize_efficiency_state(
                    efficiency_state,
                    time.perf_counter() - sample_started,
                )

        if batch_ready_requests:
            print(f"{batch_label} stage=generate request_count={len(batch_ready_requests)}", flush=True)
            try:
                batch_generation_started = time.perf_counter()
                batch_outputs = llm.generate(batch_ready_requests, sampling_params=sampling_params)
                batch_generation_seconds = time.perf_counter() - batch_generation_started
            except Exception as batch_exc:  # noqa: BLE001
                print(
                    f"{batch_label} stage=generate status=fallback_to_single error={batch_exc.__class__.__name__}: {batch_exc}",
                    flush=True,
                )
                for request, meta in zip(batch_ready_requests, batch_request_metas, strict=True):
                    row = meta["row"]
                    stage_prefix = meta["stage_prefix"]
                    sample = meta["sample"]
                    prompt = meta["prompt"]
                    efficiency_state = meta["efficiency_state"]
                    sample_started = meta["sample_started"]
                    raw_text = ""
                    parse_started = None
                    try:
                        print(f"{stage_prefix} outfit_id={sample['outfit_id']} stage=generate_single", flush=True)
                        generation_started = time.perf_counter()
                        single_output = llm.generate([request], sampling_params=sampling_params)[0]
                        generation_seconds = time.perf_counter() - generation_started
                        print(f"{stage_prefix} outfit_id={sample['outfit_id']} stage=parse_output", flush=True)
                        raw_text = extract_generation_text(single_output)
                        record_generation_attempt(
                            efficiency_state,
                            single_output,
                            prompt,
                            raw_text,
                            prompt_tokenizer,
                            generation_seconds,
                            batch_request_count=1,
                        )
                        parse_started = time.perf_counter()
                        finalize_row_from_raw_text(row, sample, raw_text)
                        efficiency_state["parse_seconds"] += time.perf_counter() - parse_started
                    except Exception as exc:  # noqa: BLE001
                        if parse_started is not None:
                            efficiency_state["parse_seconds"] += time.perf_counter() - parse_started
                        if should_retry_with_json_repair_prompt(raw_text, args.thinking_mode, model_family):
                            try:
                                retry_row_with_json_repair_prompt(
                                    llm=llm,
                                    sampling_params=sampling_params,
                                    sample=sample,
                                    row=row,
                                    stage_prefix=stage_prefix,
                                    model_family=model_family,
                                    tokenizer=prompt_tokenizer,
                                    efficiency_state=efficiency_state,
                                )
                            except Exception as repair_exc:  # noqa: BLE001
                                row["error"] = f"{repair_exc.__class__.__name__}: {repair_exc}"
                        else:
                            row["error"] = f"{exc.__class__.__name__}: {exc}"
                    finally:
                        row["efficiency"] = finalize_efficiency_state(
                            efficiency_state,
                            time.perf_counter() - sample_started,
                        )
            else:
                for output, meta, request in zip(batch_outputs, batch_request_metas, batch_ready_requests, strict=True):
                    row = meta["row"]
                    stage_prefix = meta["stage_prefix"]
                    sample = meta["sample"]
                    prompt = meta["prompt"]
                    efficiency_state = meta["efficiency_state"]
                    sample_started = meta["sample_started"]
                    raw_text = ""
                    parse_started = None
                    try:
                        row["stage"] = "parse_output"
                        print(f"{stage_prefix} outfit_id={sample['outfit_id']} stage=parse_output", flush=True)
                        raw_text = extract_generation_text(output)
                        record_generation_attempt(
                            efficiency_state,
                            output,
                            prompt,
                            raw_text,
                            prompt_tokenizer,
                            batch_generation_seconds,
                            batch_request_count=len(batch_ready_requests),
                        )
                        parse_started = time.perf_counter()
                        finalize_row_from_raw_text(row, sample, raw_text)
                        efficiency_state["parse_seconds"] += time.perf_counter() - parse_started
                    except Exception as exc:  # noqa: BLE001
                        if parse_started is not None:
                            efficiency_state["parse_seconds"] += time.perf_counter() - parse_started
                        if should_retry_with_json_repair_prompt(raw_text, args.thinking_mode, model_family):
                            try:
                                retry_row_with_json_repair_prompt(
                                    llm=llm,
                                    sampling_params=sampling_params,
                                    sample=sample,
                                    row=row,
                                    stage_prefix=stage_prefix,
                                    model_family=model_family,
                                    tokenizer=prompt_tokenizer,
                                    efficiency_state=efficiency_state,
                                )
                            except Exception as repair_exc:  # noqa: BLE001
                                row["error"] = f"{repair_exc.__class__.__name__}: {repair_exc}"
                        else:
                            row["error"] = f"{exc.__class__.__name__}: {exc}"
                    finally:
                        row["efficiency"] = finalize_efficiency_state(
                            efficiency_state,
                            time.perf_counter() - sample_started,
                        )

        rows.extend(batch_rows)
        for row in batch_rows:
            progress = row["index"] - args.start_index + 1
            stage_prefix = f"[sample {progress}/{total_samples}]"
            print(
                f"{stage_prefix} outfit_id={row['outfit_id']} stage={row['stage']} "
                f"json_valid={row['json_valid']} error={row['error']}",
                flush=True,
            )

    print("[step 7/7] writing predictions and metrics", flush=True)
    write_json(output_path, rows)
    metrics = summarize_metrics(rows)
    metrics["efficiency"] = summarize_efficiency(rows)
    metrics["evaluation_wall_seconds"] = round(time.perf_counter() - evaluation_started, 4)
    metrics["run_seconds"] = round(time.perf_counter() - run_started, 4)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nMetrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nPredictions written to: {output_path}")
    print(f"Metrics written to: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
