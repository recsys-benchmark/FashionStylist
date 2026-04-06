#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import mimetypes
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    import Benchamark.Task3.mllm_eval as eval_utils
except ModuleNotFoundError:
    import mllm_eval as eval_utils
from task3_dataset import infer_audience_segment

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DATASET_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_GEMINI_API_VERSION = "v1beta"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_EVAL_SPLIT = "test"
TOKEN_USAGE_ALIASES = {
    "prompt_tokens": (
        "prompt_tokens",
        "input_tokens",
        "prompt_token_count",
        "input_token_count",
        "promptTokenCount",
    ),
    "completion_tokens": (
        "completion_tokens",
        "output_tokens",
        "completion_token_count",
        "output_token_count",
        "candidate_token_count",
        "candidates_token_count",
        "candidatesTokenCount",
    ),
    "total_tokens": (
        "total_tokens",
        "total_token_count",
        "totalTokenCount",
    ),
}


@dataclass(frozen=True)
class RuntimeDefaults:
    dataset_root: str
    prompt_file: str
    results_dir: Path
    api_model: str
    api_version: str
    api_base_url: str
    run_name: str
    output: str | None
    requests_path: str | None
    sample_manifest: str | None
    input_manifest: str | None
    local_api_config: str
    parallelism: int
    timeout: int
    retry_attempts: int
    retry_backoff_seconds: float
    split_seed: int
    thinking_mode: str
    temperature: float
    max_output_tokens: int
    media_resolution: str | None


def load_runtime_defaults() -> RuntimeDefaults:
    return RuntimeDefaults(
        dataset_root=os.environ.get(
            "FASHION_STYLIST_DATA_ROOT",
            os.environ.get("TASK3_DATASET_ROOT", str(SHARED_DATASET_ROOT)),
        ),
        prompt_file=os.environ.get("TASK3_PROMPT_FILE", str(SCRIPT_DIR / "template" / "prompt.txt")),
        results_dir=Path(os.environ.get("TASK3_RESULTS_DIR", str(SCRIPT_DIR / "results"))),
        api_model=os.environ.get("API_EVAL_MODEL", "").strip(),
        api_version=os.environ.get("API_EVAL_API_VERSION", "").strip().lower(),
        api_base_url=os.environ.get("API_EVAL_BASE_URL", "").strip().rstrip("/"),
        run_name=os.environ.get("API_EVAL_RUN_NAME", "closed_model"),
        output=os.environ.get("API_EVAL_OUTPUT") or None,
        requests_path=os.environ.get("API_EVAL_REQUESTS_PATH") or None,
        sample_manifest=os.environ.get("API_EVAL_SAMPLE_MANIFEST") or None,
        input_manifest=os.environ.get("API_EVAL_INPUT_MANIFEST") or None,
        local_api_config=os.environ.get("API_EVAL_LOCAL_CONFIG", str(SCRIPT_DIR / "template" / "api_clients_local.py")),
        parallelism=int(os.environ.get("API_EVAL_PARALLELISM", "1")),
        timeout=int(os.environ.get("API_EVAL_TIMEOUT_SECONDS", "120")),
        retry_attempts=int(os.environ.get("API_EVAL_RETRY_ATTEMPTS", "2")),
        retry_backoff_seconds=float(os.environ.get("API_EVAL_RETRY_BACKOFF_SECONDS", "5.0")),
        split_seed=int(os.environ.get("API_EVAL_SPLIT_SEED", "42")),
        thinking_mode=os.environ.get("API_EVAL_THINKING_MODE", "hidden").strip().lower() or "hidden",
        temperature=float(os.environ.get("API_EVAL_TEMPERATURE", "1.0")),
        max_output_tokens=int(os.environ.get("API_EVAL_MAX_OUTPUT_TOKENS", "2048")),
        media_resolution=os.environ.get("API_EVAL_MEDIA_RESOLUTION", "").strip().lower() or None,
    )

def resolve_output_path(
    results_dir: Path,
    run_name: str,
    split: str,
    output: str | None,
) -> Path:
    if output:
        return Path(output)
    run_token = eval_utils.sanitize_path_token(run_name)
    split_token = eval_utils.sanitize_path_token(split)
    return results_dir / f"{run_token}_zero-shot_{split_token}_results.json"


def resolve_requests_path(output_path: Path, requests_path: str | None) -> Path:
    if requests_path:
        return Path(requests_path)
    stem = output_path.stem
    if stem.endswith("_results"):
        stem = stem[:-8]
    return output_path.with_name(f"{stem}_requests.jsonl")


def resolve_sample_manifest_path(output_path: Path, sample_manifest: str | None) -> Path:
    if sample_manifest:
        return Path(sample_manifest)
    stem = output_path.stem
    if stem.endswith("_results"):
        stem = stem[:-8]
    return output_path.with_name(f"{stem}_sample_manifest.json")


def load_samples_from_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Sample manifest must be a JSON object: {manifest_path}")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise RuntimeError(f"Sample manifest is missing a 'samples' list: {manifest_path}")

    normalized_samples: list[dict[str, Any]] = []
    required_fields = ("index", "outfit_id", "prompt_input", "image_paths", "gold")
    for offset, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise RuntimeError(f"Sample manifest entry #{offset} is not a JSON object: {manifest_path}")
        missing = [field for field in required_fields if field not in sample]
        if missing:
            missing_text = ", ".join(missing)
            raise RuntimeError(f"Sample manifest entry #{offset} is missing fields [{missing_text}]: {manifest_path}")
        normalized_samples.append(
            {
                "index": int(sample["index"]),
                "outfit_id": str(sample["outfit_id"]),
                "prompt_input": sample["prompt_input"],
                "image_paths": [str(path) for path in sample["image_paths"]],
                "gold": sample["gold"],
            }
        )
    return payload, normalized_samples


def load_local_api_variables(config_path: Path) -> dict[str, str]:
    if not config_path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("api_clients_local", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load local API config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded: dict[str, str] = {}
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if not isinstance(value, (str, int, float, bool)):
            continue
        loaded[name] = str(value)
    return loaded


def apply_local_api_variables(config_path: Path) -> list[str]:
    loaded = load_local_api_variables(config_path)
    applied_names: list[str] = []
    for name, value in loaded.items():
        os.environ[name] = value
        applied_names.append(name)
    return sorted(applied_names)


def resolve_api_key() -> str:
    candidates: tuple[str, ...] = ("GEMINI_API_KEY",)
    for name in candidates:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(
        f"Missing Gemini API key. Set one of: {', '.join(candidates)}."
    )


def resolve_api_model(requested_model: str) -> str:
    if requested_model.strip():
        return requested_model.strip()
    return DEFAULT_GEMINI_MODEL


def resolve_api_version(requested_version: str) -> str:
    if requested_version.strip():
        return requested_version.strip().lower()
    return DEFAULT_GEMINI_API_VERSION


def resolve_api_base_url(requested_base_url: str) -> str:
    if requested_base_url.strip():
        return requested_base_url.strip().rstrip("/")
    return DEFAULT_GEMINI_BASE_URL


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


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def normalize_usage_payload(payload: Any) -> dict[str, int]:
    best: dict[str, int] = {}
    if isinstance(payload, dict):
        current: dict[str, int] = {}
        for target_key, aliases in TOKEN_USAGE_ALIASES.items():
            for alias in aliases:
                value = coerce_int(payload.get(alias))
                if value is not None:
                    current[target_key] = value
                    break
        if "total_tokens" not in current and {
            "prompt_tokens",
            "completion_tokens",
        } <= current.keys():
            current["total_tokens"] = current["prompt_tokens"] + current["completion_tokens"]
        best = current
        for nested_key in (
            "usage",
            "usage_metadata",
            "token_usage",
            "usageMetadata",
            "tokenUsage",
            "response_metadata",
            "responseMetadata",
            "metadata",
        ):
            nested = normalize_usage_payload(payload.get(nested_key))
            if len(nested) > len(best):
                best = nested
        if len(best) < 3:
            for nested_value in payload.values():
                nested = normalize_usage_payload(nested_value)
                if len(nested) > len(best):
                    best = nested
    elif isinstance(payload, list):
        for item in payload:
            nested = normalize_usage_payload(item)
            if len(nested) > len(best):
                best = nested
    return best


def build_api_efficiency(
    request_payload: dict[str, Any],
    raw_text: str,
    api_metadata: dict[str, Any] | None,
    total_seconds: float,
    parse_seconds: float,
) -> dict[str, Any]:
    api_metadata = api_metadata or {}
    usage = normalize_usage_payload(api_metadata.get("provider_payload"))
    latency_seconds = coerce_float(api_metadata.get("latency_seconds"))
    efficiency: dict[str, Any] = {
        "usage_source": "provider_payload" if usage else "unavailable",
        "latency_seconds": round(latency_seconds, 4) if latency_seconds is not None else None,
        "parse_seconds": round(parse_seconds, 4),
        "total_seconds": round(total_seconds, 4),
        "attempt_count": coerce_int(api_metadata.get("attempt_count")),
        "retry_count": coerce_int(api_metadata.get("retry_count")),
        "prompt_characters": len(request_payload.get("system_prompt", "")) + len(request_payload.get("user_prompt", "")),
        "response_characters": len(raw_text),
        "image_count": len(request_payload.get("image_paths", [])),
    }
    efficiency.update(usage)
    return efficiency


def summarize_efficiency(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    efficiencies = [row.get("efficiency") for row in rows if isinstance(row.get("efficiency"), dict)]
    if not efficiencies:
        return {
            "rows_with_efficiency": 0,
            "rows_with_token_usage": 0,
            "total_seconds_sum": 0.0,
            "total_seconds_avg": None,
            "total_seconds_max": None,
            "latency_seconds_sum": 0.0,
            "latency_seconds_avg": None,
            "latency_seconds_max": None,
            "prompt_tokens_sum": None,
            "completion_tokens_sum": None,
            "total_tokens_sum": None,
        }

    def numeric_values(key: str) -> list[float]:
        values: list[float] = []
        for item in efficiencies:
            value = coerce_float(item.get(key))
            if value is not None:
                values.append(value)
        return values

    def int_values(key: str) -> list[int]:
        values: list[int] = []
        for item in efficiencies:
            value = coerce_int(item.get(key))
            if value is not None:
                values.append(value)
        return values

    total_seconds_values = numeric_values("total_seconds")
    latency_seconds_values = numeric_values("latency_seconds")
    prompt_token_values = int_values("prompt_tokens")
    completion_token_values = int_values("completion_tokens")
    total_token_values = int_values("total_tokens")
    token_usage_rows = sum(
        1
        for item in efficiencies
        if any(coerce_int(item.get(key)) is not None for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
    )
    return {
        "rows_with_efficiency": len(efficiencies),
        "rows_with_token_usage": token_usage_rows,
        "total_seconds_sum": round(sum(total_seconds_values), 4),
        "total_seconds_avg": round(sum(total_seconds_values) / len(total_seconds_values), 4) if total_seconds_values else None,
        "total_seconds_max": round(max(total_seconds_values), 4) if total_seconds_values else None,
        "latency_seconds_sum": round(sum(latency_seconds_values), 4),
        "latency_seconds_avg": round(sum(latency_seconds_values) / len(latency_seconds_values), 4)
        if latency_seconds_values
        else None,
        "latency_seconds_max": round(max(latency_seconds_values), 4) if latency_seconds_values else None,
        "prompt_tokens_sum": sum(prompt_token_values) if prompt_token_values else None,
        "completion_tokens_sum": sum(completion_token_values) if completion_token_values else None,
        "total_tokens_sum": sum(total_token_values) if total_token_values else None,
        "avg_total_tokens_per_token_usage_row": round(sum(total_token_values) / len(total_token_values), 4)
        if total_token_values
        else None,
    }


def resolve_effective_max_output_tokens(max_output_tokens: int, thinking_mode: str) -> int:
    if thinking_mode == "visible" and max_output_tokens < 8192:
        return 8192
    if max_output_tokens < 256:
        return 256
    return max_output_tokens


def guess_image_mime_type(image_path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(image_path))
    if guessed:
        return guessed
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def build_gemini_user_parts(request_payload: dict[str, Any], media_resolution: str | None) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for image_path_str in request_payload.get("image_paths", []):
        image_path = Path(image_path_str)
        inline_data: dict[str, Any] = {
            "mimeType": guess_image_mime_type(image_path),
            "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
        }
        if media_resolution:
            inline_data["mediaResolution"] = media_resolution
        parts.append({"inlineData": inline_data})
    parts.append({"text": request_payload["user_prompt"]})
    return parts


def build_gemini_request_body(
    request_payload: dict[str, Any],
    temperature: float,
    max_output_tokens: int,
    media_resolution: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "systemInstruction": {
            "parts": [{"text": request_payload["system_prompt"]}],
        },
        "contents": [
            {
                "role": "user",
                "parts": build_gemini_user_parts(request_payload, media_resolution),
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    return body


def extract_gemini_response_text(response_payload: dict[str, Any]) -> str:
    candidates = response_payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    first_candidate = candidates[0]
    if not isinstance(first_candidate, dict):
        return ""
    content = first_candidate.get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    text_parts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            text_parts.append(text)
    return "".join(text_parts).strip()


def invoke_gemini_api(
    request_payload: dict[str, Any],
    api_key: str,
    api_model: str,
    api_version: str,
    api_base_url: str,
    timeout_seconds: int,
    temperature: float,
    max_output_tokens: int,
    media_resolution: str | None,
) -> tuple[str, dict[str, Any]]:
    request_body = build_gemini_request_body(
        request_payload=request_payload,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        media_resolution=media_resolution,
    )
    body_bytes = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    url = f"{api_base_url}/{api_version}/models/{api_model}:generateContent"
    request = urllib_request.Request(
        url=url,
        data=body_bytes,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            response_text = response.read().decode("utf-8")
            status_code = getattr(response, "status", None) or response.getcode()
    except urllib_error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemini API request failed with status={exc.code}: {error_text[:1000]}"
        ) from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc
    latency_seconds = time.perf_counter() - started
    try:
        response_payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini API returned non-JSON response: {response_text[:1000]}") from exc
    raw_response = extract_gemini_response_text(response_payload)
    metadata = {
        "provider": "gemini",
        "api_model": api_model,
        "api_version": api_version,
        "status_code": status_code,
        "latency_seconds": round(latency_seconds, 4),
        "provider_payload": {
            "usageMetadata": response_payload.get("usageMetadata"),
            "modelVersion": response_payload.get("modelVersion"),
            "responseId": response_payload.get("responseId"),
            "finishReason": (
                response_payload.get("candidates", [{}])[0].get("finishReason")
                if isinstance(response_payload.get("candidates"), list) and response_payload.get("candidates")
                else None
            ),
        },
    }
    if not raw_response:
        raise RuntimeError(f"Gemini API returned no text content: {json.dumps(response_payload, ensure_ascii=False)[:1000]}")
    return raw_response, metadata


def is_retryable_api_exception(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    message = f"{exc.__class__.__name__}: {exc}".lower()
    if any(token in message for token in ("timed out", "timeout", "temporarily unavailable", "temporary failure")):
        return True
    return any(f"status={status_code}" in message for status_code in (408, 409, 425, 429, 500, 502, 503, 504))


def build_attempt_error_record(attempt: int, exc: Exception, retryable: bool) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "retryable": retryable,
        "error": f"{exc.__class__.__name__}: {exc}",
    }


def get_sample_stratum_key(dataset: Any, index: int) -> tuple[str, int]:
    outfit, need_to_modify = dataset._resolve_sample(index)
    return infer_audience_segment(outfit.source_group), int(need_to_modify)


def summarize_selected_indices(dataset: Any, selected_indices: Sequence[int]) -> dict[str, Any]:
    segment_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    stratum_counts: dict[str, int] = {}
    for index in selected_indices:
        segment, need_to_modify = get_sample_stratum_key(dataset, index)
        segment_counts[segment] = segment_counts.get(segment, 0) + 1
        label_key = str(need_to_modify)
        label_counts[label_key] = label_counts.get(label_key, 0) + 1
        stratum_key = f"{segment}|{need_to_modify}"
        stratum_counts[stratum_key] = stratum_counts.get(stratum_key, 0) + 1
    return {
        "total": len(selected_indices),
        "segment": dict(sorted(segment_counts.items())),
        "need_to_modify": dict(sorted(label_counts.items())),
        "segment_need_to_modify": dict(sorted(stratum_counts.items())),
    }


def build_api_request(
    prompt_template: str,
    sample: dict[str, Any],
    thinking_mode: str,
) -> dict[str, Any]:
    system_prompt = eval_utils.build_effective_prompt_template(prompt_template, thinking_mode)
    user_prompt = eval_utils.build_structured_user_prompt(sample["prompt_input"])
    return {
        "index": sample["index"],
        "outfit_id": sample["outfit_id"],
        "prompt_input": sample["prompt_input"],
        "candidate_style": sample["prompt_input"].get("candidate_style", []),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "image_paths": [str(path) for path in sample["image_paths"]],
        "gold": sample["gold"],
    }


def build_sample_manifest(
    dataset: Any,
    selected_indices: Sequence[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    samples = [eval_utils.build_sample_record(dataset, index) for index in selected_indices]
    return {
        "selection_strategy": "full_test_split",
        "root": str(Path(args.root).expanduser()),
        "split": DEFAULT_EVAL_SPLIT,
        "split_seed": args.split_seed,
        "sample_mode": args.sample_mode,
        "selected_count": len(selected_indices),
        "distribution": summarize_selected_indices(dataset, selected_indices),
        "samples": [
            {
                "index": sample["index"],
                "outfit_id": sample["outfit_id"],
                "source_group": dataset._resolve_sample(sample["index"])[0].source_group,
                "audience_segment": infer_audience_segment(dataset._resolve_sample(sample["index"])[0].source_group),
                "prompt_input": sample["prompt_input"],
                "image_paths": [str(path) for path in sample["image_paths"]],
                "gold": sample["gold"],
            }
            for sample in samples
        ],
    }


def write_json_object(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def evaluate_one_request(
    request_row: dict[str, Any],
    timeout_seconds: int,
    api_model: str,
    api_version: str,
    api_base_url: str,
    temperature: float,
    max_output_tokens: int,
    media_resolution: str | None,
    retry_attempts: int,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    sample = request_row["sample"]
    row = eval_utils.build_result_row(sample)
    row["stage"] = "invoke_gemini"
    row["api_request"] = request_row["request"]
    raw_text = ""
    parse_seconds = 0.0
    max_attempts = max(1, retry_attempts + 1)
    attempt_errors: list[dict[str, Any]] = []
    parse_started: float | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw_text, metadata = invoke_gemini_api(
                request_payload=request_row["request"],
                api_key=resolve_api_key(),
                timeout_seconds=timeout_seconds,
                api_model=api_model,
                api_version=api_version,
                api_base_url=api_base_url,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                media_resolution=media_resolution,
            )
            metadata = dict(metadata)
            metadata["attempt_count"] = attempt
            metadata["retry_count"] = attempt - 1
            if attempt_errors:
                metadata["retry_errors"] = list(attempt_errors)
            row["api_metadata"] = metadata
            row["raw_response"] = raw_text
            break
        except Exception as exc:  # noqa: BLE001
            retryable = is_retryable_api_exception(exc)
            attempt_error = build_attempt_error_record(attempt, exc, retryable)
            attempt_errors.append(attempt_error)
            row["api_attempts"] = list(attempt_errors)
            if attempt >= max_attempts or not retryable:
                row["error"] = attempt_error["error"]
                break
            if retry_backoff_seconds > 0:
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))

    if attempt_errors:
        row["api_attempts"] = list(attempt_errors)
    if "api_metadata" not in row:
        row["api_metadata"] = {
            "provider": "gemini",
            "api_model": api_model,
            "api_version": api_version,
            "attempt_count": len(attempt_errors) if attempt_errors else 1,
            "retry_count": max(0, (len(attempt_errors) if attempt_errors else 1) - 1),
        }
        if attempt_errors:
            row["api_metadata"]["retry_errors"] = list(attempt_errors)

    if row.get("error") is None and raw_text:
        try:
            row["stage"] = "parse_output"
            parse_started = time.perf_counter()
            eval_utils.finalize_row_from_raw_text(row, sample, raw_text)
            parse_seconds = time.perf_counter() - parse_started
        except Exception as exc:  # noqa: BLE001
            row["raw_response"] = raw_text
            if parse_started is not None and row["stage"] == "parse_output":
                parse_seconds = time.perf_counter() - parse_started
            row["error"] = f"{exc.__class__.__name__}: {exc}"
    elif raw_text:
        row["raw_response"] = raw_text
    row["efficiency"] = build_api_efficiency(
        request_payload=request_row["request"],
        raw_text=raw_text,
        api_metadata=row.get("api_metadata"),
        total_seconds=time.perf_counter() - started,
        parse_seconds=parse_seconds,
    )
    return row


def build_argument_parser(defaults: RuntimeDefaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate zero-shot performance of externally hosted Gemini multimodal models on the full test split."
        )
    )
    parser.add_argument(
        "--api-model",
        default=defaults.api_model,
        help=(
            "Gemini model id. If omitted, the default Gemini model is used "
            f"({DEFAULT_GEMINI_MODEL})."
        ),
    )
    parser.add_argument(
        "--api-version",
        default=defaults.api_version,
        help=(
            "Gemini API version path. If omitted, the default Gemini API version is used "
            f"({DEFAULT_GEMINI_API_VERSION})."
        ),
    )
    parser.add_argument(
        "--api-base-url",
        default=defaults.api_base_url,
        help=(
            "Gemini API base URL. If omitted, the default Gemini base URL is used "
            f"({DEFAULT_GEMINI_BASE_URL})."
        ),
    )
    parser.add_argument("--run-name", default=defaults.run_name, help=f"Run label used in result filenames. Default: {defaults.run_name}")
    parser.add_argument(
        "--root",
        default=defaults.dataset_root,
        help="Dataset root directory containing the Female, Male, and Child folders, each with look*.csv, label*.csv, and photos*.",
    )
    parser.add_argument("--prompt-file", default=defaults.prompt_file, help="Prompt template file.")
    parser.add_argument("--sample-mode", choices=("original", "modified", "both"), default="both")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=defaults.split_seed,
        help=f"Random seed used to build the fixed train/val/test split before evaluating the full test set. Default: {defaults.split_seed}",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("hidden", "visible"),
        default=defaults.thinking_mode if defaults.thinking_mode in {"hidden", "visible"} else "hidden",
        help="How to render reasoning instructions in the exported prompt. Default: hidden.",
    )
    parser.add_argument(
        "--local-api-config",
        default=defaults.local_api_config,
        help=(
            "Path to the local-only Python config file that stores private Gemini credentials. "
            f"Default: {defaults.local_api_config}"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=defaults.temperature,
        help=(
            f"Generation temperature for Gemini calls. Default: {defaults.temperature}. "
            "Google's Gemini 3.1 docs recommend 1.0."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=defaults.max_output_tokens,
        help=f"Max output tokens for Gemini calls. Default: {defaults.max_output_tokens}.",
    )
    parser.add_argument(
        "--media-resolution",
        choices=("low", "medium", "high"),
        default=defaults.media_resolution if defaults.media_resolution in {"low", "medium", "high"} else None,
        help="Optional media resolution hint for Gemini.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=defaults.parallelism,
        help=f"How many Gemini requests to run concurrently. Default: {defaults.parallelism}.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=defaults.timeout,
        help=f"Per-request timeout for the Gemini call. Default: {defaults.timeout}.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=defaults.retry_attempts,
        help=(
            "How many times to retry transient provider failures after the first attempt. "
            f"Default: {defaults.retry_attempts}."
        ),
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=defaults.retry_backoff_seconds,
        help=(
            "Base backoff in seconds between retries. "
            "Later retries use exponential backoff from this base. "
            f"Default: {defaults.retry_backoff_seconds}."
        ),
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only export the evaluation manifest and request JSONL, then exit.")
    parser.add_argument(
        "--sample-manifest",
        default=defaults.sample_manifest,
        help="Optional path for the selected sample manifest JSON.",
    )
    parser.add_argument(
        "--input-manifest",
        default=defaults.input_manifest,
        help=(
            "Optional path to an existing sample manifest JSON. "
            "If set, evaluate exactly those samples instead of rebuilding them from the dataset."
        ),
    )
    parser.add_argument(
        "--requests-path",
        default=defaults.requests_path,
        help="Optional path for the exported request JSONL.",
    )
    parser.add_argument(
        "--output",
        default=defaults.output,
        help=(
            "Path to write prediction JSON. "
            f"Default: {defaults.results_dir}/<run_name>_zero-shot_{DEFAULT_EVAL_SPLIT}_results.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run_started = time.perf_counter()
    defaults = load_runtime_defaults()
    parser = build_argument_parser(defaults)
    args = parser.parse_args(argv)
    try:
        loaded_api_names = apply_local_api_variables(Path(args.local_api_config).expanduser())
    except Exception as exc:  # noqa: BLE001
        parser.error(str(exc))
    if args.parallelism < 1:
        parser.error("--parallelism must be at least 1.")
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be at least 1.")
    if args.retry_attempts < 0:
        parser.error("--retry-attempts must be >= 0.")
    if args.retry_backoff_seconds < 0:
        parser.error("--retry-backoff-seconds must be >= 0.")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be >= 1.")
    requested_max_output_tokens = args.max_output_tokens
    args.max_output_tokens = resolve_effective_max_output_tokens(args.max_output_tokens, args.thinking_mode)
    args.api_model = resolve_api_model(args.api_model)
    args.api_version = resolve_api_version(args.api_version)
    args.api_base_url = resolve_api_base_url(args.api_base_url)
    if loaded_api_names:
        print(
            "[config] loaded local Gemini variables from "
            f"{Path(args.local_api_config).expanduser()}: {', '.join(loaded_api_names)}",
            flush=True,
        )
    if not args.prepare_only:
        try:
            resolve_api_key()
        except RuntimeError as exc:
            parser.error(str(exc))

    config = {
        "provider": "gemini",
        "api_model": args.api_model,
        "api_version": args.api_version,
        "api_base_url": args.api_base_url,
        "run_name": args.run_name,
        "split": DEFAULT_EVAL_SPLIT,
        "split_seed": args.split_seed,
        "sample_mode": args.sample_mode,
        "thinking_mode": args.thinking_mode,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "parallelism": args.parallelism,
        "timeout_seconds": args.timeout_seconds,
        "retry_attempts": args.retry_attempts,
        "retry_backoff_seconds": args.retry_backoff_seconds,
        "prepare_only": args.prepare_only,
        "input_manifest": args.input_manifest,
    }
    print("[config] gemini evaluation configuration:", flush=True)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    if args.max_output_tokens != requested_max_output_tokens:
        print(
            f"[config] increased max_output_tokens from {requested_max_output_tokens} to {args.max_output_tokens} "
            f"for thinking_mode={args.thinking_mode}",
            flush=True,
        )

    manifest_payload: dict[str, Any] | None = None
    samples: list[dict[str, Any]]
    effective_split = DEFAULT_EVAL_SPLIT
    if args.input_manifest:
        manifest_input_path = Path(args.input_manifest).expanduser()
        print(f"[step 1/6] loading selected samples from manifest: {manifest_input_path}", flush=True)
        try:
            manifest_payload, samples = load_samples_from_manifest(manifest_input_path)
        except Exception as exc:  # noqa: BLE001
            parser.error(str(exc))
        if not samples:
            parser.error(f"No samples found in manifest: {manifest_input_path}")
        effective_split = str(manifest_payload.get("split") or DEFAULT_EVAL_SPLIT)
        output_path = resolve_output_path(
            defaults.results_dir,
            args.run_name,
            effective_split,
            args.output,
        )
        metrics_path = eval_utils.build_metrics_path(output_path)
        requests_path = resolve_requests_path(output_path, args.requests_path)
        manifest_path = resolve_sample_manifest_path(
            output_path,
            args.sample_manifest or str(manifest_input_path),
        )
        distribution = manifest_payload.get("distribution")
        print(
            "[step 1/6] manifest summary: "
            f"selected_size={len(samples)} "
            f"distribution={json.dumps(distribution, ensure_ascii=False) if distribution is not None else 'null'}",
            flush=True,
        )
    else:
        print("[step 1/6] loading dataset", flush=True)
        OutfitNegativeSampleDataset = eval_utils.import_dataset_class()
        dataset = OutfitNegativeSampleDataset(
            root=args.root,
            transform=None,
            deterministic=True,
            sample_mode=args.sample_mode,
            split=DEFAULT_EVAL_SPLIT,
            split_seed=args.split_seed,
        )
        selected_indices = list(range(len(dataset)))
        distribution = summarize_selected_indices(dataset, selected_indices)

        output_path = resolve_output_path(
            defaults.results_dir,
            args.run_name,
            DEFAULT_EVAL_SPLIT,
            args.output,
        )
        metrics_path = eval_utils.build_metrics_path(output_path)
        requests_path = resolve_requests_path(output_path, args.requests_path)
        manifest_path = resolve_sample_manifest_path(output_path, args.sample_manifest)

        print(
            "[step 1/6] selection summary: "
            f"test_split_size={len(dataset)} selected_size={len(selected_indices)} "
            f"distribution={json.dumps(distribution, ensure_ascii=False)}",
            flush=True,
        )

    print("[step 2/6] loading prompt template", flush=True)
    prompt_template = eval_utils.load_prompt(Path(args.prompt_file))

    print("[step 3/6] building sample manifest and request payloads", flush=True)
    if manifest_payload is None:
        samples = [eval_utils.build_sample_record(dataset, index) for index in selected_indices]
    request_rows = [
        {
            "selection_rank": rank,
            "sample": sample,
            "request": build_api_request(prompt_template, sample, args.thinking_mode),
        }
        for rank, sample in enumerate(samples, start=1)
    ]
    if manifest_payload is None:
        manifest_payload = build_sample_manifest(dataset, selected_indices, args)
    write_json_object(manifest_path, manifest_payload)
    write_jsonl(
        requests_path,
        [
            {
                "selection_rank": row["selection_rank"],
                **row["request"],
            }
            for row in request_rows
        ],
    )
    print(f"[step 3/6] sample manifest written to: {manifest_path}", flush=True)
    print(f"[step 3/6] request jsonl written to: {requests_path}", flush=True)

    if args.prepare_only:
        return 0

    print(
        f"[step 4/6] invoking gemini endpoint with parallelism={args.parallelism}",
        flush=True,
    )
    evaluation_started = time.perf_counter()
    rows_by_rank: dict[int, dict[str, Any]] = {}
    if args.parallelism == 1:
        for request_row in request_rows:
            rank = request_row["selection_rank"]
            print(
                f"[sample {rank}/{len(request_rows)}] outfit_id={request_row['sample']['outfit_id']} stage=invoke_gemini",
                flush=True,
            )
            row = evaluate_one_request(
                request_row,
                args.timeout_seconds,
                args.api_model,
                args.api_version,
                args.api_base_url,
                args.temperature,
                args.max_output_tokens,
                args.media_resolution,
                args.retry_attempts,
                args.retry_backoff_seconds,
            )
            row["selection_rank"] = rank
            rows_by_rank[rank] = row
    else:
        with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            future_to_rank = {
                executor.submit(
                    evaluate_one_request,
                    request_row,
                    args.timeout_seconds,
                    args.api_model,
                    args.api_version,
                    args.api_base_url,
                    args.temperature,
                    args.max_output_tokens,
                    args.media_resolution,
                    args.retry_attempts,
                    args.retry_backoff_seconds,
                ): request_row["selection_rank"]
                for request_row in request_rows
            }
            for future in as_completed(future_to_rank):
                rank = future_to_rank[future]
                row = future.result()
                row["selection_rank"] = rank
                rows_by_rank[rank] = row
                print(
                    f"[sample {rank}/{len(request_rows)}] outfit_id={row['outfit_id']} stage={row['stage']} "
                    f"json_valid={row['json_valid']} error={row['error']}",
                    flush=True,
                )

    rows = [rows_by_rank[rank] for rank in sorted(rows_by_rank)]

    print("[step 5/6] writing predictions", flush=True)
    eval_utils.write_json(output_path, rows)

    print("[step 6/6] writing metrics", flush=True)
    metrics = eval_utils.summarize_metrics(rows)
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
