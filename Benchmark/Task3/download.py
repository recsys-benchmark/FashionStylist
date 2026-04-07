#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = os.environ.get("VLLM_MODELS_DIR", str(SCRIPT_DIR / "models"))
DEFAULT_ENDPOINT = (
    os.environ.get("HF_ENDPOINT")
    or os.environ.get("HUGGINGFACE_HUB_ENDPOINT")
    or os.environ.get("HF_MIRROR_ENDPOINT")
    or "https://hf-mirror.com"
)
DEFAULT_DOWNLOAD_TIMEOUT = int(float(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "1200")))
DEFAULT_ETAG_TIMEOUT = float(os.environ.get("HF_HUB_ETAG_TIMEOUT", "10"))
DEFAULT_MAX_WORKERS = int(os.environ.get("HF_DOWNLOAD_MAX_WORKERS", "8"))
DEFAULT_PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
WEIGHT_INDEX_FILENAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
DIRECT_WEIGHT_PATTERNS = (
    "model*.safetensors",
    "pytorch_model*.bin",
    "consolidated*.safetensors",
)
MODEL_SPECS = {
    "qwen25vl-7b": {
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "local_dir_name": "Qwen2.5-VL-7B-Instruct",
    },
    "qwen3vl-4b": {
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "local_dir_name": "Qwen3-VL-4B-Instruct",
    },
    "qwen3vl-4b-thinking": {
        "repo_id": "Qwen/Qwen3-VL-4B-Thinking",
        "local_dir_name": "Qwen3-VL-4B-Thinking",
    },
    "qwen3vl-8b": {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
        "local_dir_name": "Qwen3-VL-8B-Instruct",
    },
    "qwen3vl-8b-thinking": {
        "repo_id": "Qwen/Qwen3-VL-8B-Thinking",
        "local_dir_name": "Qwen3-VL-8B-Thinking",
    },
    "unsloth-gemma3-4b": {
        "repo_id": "unsloth/gemma-3-4b-it",
        "local_dir_name": "unsloth-gemma-3-4b-it",
    },
}


def import_snapshot_download() -> Any:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it with `pip install huggingface_hub`."
        ) from exc
    return snapshot_download


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download supported multimodal checkpoints into the local models directory for offline evaluation."
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(MODEL_SPECS),
        help="Model alias to download. Repeat this flag to download a subset. Default: all supported models.",
    )
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help=(
            f"Directory to store downloaded models. Default: {DEFAULT_MODELS_DIR}. "
            "Relative paths are resolved against the directory containing download.py."
        ),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="Optional Hugging Face token. Default: HF_TOKEN/HUGGINGFACE_TOKEN from environment.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision, tag, or commit.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=(
            "Optional Hugging Face Hub endpoint. "
            "Defaults to HF_ENDPOINT/HUGGINGFACE_HUB_ENDPOINT/HF_MIRROR_ENDPOINT if set, "
            "otherwise https://hf-mirror.com."
        ),
    )
    parser.add_argument(
        "--etag-timeout",
        type=float,
        default=DEFAULT_ETAG_TIMEOUT,
        help=(
            f"Metadata/etag request timeout in seconds passed to snapshot_download. Default: {DEFAULT_ETAG_TIMEOUT}."
        ),
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=DEFAULT_DOWNLOAD_TIMEOUT,
        help=(
            "Per-request file download timeout in seconds for huggingface_hub. "
            f"Implemented via HF_HUB_DOWNLOAD_TIMEOUT. Default: {DEFAULT_DOWNLOAD_TIMEOUT}."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=(
            f"Maximum concurrent workers used by snapshot_download. Lower this on unstable links. Default: {DEFAULT_MAX_WORKERS}."
        ),
    )
    parser.add_argument(
        "--proxy",
        default=DEFAULT_PROXY,
        help=(
            "Optional HTTP/HTTPS proxy URL. Defaults to HTTPS_PROXY/HTTP_PROXY if set. "
            "Example: http://127.0.0.1:7890"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if the target directory already contains config.json.",
    )
    return parser


def resolve_selected_models(selected: list[str] | None, token: str | None) -> list[tuple[str, dict[str, Any]]]:
    if selected:
        aliases = selected
    else:
        aliases = [
            alias
            for alias, spec in MODEL_SPECS.items()
            if token or not spec.get("requires_hf_token", False)
        ]
    return [(alias, MODEL_SPECS[alias]) for alias in aliases]


def print_config(args: argparse.Namespace, models_dir: Path, selected_models: list[tuple[str, dict[str, Any]]]) -> None:
    config = {
        "models_dir": str(models_dir),
        "selected_models": [
            {
                "alias": alias,
                "repo_id": spec["repo_id"],
                "target_dir": str((models_dir / spec["local_dir_name"]).resolve()),
                "requires_hf_token": bool(spec.get("requires_hf_token", False)),
            }
            for alias, spec in selected_models
        ],
        "revision": args.revision,
        "endpoint": args.endpoint,
        "etag_timeout": args.etag_timeout,
        "download_timeout": args.download_timeout,
        "max_workers": args.max_workers,
        "proxy_configured": bool(args.proxy),
        "token_provided": bool(args.token),
        "force": args.force,
    }
    print("[config] download configuration:", flush=True)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


def resolve_proxies(proxy: str | None) -> dict[str, str] | None:
    if not proxy:
        return None
    return {
        "http": proxy,
        "https": proxy,
    }


def is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def check_local_model_completeness(target_dir: Path) -> tuple[bool, str, list[str]]:
    config_path = target_dir / "config.json"
    if not is_nonempty_file(config_path):
        return False, "config_missing", ["config.json"]

    incomplete_files = sorted(str(path.relative_to(target_dir)) for path in target_dir.rglob("*.incomplete"))
    if incomplete_files:
        return False, "partial_files_present", incomplete_files

    for index_filename in WEIGHT_INDEX_FILENAMES:
        index_path = target_dir / index_filename
        if not index_path.exists():
            continue
        try:
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return False, f"unreadable_{index_filename}", [index_filename]

        weight_map = index_payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False, f"invalid_{index_filename}", [index_filename]

        shard_names = sorted({str(value) for value in weight_map.values() if value})
        missing_shards = [name for name in shard_names if not is_nonempty_file(target_dir / name)]
        if missing_shards:
            return False, f"missing_shards_from_{index_filename}", missing_shards
        return True, f"complete_sharded_{index_filename}", []

    direct_weight_files = sorted(
        {
            str(path.relative_to(target_dir))
            for pattern in DIRECT_WEIGHT_PATTERNS
            for path in target_dir.glob(pattern)
            if is_nonempty_file(path)
        }
    )
    if direct_weight_files:
        return True, "complete_direct_weight_files", []

    return False, "weight_files_missing", list(DIRECT_WEIGHT_PATTERNS)


def build_download_error_message(
    alias: str,
    repo_id: str,
    exc: Exception,
    endpoint: str | None,
    proxy: str | None,
    max_workers: int,
) -> str:
    message = f"{exc.__class__.__name__}: {exc}"
    text = message.lower()
    network_tokens = (
        "timeout",
        "timed out",
        "connection",
        "proxy",
        "ssl",
        "tls",
        "max retries exceeded",
        "temporary failure",
        "name or service not known",
        "connection reset",
    )
    if not any(token in text for token in network_tokens):
        return message

    hints = [
        f"{alias} download looks like a network/connectivity failure while reaching {repo_id}.",
        "If this machine is in mainland China, try one of the following:",
        f"1. Configure a proxy and rerun with --proxy ... or HTTPS_PROXY/HTTP_PROXY (currently {'set' if proxy else 'unset'}).",
        f"2. Route through an approved Hub endpoint with --endpoint ... or HF_ENDPOINT (currently {endpoint or 'default https://huggingface.co'}).",
        f"3. Lower concurrency with --max-workers 1 or 2 (currently {max_workers}).",
        "4. Increase HF_HUB_DOWNLOAD_TIMEOUT or pass --download-timeout 1200 to avoid 10s shard read timeouts.",
        "5. Download on an overseas server first, then rsync/scp the model directory to this machine.",
    ]
    return f"{message}\n" + "\n".join(hints)


def download_one_model(
    snapshot_download: Any,
    alias: str,
    spec: dict[str, Any],
    models_dir: Path,
    token: str | None,
    revision: str | None,
    endpoint: str | None,
    etag_timeout: float,
    max_workers: int,
    proxy: str | None,
    force: bool,
    index: int,
    total: int,
) -> Path:
    try:
        from huggingface_hub.errors import GatedRepoError
    except ImportError:  # pragma: no cover - huggingface_hub already required above
        GatedRepoError = None  # type: ignore[assignment]

    target_dir = (models_dir / spec["local_dir_name"]).resolve()
    prefix = f"[model {index}/{total}]"
    print(f"{prefix} alias={alias} repo_id={spec['repo_id']} target_dir={target_dir}", flush=True)

    if target_dir.exists() and not force:
        complete, reason, details = check_local_model_completeness(target_dir)
        if complete:
            print(f"{prefix} stage=skip_existing reason={reason}", flush=True)
            return target_dir
        preview = details[:5]
        suffix = "" if len(details) <= 5 else f" ... (+{len(details) - 5} more)"
        print(
            f"{prefix} stage=resume_existing reason={reason} details={preview}{suffix}",
            flush=True,
        )

    if spec.get("requires_hf_token") and not token:
        raise RuntimeError(
            f"{alias} requires a Hugging Face access token and accepted gated-model access. "
            f"Visit https://huggingface.co/{spec['repo_id']} to request access, then rerun with "
            f"`HF_TOKEN=... python3 download.py --model {alias}` or `--token ...`."
        )

    print(f"{prefix} stage=download_start", flush=True)
    proxies = resolve_proxies(proxy)
    try:
        snapshot_download(
            repo_id=spec["repo_id"],
            local_dir=str(target_dir),
            token=token,
            revision=revision,
            endpoint=endpoint,
            etag_timeout=etag_timeout,
            max_workers=max_workers,
            proxies=proxies,
            force_download=force,
        )
    except Exception as exc:
        if GatedRepoError is not None and isinstance(exc, GatedRepoError):
            raise RuntimeError(
                f"Access to {spec['repo_id']} is gated. Visit https://huggingface.co/{spec['repo_id']} "
                "to request access, then rerun with a valid HF token."
            ) from exc
        raise RuntimeError(
            build_download_error_message(
                alias=alias,
                repo_id=spec["repo_id"],
                exc=exc,
                endpoint=endpoint,
                proxy=proxy,
                max_workers=max_workers,
            )
        ) from exc
    print(f"{prefix} stage=download_done", flush=True)
    return target_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1.")
    if args.etag_timeout <= 0:
        parser.error("--etag-timeout must be > 0.")
    if args.download_timeout <= 0:
        parser.error("--download-timeout must be > 0.")
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.download_timeout)
    models_dir = Path(args.models_dir).expanduser()
    if not models_dir.is_absolute():
        models_dir = SCRIPT_DIR / models_dir
    models_dir = models_dir.resolve()
    selected_models = resolve_selected_models(args.model, args.token)
    print_config(args, models_dir, selected_models)

    snapshot_download = import_snapshot_download()

    downloaded_paths: list[str] = []
    failures: list[dict[str, str]] = []
    total = len(selected_models)
    for index, (alias, spec) in enumerate(selected_models, start=1):
        try:
            target_dir = download_one_model(
                snapshot_download=snapshot_download,
                alias=alias,
                spec=spec,
                models_dir=models_dir,
                token=args.token,
                revision=args.revision,
                endpoint=args.endpoint,
                etag_timeout=args.etag_timeout,
                max_workers=args.max_workers,
                proxy=args.proxy,
                force=args.force,
                index=index,
                total=total,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[model {index}/{total}] stage=download_failed error={exc.__class__.__name__}: {exc}",
                flush=True,
            )
            failures.append({"alias": alias, "error": f"{exc.__class__.__name__}: {exc}"})
            continue
        downloaded_paths.append(str(target_dir))

    print("\nDownloaded model directories:", flush=True)
    print(json.dumps(downloaded_paths, ensure_ascii=False, indent=2), flush=True)
    if failures:
        print("\nFailed model downloads:", flush=True)
        print(json.dumps(failures, ensure_ascii=False, indent=2), flush=True)
    print("\nUse them with evaluate.py, for example:", flush=True)
    print(
        "python3 evaluate.py --model models/Qwen2.5-VL-7B-Instruct",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
