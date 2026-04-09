from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from task1_image_edit.io import project_root


def resolve_diffsynth_root(path_like: str | Path | None = None) -> Path:
    if path_like not in (None, ""):
        return Path(path_like).expanduser().resolve()
    env_path = os.environ.get("DIFFSYNTH_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (project_root() / "external" / "DiffSynth-Studio").resolve()


def prepend_pythonpath(path_like: str | Path) -> Path:
    resolved = Path(path_like).expanduser().resolve()
    resolved_str = str(resolved)
    if resolved_str not in sys.path:
        sys.path.insert(0, resolved_str)
    return resolved


def ensure_diffsynth_available(diffsynth_root: str | Path | None = None) -> Path | None:
    try:
        importlib.import_module("diffsynth")
        return None
    except ModuleNotFoundError as exc:
        if exc.name != "diffsynth":
            raise

    resolved_root = resolve_diffsynth_root(diffsynth_root)
    if not resolved_root.exists():
        raise ModuleNotFoundError(
            "qwen_edit requires the 'diffsynth' package. "
            "Install DiffSynth-Studio or point --diffsynth-root / DIFFSYNTH_DIR to a local checkout. "
            f"Looked for: {resolved_root}"
        )

    prepend_pythonpath(resolved_root)
    try:
        importlib.import_module("diffsynth")
    except ModuleNotFoundError as exc:
        if exc.name == "diffsynth":
            raise ModuleNotFoundError(
                "Found a DiffSynth-Studio checkout but still could not import 'diffsynth'. "
                f"Check that {resolved_root} is a valid DiffSynth-Studio repository."
            ) from exc
        raise
    return resolved_root


