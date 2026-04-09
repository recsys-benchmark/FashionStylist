from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable

from task1_image_edit.io import project_root


def ensure_project_script(root_path: str | Path, relative_script: str, project_name: str) -> Path:
    root = Path(root_path).resolve()
    script_path = root / relative_script
    if not script_path.exists():
        raise FileNotFoundError(f"Missing {project_name} script: {script_path}")
    return script_path


def ensure_diffusers_script(diffusers_root: str | Path, relative_script: str) -> Path:
    return ensure_project_script(diffusers_root, relative_script, "diffusers")


def ensure_diffsynth_script(diffsynth_root: str | Path, relative_script: str) -> Path:
    return ensure_project_script(diffsynth_root, relative_script, "DiffSynth-Studio")


def _merge_pythonpath(current_pythonpath: str | None, extra_paths: Iterable[str | Path]) -> str:
    merged = []
    for path in extra_paths:
        resolved = str(Path(path).resolve())
        if resolved not in merged:
            merged.append(resolved)
    if current_pythonpath:
        for entry in current_pythonpath.split(os.pathsep):
            if entry and entry not in merged:
                merged.append(entry)
    return os.pathsep.join(merged)


def build_training_env(manifest_path: str | Path, extra_pythonpaths: Iterable[str | Path] | None = None) -> dict[str, str]:
    root = project_root()
    env = os.environ.copy()
    env["GARMENT_DATASET_MANIFEST"] = str(Path(manifest_path).resolve())
    pythonpaths = [root / "src"]
    if extra_pythonpaths:
        pythonpaths.extend(extra_pythonpaths)
    env["PYTHONPATH"] = _merge_pythonpath(env.get("PYTHONPATH"), pythonpaths)
    return env


def run_command(command: list[str], env: dict[str, str], dry_run: bool = False) -> None:
    pretty = " ".join(shlex.quote(part) for part in command)
    print(pretty)
    if dry_run:
        return
    subprocess.run(command, env=env, check=True)
