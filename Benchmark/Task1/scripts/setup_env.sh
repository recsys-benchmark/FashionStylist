#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
DIFFUSERS_DIR="${DIFFUSERS_DIR:-$ROOT_DIR/external/diffusers}"
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-$ROOT_DIR/external/DiffSynth-Studio}"
LONGCAT_DIR="${LONGCAT_DIR:-$ROOT_DIR/external/LongCat-Image}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

# Check that external repos have been cloned
for repo_dir in "$DIFFUSERS_DIR" "$DIFFSYNTH_DIR" "$LONGCAT_DIR"; do
  if [ ! -d "$repo_dir" ]; then
    echo "ERROR: $repo_dir not found. Please clone the required external repos first. See README.md." >&2
    exit 1
  fi
done

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url "$PYTORCH_INDEX_URL"
python -m pip install -r "$ROOT_DIR/requirements/base.txt" -r "$ROOT_DIR/requirements/train.txt"

python -m pip install -e "$ROOT_DIR"
python -m pip install -e "$DIFFUSERS_DIR"
python -m pip install -e "$DIFFSYNTH_DIR"
python -m pip install -e "$LONGCAT_DIR"
python -m pip install -r "$DIFFUSERS_DIR/examples/dreambooth/requirements_flux.txt"

cat <<EOF
Environment ready.
1. source "$VENV_DIR/bin/activate"
2. hf auth login
3. export DIFFUSERS_DIR="$DIFFUSERS_DIR"
4. export DIFFSYNTH_DIR="$DIFFSYNTH_DIR"
5. export LONGCAT_DIR="$LONGCAT_DIR"
EOF
