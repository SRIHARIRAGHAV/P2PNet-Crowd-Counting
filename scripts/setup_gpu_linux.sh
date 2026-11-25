#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=python3.10
CUDA_INDEX="${1:-cu128}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python 3.10 is required; install python3.10 and python3.10-venv first." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found; install and verify the NVIDIA driver first." >&2
    exit 1
fi

case "$CUDA_INDEX" in
    cu128|cu126|cu118)
        ;;
    *)
        echo "Use a PyTorch wheel index such as cu128, cu126, or cu118." >&2
        exit 1
        ;;
esac

nvidia-smi
"$PYTHON" -m venv "$ROOT/.venv-gpu"
"$ROOT/.venv-gpu/bin/python" -m pip install --upgrade pip
"$ROOT/.venv-gpu/bin/python" -m pip install torch torchvision \
    --index-url "https://download.pytorch.org/whl/$CUDA_INDEX"
"$ROOT/.venv-gpu/bin/python" -m pip install -r "$ROOT/requirements-gpu.txt"
"$ROOT/.venv-gpu/bin/python" -m pip check
