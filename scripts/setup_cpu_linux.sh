#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=python3.10
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python 3.10 is required; install python3.10 and python3.10-venv first." >&2
    exit 1
fi

"$PYTHON" -m venv "$ROOT/.venv-cpu"
"$ROOT/.venv-cpu/bin/python" -m pip install --upgrade pip
"$ROOT/.venv-cpu/bin/python" -m pip install -r "$ROOT/requirements-cpu.txt"
"$ROOT/.venv-cpu/bin/python" -m pip check
