#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python3.13 -m venv "$ROOT/.venv-cpu"
"$ROOT/.venv-cpu/bin/python" -m pip install --upgrade pip
"$ROOT/.venv-cpu/bin/python" -m pip install -r "$ROOT/requirements-cpu.txt"
"$ROOT/.venv-cpu/bin/python" -m pip check
