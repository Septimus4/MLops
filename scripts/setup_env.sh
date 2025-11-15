#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ_FILE="requirements.txt"

if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] Creating virtual environment in $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "[setup] Reusing existing virtual environment $VENV_DIR"
fi

echo "[setup] Upgrading pip/setuptools/wheel"
"$VENV_DIR/bin/pip" install -U pip setuptools wheel

if [ -f "$REQ_FILE" ]; then
  echo "[setup] Installing dependencies from $REQ_FILE"
  "$VENV_DIR/bin/pip" install -r "$REQ_FILE"
else
  echo "[setup] $REQ_FILE not found; skipping dependency install"
fi

echo "[setup] Running test suite (disable with SKIP_TESTS=1)"
if [ "${SKIP_TESTS:-0}" != "1" ]; then
  if ls tests/*.py >/dev/null 2>&1; then
    "$VENV_DIR/bin/pytest" -q || echo "[setup] Tests failed; inspect output above."
  else
    echo "[setup] No tests directory detected."
  fi
fi

echo "\n[setup] Done. Activate with: source $VENV_DIR/bin/activate"
