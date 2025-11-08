#!/usr/bin/env bash
set -euo pipefail

# Petit wrapper pour créer un venv, installer les requirements et lancer src/main.py
# Usage: ./scripts/run.sh [--] [args passed to main.py]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Creating virtual environment in $VENV_DIR using $PYTHON..."
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

REQ_CANDIDATES=("requirements.txt" "requirement.txt" "requirements-dev.txt")
REQ_FILE=""
for f in "${REQ_CANDIDATES[@]}"; do
  if [ -f "$ROOT_DIR/$f" ]; then
    REQ_FILE="$ROOT_DIR/$f"
    break
  fi
done

if [ -n "$REQ_FILE" ]; then
  echo "Installing requirements from $REQ_FILE..."
  pip install -r "$REQ_FILE"
else
  echo "No requirements file found in $ROOT_DIR (checked: ${REQ_CANDIDATES[*]}) — skipping install."
fi

echo "Running src/main.py..."
python src/main.py "$@"
