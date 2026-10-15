#!/usr/bin/env bash
# Idempotent dev-environment bootstrap: Python venv + project dependencies.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# The base image ships python3 but not always the venv/ensurepip module.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y python3-venv
fi

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo "dev environment ready: $(.venv/bin/python --version)"
