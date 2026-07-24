#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${CANONICAL_STUDY_VENV:-${PROJECT_ROOT}/.venv}"

if [[ ! -d "${VENV_PATH}" ]]; then
  # virtualenv is considerably more reliable than stdlib venv on network filesystems.
  virtualenv --system-site-packages "${VENV_PATH}"
fi

"${VENV_PATH}/bin/python" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  setuptools==83.0.0 \
  wheel==0.47.0 \
  torchvision==0.18.0+cu121 \
  open-clip-torch==2.24.0
"${VENV_PATH}/bin/python" -m pip install \
  --no-build-isolation \
  --no-deps \
  --editable "${PROJECT_ROOT}"

echo "Environment ready: ${VENV_PATH}"
