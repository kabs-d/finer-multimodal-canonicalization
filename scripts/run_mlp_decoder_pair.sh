#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <openai-laion|openai-flava> <physical-gpu-id>" >&2
  exit 2
fi

PAIR="$1"
GPU_ID="$2"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CANONICAL_STUDY_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
EMBEDDING_ROOT="${CANONICAL_STUDY_CUB_EMBEDDING_ROOT:-${PROJECT_ROOT}/artifacts/embeddings/cub}"
PREDICTION_ROOT="${CANONICAL_STUDY_CUB_PREDICTION_ROOT:-${PROJECT_ROOT}/artifacts/predictions/cub}"
OUTPUT_ROOT="${PROJECT_ROOT}/artifacts/results/frozen_decoder"
ALIGNMENT_ROOT="${PROJECT_ROOT}/artifacts/alignments"
LOG_ROOT="${PROJECT_ROOT}/artifacts/logs/mlp_decoder"

mkdir -p "${PREDICTION_ROOT}" "${OUTPUT_ROOT}" "${LOG_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

if [[ "${PAIR}" == "openai-laion" ]]; then
  CONFIG="${PROJECT_ROOT}/configs/mlp_decoder/openai_laion_cub.json"
  ALIGNMENT="${ALIGNMENT_ROOT}/oxford_openai_vitb32_to_laion_vitb32.pt"
elif [[ "${PAIR}" == "openai-flava" ]]; then
  CONFIG="${PROJECT_ROOT}/configs/mlp_decoder/openai_flava_cub.json"
  ALIGNMENT="${ALIGNMENT_ROOT}/oxford_openai_vitl14_to_flava.pt"
else
  echo "unknown pair: ${PAIR}" >&2
  exit 2
fi

"${PYTHON}" -m canonical_study validate-decoder-config --config "${CONFIG}"

"${PYTHON}" -m canonical_study run-cached-mlp-decoder \
  --config "${CONFIG}" \
  --alignment "${ALIGNMENT}" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --prediction-root "${PREDICTION_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda 2>&1 | tee "${LOG_ROOT}/${PAIR}.training.log"

"${PYTHON}" -m canonical_study analyze-attributes \
  --config "${CONFIG}" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --prediction-root "${PREDICTION_ROOT}" \
  --output-root "${OUTPUT_ROOT}" 2>&1 | tee "${LOG_ROOT}/${PAIR}.attributes.log"
