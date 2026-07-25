#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <openai-laion|openai-flava> <physical-gpu-id|cpu>" >&2
  exit 2
fi

PAIR="$1"
GPU_ID="$2"
DEVICE="${CANONICAL_STUDY_DEVICE:-cuda}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CANONICAL_STUDY_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
CUB_DATA_ROOT="${CANONICAL_STUDY_CUB_DATA_ROOT:-${PROJECT_ROOT}/artifacts/data/cub}"
MODEL_CACHE_ROOT="${CANONICAL_STUDY_MODEL_CACHE_ROOT:-${PROJECT_ROOT}/artifacts/cache/models}"
EMBEDDING_ROOT="${CANONICAL_STUDY_CUB_EMBEDDING_ROOT:-${PROJECT_ROOT}/artifacts/embeddings/cub}"
PREDICTION_ROOT="${CANONICAL_STUDY_CUB_PREDICTION_ROOT:-${PROJECT_ROOT}/artifacts/predictions/cub}"
OUTPUT_ROOT="${PROJECT_ROOT}/artifacts/results/frozen_decoder"
ALIGNMENT_ROOT="${PROJECT_ROOT}/artifacts/alignments"
LOG_ROOT="${PROJECT_ROOT}/artifacts/logs/cub_train_q_control"

mkdir -p \
  "${MODEL_CACHE_ROOT}" \
  "${EMBEDDING_ROOT}" \
  "${PREDICTION_ROOT}" \
  "${OUTPUT_ROOT}" \
  "${ALIGNMENT_ROOT}" \
  "${LOG_ROOT}"

if [[ "${DEVICE}" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
export PYTHONUNBUFFERED=1
export HF_HOME="${MODEL_CACHE_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${MODEL_CACHE_ROOT}/huggingface"

if [[ "${PAIR}" == "openai-laion" ]]; then
  CONFIG="${PROJECT_ROOT}/configs/cub_train_q_control/openai_laion_cub.json"
elif [[ "${PAIR}" == "openai-flava" ]]; then
  CONFIG="${PROJECT_ROOT}/configs/cub_train_q_control/openai_flava_cub.json"
else
  echo "unknown pair: ${PAIR}" >&2
  exit 2
fi

"${PYTHON}" -m canonical_study validate-cub \
  --data-root "${CUB_DATA_ROOT}"

"${PYTHON}" -m canonical_study cub-train-q-control \
  --config "${CONFIG}" \
  --data-root "${CUB_DATA_ROOT}" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --prediction-root "${PREDICTION_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --alignment-root "${ALIGNMENT_ROOT}" \
  --model-cache-root "${MODEL_CACHE_ROOT}" \
  --device "${DEVICE}" 2>&1 | tee "${LOG_ROOT}/${PAIR}.live.log"
