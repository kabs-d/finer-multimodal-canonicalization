#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <openai-laion|openai-flava> <physical-gpu-id>" >&2
  exit 2
fi

PAIR="$1"
GPU_ID="$2"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
PYTHON="${CANONICAL_STUDY_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
CUB_DATA_ROOT="${CANONICAL_STUDY_CUB_DATA_ROOT:-${PROJECT_ROOT}/artifacts/data/cub}"
MODEL_CACHE_ROOT="${CANONICAL_STUDY_MODEL_CACHE_ROOT:-${PROJECT_ROOT}/artifacts/cache/models}"
EMBEDDING_ROOT="${CANONICAL_STUDY_CUB_EMBEDDING_ROOT:-${PROJECT_ROOT}/artifacts/embeddings/cub}"
PREDICTION_ROOT="${CANONICAL_STUDY_CUB_PREDICTION_ROOT:-${PROJECT_ROOT}/artifacts/predictions/cub}"
OUTPUT_ROOT="${PROJECT_ROOT}/artifacts/results/frozen_decoder"
ALIGNMENT_ROOT="${PROJECT_ROOT}/artifacts/alignments"
LOG_ROOT="${PROJECT_ROOT}/artifacts/logs/frozen_decoder"

mkdir -p \
  "${MODEL_CACHE_ROOT}" \
  "${EMBEDDING_ROOT}" \
  "${PREDICTION_ROOT}" \
  "${OUTPUT_ROOT}" \
  "${ALIGNMENT_ROOT}" \
  "${LOG_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export HF_HOME="${MODEL_CACHE_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${MODEL_CACHE_ROOT}/huggingface"

if [[ "${PAIR}" == "openai-laion" ]]; then
  CONFIG="${PROJECT_ROOT}/configs/frozen_decoder/openai_laion_cub.json"
  ALIGNMENT="${ALIGNMENT_ROOT}/oxford_openai_vitb32_to_laion_vitb32.pt"
  OXFORD_PREFIX="${WORKSPACE_ROOT}/paper/canon/embeddings/oxford/A=ViT-B-32-openai__B=ViT-B-32-laion400m_e31/ViT-B-32_openai_ViT-B-32_laion400m_e31"
elif [[ "${PAIR}" == "openai-flava" ]]; then
  CONFIG="${PROJECT_ROOT}/configs/frozen_decoder/openai_flava_cub.json"
  ALIGNMENT="${ALIGNMENT_ROOT}/oxford_openai_vitl14_to_flava.pt"
  OXFORD_PREFIX="${WORKSPACE_ROOT}/paper/canon/embeddings/oxford/A=ViT-L-14-openai__B=FLAVA-facebook_flava-full/ViT-L-14_openai_flava_facebook_flava-full"
else
  echo "unknown pair: ${PAIR}" >&2
  exit 2
fi

"${PYTHON}" -m canonical_study validate-cub \
  --data-root "${CUB_DATA_ROOT}"

if [[ ! -f "${ALIGNMENT}" ]]; then
  "${PYTHON}" -m canonical_study materialize-oxford-alignment \
    --config "${CONFIG}" \
    --upstream-embedding-prefix "${OXFORD_PREFIX}" \
    --output "${ALIGNMENT}"
fi

"${PYTHON}" -m canonical_study collect-env \
  --output "${OUTPUT_ROOT}/${PAIR}.environment.json"

"${PYTHON}" -m canonical_study run-frozen-decoder \
  --config "${CONFIG}" \
  --alignment "${ALIGNMENT}" \
  --data-root "${CUB_DATA_ROOT}" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --prediction-root "${PREDICTION_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --model-cache-root "${MODEL_CACHE_ROOT}" \
  --device cuda 2>&1 | tee "${LOG_ROOT}/${PAIR}.live.log"
