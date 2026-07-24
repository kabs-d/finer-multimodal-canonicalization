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
UPSTREAM_ROOT="${WORKSPACE_ROOT}/paper/canon/canonical-multimodal-rep"
PYTHON="${CANONICAL_STUDY_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
DATA_ROOT="${CANONICAL_STUDY_DATA_ROOT:-${PROJECT_ROOT}/artifacts/data}"
LOG_ROOT="${PROJECT_ROOT}/artifacts/logs"
MODEL_CACHE_ROOT="${CANONICAL_STUDY_MODEL_CACHE_ROOT:-${PROJECT_ROOT}/artifacts/cache/models}"
EMBEDDING_ROOT="${CANONICAL_STUDY_EMBEDDING_ROOT:-${PROJECT_ROOT}/artifacts/embeddings}"
OUTPUT_ROOT="${PROJECT_ROOT}/artifacts/results/standalone"
UPSTREAM_LOG="${LOG_ROOT}/${PAIR}.upstream.log"
STANDALONE_LOG="${LOG_ROOT}/${PAIR}.standalone.log"

mkdir -p "${LOG_ROOT}" "${MODEL_CACHE_ROOT}" "${EMBEDDING_ROOT}" "${OUTPUT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export HF_HOME="${MODEL_CACHE_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${MODEL_CACHE_ROOT}/huggingface"

if [[ "${PAIR}" == "openai-laion" ]]; then
  CONFIG="${PROJECT_ROOT}/configs/baseline/openai_laion_oxford.json"
  UPSTREAM_EMBEDDING_PREFIX="${WORKSPACE_ROOT}/paper/canon/embeddings/oxford/A=ViT-B-32-openai__B=ViT-B-32-laion400m_e31/ViT-B-32_openai_ViT-B-32_laion400m_e31"
  if [[ "${CANONICAL_STUDY_SKIP_UPSTREAM:-0}" != "1" ]]; then
    (
      cd "${UPSTREAM_ROOT}"
      "${PYTHON}" main.py \
        --dataset oxford \
        --data_root "${DATA_ROOT}" \
        --clip_model_1 ViT-B-32 \
        --pretrained_1 openai \
        --clip_model_2 ViT-B-32 \
        --pretrained_2 laion400m_e31 \
        --seeds 42,43,44
    ) 2>&1 | tee "${UPSTREAM_LOG}"
  fi
elif [[ "${PAIR}" == "openai-flava" ]]; then
  CONFIG="${PROJECT_ROOT}/configs/baseline/openai_flava_oxford.json"
  UPSTREAM_EMBEDDING_PREFIX="${WORKSPACE_ROOT}/paper/canon/embeddings/oxford/A=ViT-L-14-openai__B=FLAVA-facebook_flava-full/ViT-L-14_openai_flava_facebook_flava-full"
  if [[ "${CANONICAL_STUDY_SKIP_UPSTREAM:-0}" != "1" ]]; then
    (
      cd "${UPSTREAM_ROOT}"
      "${PYTHON}" main.py \
        --dataset oxford \
        --data_root "${DATA_ROOT}" \
        --clip_model_1 ViT-L-14 \
        --pretrained_1 openai \
        --clip_model_2 FLAVA \
        --pretrained_2 facebook/flava-full \
        --seeds 42,43,44
    ) 2>&1 | tee "${UPSTREAM_LOG}"
  fi
else
  echo "unknown pair: ${PAIR}" >&2
  exit 2
fi

"${PYTHON}" -m canonical_study run \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --model-cache-root "${MODEL_CACHE_ROOT}" \
  --upstream-embedding-prefix "${UPSTREAM_EMBEDDING_PREFIX}" \
  --device cuda 2>&1 | tee "${STANDALONE_LOG}"
