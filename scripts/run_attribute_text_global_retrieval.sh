#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CANONICAL_STUDY_PYTHON:-python3}"
DEVICE="${CANONICAL_STUDY_DEVICE:-cuda}"
MODEL_CACHE_ROOT="${CANONICAL_STUDY_MODEL_CACHE_ROOT:-${PROJECT_ROOT}/artifacts/cache/models}"
FLAVA_MODEL_CACHE_ROOT="${CANONICAL_STUDY_FLAVA_MODEL_CACHE_ROOT:-${MODEL_CACHE_ROOT}}"
if [[ ! -d "${FLAVA_MODEL_CACHE_ROOT}/huggingface/models--facebook--flava-full" ]] \
  && [[ -d "/tmp/canonical-study-model-cache/huggingface/models--facebook--flava-full" ]]; then
  FLAVA_MODEL_CACHE_ROOT="/tmp/canonical-study-model-cache"
fi
EMBEDDING_ROOT="${CANONICAL_STUDY_CUB_EMBEDDING_ROOT:-${PROJECT_ROOT}/artifacts/embeddings/cub}"
ALIGNMENT_ROOT="${CANONICAL_STUDY_ALIGNMENT_ROOT:-${PROJECT_ROOT}/artifacts/alignments}"
OUTPUT_ROOT="${CANONICAL_STUDY_ATTRIBUTE_TEXT_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/results/attribute_text_retrieval}"
AUDIT_ROOT="${CANONICAL_STUDY_ATTRIBUTE_TEXT_AUDIT_ROOT:-${OUTPUT_ROOT}/audit}"
TEXT_BATCH_SIZE="${CANONICAL_STUDY_TEXT_BATCH_SIZE:-64}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m canonical_study attribute-text-global-retrieval \
  --config "${PROJECT_ROOT}/configs/frozen_decoder/openai_laion_cub.json" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --alignment-root "${ALIGNMENT_ROOT}" \
  --model-cache-root "${MODEL_CACHE_ROOT}" \
  --audit-root "${AUDIT_ROOT}" \
  --text-batch-size "${TEXT_BATCH_SIZE}" \
  --device "${DEVICE}" \
  "$@"

"${PYTHON_BIN}" -m canonical_study attribute-text-global-retrieval \
  --config "${PROJECT_ROOT}/configs/frozen_decoder/openai_flava_cub.json" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --alignment-root "${ALIGNMENT_ROOT}" \
  --model-cache-root "${FLAVA_MODEL_CACHE_ROOT}" \
  --audit-root "${AUDIT_ROOT}" \
  --text-batch-size "${TEXT_BATCH_SIZE}" \
  --device "${DEVICE}" \
  "$@"
