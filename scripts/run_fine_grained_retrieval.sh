#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CANONICAL_STUDY_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
EMBEDDING_ROOT="${CANONICAL_STUDY_CUB_EMBEDDING_ROOT:-${PROJECT_ROOT}/artifacts/embeddings/cub}"
OUTPUT_ROOT="${PROJECT_ROOT}/artifacts/results/fine_grained_retrieval"
ALIGNMENT_ROOT="${PROJECT_ROOT}/artifacts/alignments"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -m canonical_study fine-grained-retrieval \
  --config "${PROJECT_ROOT}/configs/frozen_decoder/openai_laion_cub.json" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --alignment-root "${ALIGNMENT_ROOT}" \
  --force

"${PYTHON}" -m canonical_study fine-grained-retrieval \
  --config "${PROJECT_ROOT}/configs/frozen_decoder/openai_flava_cub.json" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --alignment-root "${ALIGNMENT_ROOT}" \
  --force
