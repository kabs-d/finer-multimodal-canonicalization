#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${1:-cub-train-q-control}"
MINIMUM_FREE_MIB="${MINIMUM_FREE_MIB:-8192}"
DEVICE="${CANONICAL_STUDY_DEVICE:-cuda}"
TMUX_SOCKET="${CANONICAL_STUDY_TMUX_SOCKET:-}"
PYTHON="${CANONICAL_STUDY_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
CUB_DATA_ROOT="${CANONICAL_STUDY_CUB_DATA_ROOT:-${PROJECT_ROOT}/artifacts/data/cub}"
MODEL_CACHE_ROOT="${CANONICAL_STUDY_MODEL_CACHE_ROOT:-${PROJECT_ROOT}/artifacts/cache/models}"
EMBEDDING_ROOT="${CANONICAL_STUDY_CUB_EMBEDDING_ROOT:-${PROJECT_ROOT}/artifacts/embeddings/cub}"
PREDICTION_ROOT="${CANONICAL_STUDY_CUB_PREDICTION_ROOT:-${PROJECT_ROOT}/artifacts/predictions/cub}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "runtime Python is not executable: ${PYTHON}" >&2
  exit 1
fi
if [[ -n "${TMUX_SOCKET}" ]]; then
  mkdir -p "$(dirname "${TMUX_SOCKET}")"
fi
"${PYTHON}" -m canonical_study validate-cub --data-root "${CUB_DATA_ROOT}"

TMUX_CMD=(tmux)
if [[ -n "${TMUX_SOCKET}" ]]; then
  TMUX_CMD=(tmux -S "${TMUX_SOCKET}")
fi

if [[ "${DEVICE}" == "cuda" ]]; then
  mapfile -t GPU_IDS < <(
    nvidia-smi --query-gpu=index,memory.free \
      --format=csv,noheader,nounits |
    awk -F, -v minimum="${MINIMUM_FREE_MIB}" \
      '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 >= minimum) print $1, $2}' |
    sort -k2,2nr |
    head -2 |
    awk '{print $1}'
  )

  if [[ "${#GPU_IDS[@]}" -lt 2 ]]; then
    echo "need two GPUs with at least ${MINIMUM_FREE_MIB} MiB free" >&2
    exit 1
  fi
else
  GPU_IDS=("cpu" "cpu")
fi
if "${TMUX_CMD[@]}" has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

COMMON_ENV="PYTHONPATH=${PROJECT_ROOT}/src CANONICAL_STUDY_PYTHON=${PYTHON} CANONICAL_STUDY_DEVICE=${DEVICE} CANONICAL_STUDY_CUB_DATA_ROOT=${CUB_DATA_ROOT} CANONICAL_STUDY_MODEL_CACHE_ROOT=${MODEL_CACHE_ROOT} CANONICAL_STUDY_CUB_EMBEDDING_ROOT=${EMBEDDING_ROOT} CANONICAL_STUDY_CUB_PREDICTION_ROOT=${PREDICTION_ROOT}"
"${TMUX_CMD[@]}" new-session -d -s "${SESSION_NAME}" -n openai-laion \
  "${COMMON_ENV} ${PROJECT_ROOT}/scripts/run_cub_train_q_control_pair.sh openai-laion ${GPU_IDS[0]}"
"${TMUX_CMD[@]}" set-option -t "${SESSION_NAME}" remain-on-exit on
"${TMUX_CMD[@]}" new-window -t "${SESSION_NAME}" -n openai-flava \
  "${COMMON_ENV} ${PROJECT_ROOT}/scripts/run_cub_train_q_control_pair.sh openai-flava ${GPU_IDS[1]}"
if [[ "${DEVICE}" == "cuda" ]]; then
  "${TMUX_CMD[@]}" new-window -t "${SESSION_NAME}" -n monitor \
    "watch -n 5 nvidia-smi"
else
  "${TMUX_CMD[@]}" new-window -t "${SESSION_NAME}" -n monitor \
    "while true; do date; sleep 60; done"
fi

if [[ "${DEVICE}" == "cuda" ]]; then
  echo "launched ${SESSION_NAME}: OpenAI-LAION on GPU ${GPU_IDS[0]}, OpenAI-FLAVA on GPU ${GPU_IDS[1]}"
else
  echo "launched ${SESSION_NAME}: OpenAI-LAION and OpenAI-FLAVA on CPU"
fi
if [[ -n "${TMUX_SOCKET}" ]]; then
  echo "attach with: tmux -S ${TMUX_SOCKET} attach -t ${SESSION_NAME}"
else
  echo "attach with: tmux attach -t ${SESSION_NAME}"
fi
