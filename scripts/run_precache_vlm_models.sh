#!/bin/bash
# run_precache_vlm_models.sh — Cluster wrapper for VLM model pre-caching.
# For local runs, use scripts/run_local.sh precache instead.
set -euo pipefail

echo "Starting VLM precache job on Cluster ${CLUSTER_ID:-} with Job ID ${JOB_ID:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}" "${XDG_CACHE_HOME}"

export SOFT_FILELOCK=1
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=0

args=("$@")
has_repo_flag=0
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[$i]}" in
    --repo_id|--repo_ids)
      has_repo_flag=1
      i=$((i + 1))
      ;;
  esac
done

MODEL_LISTS="${SCRIPT_DIR}/model_lists.sh"
if [ "${has_repo_flag}" -eq 0 ] && [ -f "${MODEL_LISTS}" ]; then
  # shellcheck disable=SC1090
  source "${MODEL_LISTS}"
  model_csv="$(trap_vlm_models_csv)"
  if [ -n "${model_csv}" ]; then
    echo "[PRECACHE] Using shared model list: ${model_csv}"
    args+=(--repo_ids "${model_csv}")
  fi
fi

python "${SCRIPT_DIR}/../src/precache_vlm_models.py" "${args[@]}"

echo "VLM precache complete."
