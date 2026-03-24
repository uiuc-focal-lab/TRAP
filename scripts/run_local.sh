#!/bin/bash
# =============================================================================
# run_local.sh — Run TRAP locally without HTCondor or Apptainer
# =============================================================================
# Usage:
#   bash scripts/run_local.sh train    [train.py args...]
#   bash scripts/run_local.sh eval     [trap_framework_eval.py args...]
#   bash scripts/run_local.sh precache [precache_vlm_models.py args...]
#
# Run from the TRAP repository root directory.
# Edit scripts/config.sh to configure paths before running.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

SRC_DIR="${TRAP_ROOT}/src"

# Create all required directories up front.
mkdir -p "${TRAP_WEIGHTS_DIR}" \
         "${TRAP_OUTPUTS_DIR}" \
         "${HF_HOME}" \
         "${TORCH_HOME}" \
         "${TMPDIR}" \
         "${XDG_CACHE_HOME}" \
         "${TRAP_PACKAGES_DIR}"

export PYTHONPATH="${TRAP_PACKAGES_DIR}:${PYTHONPATH:-}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=0
export SOFT_FILELOCK=1

# Install lpips into TRAP_PACKAGES_DIR if not already present.
python - <<PY
import importlib, subprocess, sys
target = "${TRAP_PACKAGES_DIR}"
sys.path.insert(0, target)
try:
    lpips = importlib.import_module("lpips")
    _ = lpips.LPIPS(net="alex")
    print("[TRAP] lpips already installed.")
except Exception:
    print("[TRAP] Installing lpips into", target)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--target", target,
         "--no-cache-dir", "--no-deps", "--upgrade", "lpips"]
    )
    lpips = importlib.import_module("lpips")
    _ = lpips.LPIPS(net="alex")
    print("[TRAP] lpips install complete.")
PY

COMMAND="${1:-}"
shift || true

case "${COMMAND}" in
  train)
    echo "[TRAP] Running: train"
    python "${SRC_DIR}/train.py" \
      --save_dir "${TRAP_WEIGHTS_DIR}" \
      "$@"
    ;;

  eval)
    echo "[TRAP] Running: eval"
    # Load the default eval model list if no --eval_models / --eval_model was passed.
    eval_args=("$@")
    has_eval_models=0
    has_eval_model=0
    has_eval_trust_remote_code=0
    for arg in "${eval_args[@]}"; do
      case "${arg}" in
        --eval_models) has_eval_models=1 ;;
        --eval_model) has_eval_model=1 ;;
        --eval_trust_remote_code) has_eval_trust_remote_code=1 ;;
      esac
    done
    if [ "${has_eval_models}" -eq 0 ] && [ -f "${SCRIPT_DIR}/model_lists.sh" ]; then
      source "${SCRIPT_DIR}/model_lists.sh"
      model_csv="$(trap_vlm_models_csv)"
      if [ -n "${model_csv}" ]; then
        echo "[TRAP] Using default eval model list: ${model_csv}"
        eval_args+=(--eval_models "${model_csv}")
        if [ "${has_eval_model}" -eq 0 ] && [ "${#TRAP_EVAL_MODEL_IDS[@]}" -gt 0 ]; then
          eval_args+=(--eval_model "${TRAP_EVAL_MODEL_IDS[0]}")
        fi
      fi
    fi
    if [ "${has_eval_trust_remote_code}" -eq 0 ]; then
      eval_args+=(--eval_trust_remote_code)
    fi
    python "${SRC_DIR}/trap_framework_eval.py" \
      --weights_dir "${TRAP_WEIGHTS_DIR}" \
      --output_dir  "${TRAP_OUTPUTS_DIR}" \
      "${eval_args[@]}"
    ;;

  precache)
    echo "[TRAP] Running: precache"
    # Load the default model list if no --repo_id / --repo_ids was passed.
    args=("$@")
    has_repo_flag=0
    for arg in "${args[@]}"; do
      case "${arg}" in
        --repo_id|--repo_ids) has_repo_flag=1 ;;
      esac
    done
    if [ "${has_repo_flag}" -eq 0 ]; then
      source "${SCRIPT_DIR}/model_lists.sh"
      model_csv="$(trap_vlm_models_csv)"
      if [ -n "${model_csv}" ]; then
        echo "[TRAP] Using default model list: ${model_csv}"
        args+=(--repo_ids "${model_csv}")
      fi
    fi
    python "${SRC_DIR}/precache_vlm_models.py" "${args[@]}"
    ;;

  *)
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  train    -- Train TRAP semantic networks (SiameseSemanticNetwork + SemanticLayoutGenerator)"
    echo "  eval     -- Run evaluation pipeline (add --stage generate|eval|both)"
    echo "  precache -- Pre-download HuggingFace VLM model weights"
    echo ""
    echo "Examples:"
    echo "  bash scripts/run_local.sh train --epochs 20 --batch_size 32"
    echo "  bash scripts/run_local.sh eval --stage generate --sd_model Manojb/stable-diffusion-2-1-base"
    echo "  bash scripts/run_local.sh eval --stage eval --eval_model Qwen/Qwen2.5-VL-32B-Instruct"
    echo "  bash scripts/run_local.sh precache"
    exit 1
    ;;
esac

echo "[TRAP] Done."
