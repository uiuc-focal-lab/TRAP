#!/bin/bash
# run_eval.sh — Cluster wrapper for TRAP evaluation (sourced by HTCondor jobs).
# For local runs, use scripts/run_local.sh instead.
set -euo pipefail

echo "Starting TRAP eval job on Cluster ${CLUSTER_ID:-} with Job ID ${JOB_ID:-}"

# Load path configuration. Inside the Apptainer container this script is at
# /code/scripts/run_eval.sh so config.sh resolves to /code/scripts/config.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}" "${XDG_CACHE_HOME}" \
         "${TRAP_PACKAGES_DIR}" "${TRAP_WEIGHTS_DIR}" "${TRAP_OUTPUTS_DIR}"

export PYTHONPATH="${TRAP_PACKAGES_DIR}:${PYTHONPATH:-}"
export SOFT_FILELOCK=1
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# Condor may assign one healthy GPU on hosts where another GPU is unhealthy.
# Restrict CUDA visibility early so torch does not probe all devices.
if [ -n "${_CONDOR_AssignedGPUs:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${_CONDOR_AssignedGPUs}"
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ -z "${NVIDIA_VISIBLE_DEVICES:-}" ]; then
  export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi
echo "[ENV] _CONDOR_AssignedGPUs=${_CONDOR_AssignedGPUs:-}"
echo "[ENV] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[ENV] NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-}"

# Install lpips into TRAP_PACKAGES_DIR if not already present.
PACKAGES_TARGET="${TRAP_PACKAGES_DIR}"
python - <<PY
import importlib, subprocess, sys
target = "${PACKAGES_TARGET}"
sys.path.insert(0, target)
try:
    lpips = importlib.import_module("lpips")
    _ = lpips.LPIPS(net="alex")
    print("[ENV] lpips already installed.")
except Exception:
    print("[ENV] Installing lpips into", target)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--target", target,
         "--no-cache-dir", "--no-deps", "--upgrade", "lpips"]
    )
    lpips = importlib.import_module("lpips")
    _ = lpips.LPIPS(net="alex")
    print("[ENV] lpips install complete.")
PY

python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python - <<'PY'
import os, torch
if not torch.cuda.is_available():
    print("[EVAL_WARN] CUDA unavailable. TRAP will run extremely slowly; results may be invalid.")
    print(f"[EVAL_WARN] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[EVAL_WARN] torch.__version__={torch.__version__} torch.version.cuda={torch.version.cuda}")
    try:
        print(f"[EVAL_WARN] torch.cuda.device_count()={torch.cuda.device_count()}")
    except Exception as e:
        print(f"[EVAL_WARN] torch.cuda.device_count() error: {e!r}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

if [ "${TRAP_REQUIRE_CUDA:-1}" = "1" ]; then
  python - <<'PY'
import os, socket, sys, torch
if not torch.cuda.is_available():
    print("[EVAL_ERROR] CUDA is unavailable; exiting to avoid invalid/slow CPU eval.")
    print(f"[EVAL_ERROR] host={socket.gethostname()} _CONDOR_AssignedGPUs={os.environ.get('_CONDOR_AssignedGPUs')}")
    sys.exit(42)
PY
fi

args=("$@")
stage="both"
has_eval_models=0
has_eval_model=0
has_eval_trust_remote_code=0
has_weights_dir=0
has_output_dir=0
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[$i]}" in
    --stage)
      if [ $((i + 1)) -lt ${#args[@]} ]; then
        stage="${args[$((i + 1))]}"
      fi
      i=$((i + 1))
      ;;
    --eval_models)
      has_eval_models=1
      i=$((i + 1))
      ;;
    --eval_model)
      has_eval_model=1
      i=$((i + 1))
      ;;
    --eval_trust_remote_code)
      has_eval_trust_remote_code=1
      ;;
    --weights_dir)
      has_weights_dir=1
      i=$((i + 1))
      ;;
    --output_dir)
      has_output_dir=1
      i=$((i + 1))
      ;;
  esac
done

if [ "${has_weights_dir}" -eq 0 ]; then
  args+=(--weights_dir "${TRAP_WEIGHTS_DIR}")
fi
if [ "${has_output_dir}" -eq 0 ]; then
  args+=(--output_dir "${TRAP_OUTPUTS_DIR}")
fi

MODEL_LISTS="${SCRIPT_DIR}/model_lists.sh"
if [ "${has_eval_models}" -eq 0 ] && [ -f "${MODEL_LISTS}" ]; then
  # shellcheck disable=SC1090
  source "${MODEL_LISTS}"
  model_csv="$(trap_vlm_models_csv)"
  if [ -n "${model_csv}" ]; then
    echo "[EVAL] Using shared eval model list: ${model_csv}"
    args+=(--eval_models "${model_csv}")
    if [ "${has_eval_model}" -eq 0 ] && [ "${#TRAP_EVAL_MODEL_IDS[@]}" -gt 0 ]; then
      args+=(--eval_model "${TRAP_EVAL_MODEL_IDS[0]}")
    fi
  fi
fi

if [ "${has_eval_trust_remote_code}" -eq 0 ]; then
  args+=(--eval_trust_remote_code)
fi

python "${SCRIPT_DIR}/../src/trap_framework_eval.py" "${args[@]}"

echo "TRAP eval complete."
