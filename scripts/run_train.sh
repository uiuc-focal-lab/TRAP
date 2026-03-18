#!/bin/bash
# run_train.sh — Cluster wrapper for TRAP training (sourced by HTCondor jobs).
# For local runs, use scripts/run_local.sh instead.
set -euo pipefail

echo "Starting TRAP Training Job on Cluster ${CLUSTER_ID:-} with Job ID ${JOB_ID:-}"

# Load path configuration. Inside the Apptainer container this script is at
# /code/scripts/run_train.sh so config.sh resolves to /code/scripts/config.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}" "${XDG_CACHE_HOME}" \
         "${TRAP_PACKAGES_DIR}" "${TRAP_WEIGHTS_DIR}"

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

python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

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

python "${SCRIPT_DIR}/../src/train.py" \
  --save_dir "${TRAP_WEIGHTS_DIR}" \
  "$@"

echo "Training complete."
