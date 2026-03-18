#!/bin/bash

# Shared default VLM list used by both precache and eval flows.
# Keep this list small and editable; each entry is a Hugging Face repo id.
# Targeting roughly 20B-30B class models across different families.
TRAP_EVAL_MODEL_IDS=(
  "Qwen/Qwen2.5-VL-32B-Instruct"
  # "llava-hf/llava-v1.6-34b-hf"
)

trap_vlm_models_csv() {
  local IFS=,
  echo "${TRAP_EVAL_MODEL_IDS[*]}"
}
