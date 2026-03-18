#!/bin/bash
# =============================================================================
# config.sh — User-editable TRAP path configuration
# =============================================================================
# Edit the values in this file to match your environment.
# All scripts source this file automatically.
#
# Environment variables always take precedence over the defaults defined here,
# so you can also override any setting by exporting it before running a script.
#
# FOR CLUSTER (HTCondor) RUNS:
#   1. Set TRAP_CONTAINER_SIF to the path of your Apptainer .sif image on
#      shared cluster storage accessible from all execute nodes.
#   2. Optionally export TRAP_WEIGHTS_DIR, TRAP_OUTPUTS_DIR, HF_HOME, etc.
#      to point at fast/persistent storage before running condor_submit.
#      (The .sub files pass getenv=True so submitted jobs inherit these.)
# =============================================================================

# Root of the TRAP repository.
# Auto-detected from this file's location; override only if needed.
TRAP_ROOT="${TRAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Where trained TRAP weights are saved and loaded.
TRAP_WEIGHTS_DIR="${TRAP_WEIGHTS_DIR:-${TRAP_ROOT}/trap_weights}"

# Where evaluation outputs (generated images, manifests, results) are saved.
TRAP_OUTPUTS_DIR="${TRAP_OUTPUTS_DIR:-${TRAP_ROOT}/trap_eval_outputs}"

# HuggingFace model cache.
# On a cluster, point this at fast/large shared storage to avoid re-downloading.
export HF_HOME="${HF_HOME:-${TRAP_ROOT}/hf_cache}"

# PyTorch model cache.
export TORCH_HOME="${TORCH_HOME:-${TRAP_ROOT}/torch_cache}"

# Temporary scratch space.
export TMPDIR="${TMPDIR:-${TRAP_ROOT}/tmp_cache}"

# XDG cache directory.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TRAP_ROOT}/xdg_cache}"

# Directory for auxiliary Python packages installed at runtime (e.g. lpips).
TRAP_PACKAGES_DIR="${TRAP_PACKAGES_DIR:-${TRAP_ROOT}/trap_packages}"

# HuggingFace token — required for gated models (e.g. some LLaVA variants).
# Set via environment or uncomment the line below:
# export HF_TOKEN="your_token_here"

# =============================================================================
# Cluster-only settings (ignored for local runs)
# =============================================================================

# Path to your Apptainer/Singularity .sif container image.
# Must be on shared storage accessible from all HTCondor execute nodes.
# Build or pull a container with: PyTorch, diffusers, transformers, CLIP, lpips.
# Example: TRAP_CONTAINER_SIF="/cluster/shared/trap_container.sif"
TRAP_CONTAINER_SIF="${TRAP_CONTAINER_SIF:-}"
