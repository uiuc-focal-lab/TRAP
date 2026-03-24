<div align="center">

<h1>TRAP: Targeted Redirecting of Agentic Preferences</h1>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv)](https://arxiv.org/abs/2505.23518)
[![GitHub](https://img.shields.io/badge/GitHub-TRAP-green?logo=github&logoColor=white)](https://github.com/uiuc-focal-lab/TRAP)

</div>

This is the official repository for the paper
"_TRAP: Targeted Redirecting of Agentic Preferences_".

Authors:
[Hangoo Kang*](https://hgkang02.github.io/),
[Jehyeok Yeon*](https://jeybird248.github.io/),
[Gagandeep Singh](https://ggndpsngh.github.io/)
(* Equal Contribution)
---

## Overview

TRAP is a framework for generating semantic-aware adversarial image perturbations that redirect the preferences of Vision-Language Model (VLM) agents. Given a set of candidate images presented to a VLM in a multiple-choice setting, TRAP perturbs one image to "inject" semantic information into the image without relying on random noise. The perturbations are guided by learned semantic structure (via CLIP) and applied through a Stable Diffusion img2img pipeline.

---

## Getting Started

### Installation

1. Install [PyTorch](https://pytorch.org/get-started/locally/) for your CUDA version first.

2. Install the remaining dependencies:
   ```bash
   pip install -r requirements.txt
   pip install git+https://github.com/openai/CLIP.git
   ```

   `clip` is not on PyPI and must be installed directly from GitHub.

---

## Configuration

Before running anything, open [`scripts/config.sh`](scripts/config.sh) and review the path settings. By default, everything (weights, outputs, caches) is stored inside the TRAP repo directory. Override any setting by editing `config.sh` or by exporting environment variables before running a script - environment variables always take precedence.

Key settings in `config.sh`:

| Variable | Default | Description |
|---|---|---|
| `TRAP_WEIGHTS_DIR` | `<repo>/trap_weights` | Where trained model checkpoints are saved/loaded |
| `TRAP_OUTPUTS_DIR` | `<repo>/trap_eval_outputs` | Where evaluation images and results are saved |
| `HF_HOME` | `<repo>/hf_cache` | HuggingFace model cache |
| `TORCH_HOME` | `<repo>/torch_cache` | PyTorch model cache |
| `TRAP_CONTAINER_SIF` | _(unset)_ | Path to Apptainer .sif image (cluster runs only) |

For **gated HuggingFace models** (some LLaVA variants), also set `HF_TOKEN` in `config.sh` or export it before running.

---

## Usage - Local Runs

Use [`scripts/run_local.sh`](scripts/run_local.sh) to run any stage directly. Run all commands from the TRAP repository root.

### 1. (Optional) Pre-cache VLM Models

Download VLM weights to your local cache ahead of time to avoid timeouts during evaluation:

```bash
bash scripts/run_local.sh precache
# or specify models explicitly:
bash scripts/run_local.sh precache --repo_ids "Qwen/Qwen2.5-VL-32B-Instruct"
```

Edit [`scripts/model_lists.sh`](scripts/model_lists.sh) to change the default model list.

### 2. Training

Train the TRAP semantic networks on COCO-Stuff-Captioned (auto-downloaded from HuggingFace):

```bash
bash scripts/run_local.sh train --epochs 20 --batch_size 32 --lr 5e-3
```

Or call `src/train.py` directly for full control:

```bash
python src/train.py \
  --epochs 20 \
  --batch_size 32 \
  --lr 5e-3 \
  --save_dir trap_weights \
  --distinct_weight 0.3
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `20` | Number of training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `5e-3` | Learning rate |
| `--save_dir` | `./trap_weights` | Directory to save model checkpoints |
| `--distinct_weight` | `0.3` | Weight for the distinctive-identity anchor loss |

Outputs per epoch: `siamese_epoch_N.pt`, `layout_epoch_N.pt`, and `training_stats.json`.

### 3. Evaluation

The evaluation pipeline (`src/trap_framework_eval.py`) has two stages.

#### Generation stage - produce adversarial image variants

```bash
bash scripts/run_local.sh eval \
  --stage generate \
  --sd_model Manojb/stable-diffusion-2-1-base \
  --hf_dataset SargeZT/coco-stuff-captioned \
  --split train \
  --sample_size 30 \
  --n_variations 4 \
  --runs_per_image 20
```

#### Scoring stage - evaluate generated images with a VLM

```bash
bash scripts/run_local.sh eval \
  --stage eval \
  --eval_model "Qwen/Qwen2.5-VL-32B-Instruct" \
  --eval_strategy debiased \
  --sample_size 30 \
  --runs_per_image 20
```

#### Full pipeline (both stages)

```bash
bash scripts/run_local.sh eval \
  --stage both \
  --sd_model Manojb/stable-diffusion-2-1-base
```

`--weights_dir` and `--output_dir` are set automatically from `config.sh`. Pass them explicitly to override:

```bash
bash scripts/run_local.sh eval --stage both \
  --weights_dir /path/to/trap_weights \
  --output_dir /path/to/outputs \
  --sd_model Manojb/stable-diffusion-2-1-base
```

Key evaluation arguments:

| Argument | Default | Description |
|---|---|---|
| `--stage` | `both` | `generate`, `eval`, or `both` |
| `--weights_dir` | from `config.sh` | Path to trained TRAP model checkpoints |
| `--output_dir` | from `config.sh` | Directory for generated images and results |
| `--sd_model` | `Manojb/stable-diffusion-2-1-base` | Stable Diffusion model (HuggingFace repo id or local path) |
| `--eval_model` | `Qwen/Qwen2.5-VL-32B-Instruct` | Primary VLM for scoring |
| `--eval_models` | auto from `model_lists.sh` | Comma-separated list of VLMs for multi-model evaluation |
| `--eval_strategy` | `debiased` | `auto`, `grid`, or `debiased` |
| `--eval_trust_remote_code` | auto-injected | Pass `trust_remote_code=True` when loading VLMs |
| `--sample_size` | `30` | Number of samples to process per run |
| `--n_variations` | `4` | Number of candidate images in the multiple-choice set |
| `--runs_per_image` | `20` | Number of random shuffles for n-way voting |
| `--attack_outer_steps` | `32` | Number of TRAP outer optimization iterations |
| `--attack_lr` | `0.15` | TRAP optimization learning rate |
| `--attack_eps` | `6.0` | L2 perturbation radius on the unit sphere |
| `--prompt_token_blend` | `0.4` | Residual blend scale for token-level prompt embeddings |
| `--lambda_sem` | `0.2` | Weight for semantic-preservation loss |
| `--attack_eval_runs` | `10` | Mini evaluator runs per outer step (0 disables) |
| `--attack_eval_early_stop` | `False` | Stop outer loop early once above-chance (use `--no_attack_eval_early_stop`) |

---

## Usage - HTCondor Cluster

### Prerequisites

1. **Build or pull** an Apptainer container image with the required Python dependencies. The `.sub` files assume the container is accessible from all execute nodes (e.g., on shared/Lustre storage).

2. **Create the logs directory** (HTCondor writes `.err`/`.out`/`.log` files here):
   ```bash
   mkdir -p /path/to/TRAP/logs
   ```

3. **Set required environment variables:**
   ```bash
   export TRAP_CONTAINER_SIF=/path/to/trap_container.sif
   ```
   Optionally override storage paths (all default to inside the repo):
   ```bash
   export TRAP_OUTPUTS_DIR=/fast/storage/trap_eval_outputs
   export HF_HOME=/fast/storage/hf_cache
   ```
   These are inherited by submitted jobs via `getenv = True`.

### Submitting Jobs

**Always `cd` into the TRAP root before submitting.** The `.sub` files use `$ENV(PWD)` to bind-mount the repo into the container, so the working directory must be the TRAP root.

```bash
cd /path/to/TRAP

# Pre-cache VLM model weights
condor_submit scripts/precache_vlm_models.sub

# Training
condor_submit scripts/train_container.sub

# Evaluation - generation stage only
condor_submit scripts/eval_generate_container.sub

# Evaluation - scoring stage only
condor_submit scripts/eval_score_container.sub

# Evaluation - full pipeline (generate + score)
condor_submit scripts/eval_pipeline_container.sub
```

The entire TRAP directory is bind-mounted into the container at `/code`, so no file transfer is needed. Weights, outputs, and caches all resolve to paths inside the repo via `scripts/config.sh` unless overridden with environment variables.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{kang2025traptargetedredirectingagentic,
      title={TRAP: Targeted Redirecting of Agentic Preferences},
      author={Hangoo Kang and Jehyeok Yeon and Gagandeep Singh},
      year={2025},
      eprint={2505.23518},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.23518},
}
```
