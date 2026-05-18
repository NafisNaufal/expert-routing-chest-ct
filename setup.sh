#!/usr/bin/env bash
# setup.sh — one-time environment setup on the A100 server.
# Usage:  bash setup.sh
#
# Target host: 1x A100 80GB, NVIDIA driver 470.x, CUDA 11.6.
# This driver only supports the cu118 PyTorch build — cu12x needs driver >=525.
#
# Strategy: install ONE coherent cu118 stack, then install VILA's `llava`
# package with --no-deps so it contributes Python code only and does NOT
# drag in its own (cu12-oriented) version pins.
set -e

CONDA_ENV="icsdg"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VILA_FRAMEWORK="$PROJECT_ROOT/VLM-Radiology-Agent-Framework"
VILA_REPO="$VILA_FRAMEWORK/thirdparty/VILA"

# flash-attn prebuilt wheel matching torch 2.3 + cu118 + python 3.10 (cxx11abiFALSE).
FLASH_ATTN_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# ── 0. Sanity checks ──────────────────────────────────────────────────────────
if [ ! -d "$VILA_REPO/llava" ]; then
    echo "ERROR: VILA submodule not found at $VILA_REPO/llava"
    echo "Run:  git -C $VILA_FRAMEWORK submodule update --init --recursive"
    exit 1
fi

# ── 1. Conda environment ──────────────────────────────────────────────────────
echo "==> Creating conda environment: $CONDA_ENV (python=3.10)"
conda create -y -n "$CONDA_ENV" python=3.10
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
pip install --upgrade pip

# nvcc 11.8 — required only so `import deepspeed` (pulled in by llava's import
# chain) can run its op-compatibility probe. We never build deepspeed ops.
echo "==> Installing cuda-nvcc 11.8 into the env"
conda install -y -c nvidia cuda-nvcc=11.8
conda env config vars set CUDA_HOME="$CONDA_PREFIX" -n "$CONDA_ENV"
export CUDA_HOME="$CONDA_PREFIX"

# ── 2. PyTorch (cu118 — REQUIRED for driver 470) ──────────────────────────────
echo "==> Installing PyTorch 2.3.0 + cu118"
pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu118

# ── 3. flash-attention (prebuilt wheel — no nvcc / no compilation) ────────────
echo "==> Installing flash-attn 2.5.8 (prebuilt cu118 wheel)"
if ! pip install "$FLASH_ATTN_WHL"; then
    echo ""
    echo "WARNING: flash-attn wheel install failed."
    echo "Training/eval can still run with SDPA attention — pass"
    echo "  --attn_implementation sdpa  to the train/eval scripts."
    echo ""
fi

# ── 4. Project + VILA-core dependencies (pinned, cu118-coherent) ──────────────
echo "==> Installing pinned project requirements"
pip install -r "$PROJECT_ROOT/requirements.txt"

# ── 5. VILA `llava` package — CODE ONLY (no deps, no version pins) ────────────
echo "==> Installing VILA llava (editable, --no-deps)"
pip install -e "$VILA_REPO" --no-deps

# ── 6. Apply VILA's transformers source patch (matches transformers 4.37.2) ───
echo "==> Patching transformers with VILA's transformers_replace"
SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"
TF_REPLACE="$VILA_REPO/llava/train/transformers_replace"
if [ -d "$TF_REPLACE" ] && [ -d "$SITE_PKGS/transformers" ]; then
    cp -rv "$TF_REPLACE"/* "$SITE_PKGS/transformers/"
else
    echo "WARNING: could not apply transformers_replace patch."
    echo "Model still runs unpatched (no sequence-packing optimisation)."
fi

# ── 7. Data directories (keep everything on the 241GB data disk) ──────────────
# EDIT DATA_ROOT to point at the partition with free space on your server.
DATA_ROOT="${ICSDG_DATA_ROOT:-$HOME/icsdg_data}"
echo "==> Creating data directories under $DATA_ROOT"
mkdir -p "$DATA_ROOT/ct_rate" "$DATA_ROOT/lidc_idri" "$DATA_ROOT/processed"

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "Setup complete."
echo "  conda activate $CONDA_ENV"
echo ""
echo "Next:"
echo "  1. export HF_TOKEN=<your token>   (CT-RATE is gated)"
echo "  2. Set HF_HOME onto the data disk so the HF cache does not"
echo "     fill the home partition:"
echo "       export HF_HOME=$DATA_ROOT/hf_cache"
echo "  3. Edit configs/train_config.yaml data paths to use $DATA_ROOT"
echo "──────────────────────────────────────────────────────────────"
