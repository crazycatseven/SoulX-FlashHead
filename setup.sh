#!/bin/bash
# ============================================================
# FlashHead One-Click Setup for WSL2
# Tested on: Ubuntu 22.04/24.04, RTX 4090 24GB
# Usage: bash setup.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="flashhead"
PYTHON_VER="3.10"
TORCH_VER="2.7.1"
CUDA_TAG="cu128"
MODEL_DIR="$SCRIPT_DIR/models"

echo "============================================"
echo "  FlashHead WSL2 Setup"
echo "============================================"

# ── 0. Check prerequisites ──
echo ""
echo "[0/8] Checking prerequisites..."

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA GPU drivers for WSL2 first."
    exit 1
fi
echo "  ✓ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi
echo "  ✓ conda found"

# ── 1. Create conda environment ──
echo ""
echo "[1/8] Setting up conda environment ($CONDA_ENV, Python $PYTHON_VER)..."

if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "  Environment '$CONDA_ENV' already exists, activating..."
else
    conda create -n "$CONDA_ENV" python="$PYTHON_VER" -y
fi

# Activate conda in script context
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
echo "  ✓ Python $(python --version 2>&1 | awk '{print $2}')"

# ── 2. Install PyTorch ──
echo ""
echo "[2/8] Installing PyTorch ${TORCH_VER}+${CUDA_TAG}..."

if python -c "import torch; assert torch.__version__.startswith('${TORCH_VER}')" 2>/dev/null; then
    echo "  Already installed, skipping."
else
    pip install torch==${TORCH_VER} torchvision==0.22.1 --index-url https://download.pytorch.org/whl/${CUDA_TAG}
fi
echo "  ✓ $(python -c 'import torch; print(f"torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")')"

# ── 3. Install Python dependencies ──
echo ""
echo "[3/8] Installing Python dependencies..."

# nvidia-nccl-cu12 conflicts with torch (needs different version), skip it (only for multi-GPU)
grep -v 'nvidia-nccl' requirements.txt | pip install -r /dev/stdin

pip install ninja

echo "  ✓ Dependencies installed"

# ── 4. Install flash_attn ──
echo ""
echo "[4/8] Installing flash_attn (this may take 10-15 min to compile)..."

if python -c "from flash_attn import flash_attn_func; print('OK')" 2>/dev/null; then
    echo "  Already installed, skipping."
else
    pip install flash_attn==2.8.0.post2 --no-build-isolation
fi
echo "  ✓ flash_attn ready"

# ── 5. Install sageattention ──
echo ""
echo "[5/8] Installing sageattention (compile from source, ~3-5 min)..."

if python -c "from sageattention import sageattn" 2>/dev/null; then
    echo "  Already installed, skipping."
else
    SAGE_TMP="/tmp/SageAttention_build"
    rm -rf "$SAGE_TMP"
    git clone https://github.com/thu-ml/SageAttention.git "$SAGE_TMP"
    cd "$SAGE_TMP"
    pip install . --no-build-isolation
    cd "$SCRIPT_DIR"
    rm -rf "$SAGE_TMP"
fi
echo "  ✓ sageattention ready (disabled in WSL2 mode, see below)"

# ── 6. Install ffmpeg ──
echo ""
echo "[6/8] Installing ffmpeg..."

if command -v ffmpeg &>/dev/null; then
    echo "  Already installed, skipping."
else
    sudo apt-get update && sudo apt-get install -y ffmpeg
fi
echo "  ✓ ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"

# ── 7. Download models ──
echo ""
echo "[7/8] Downloading models (~15GB total, may take a while)..."

pip install "huggingface_hub[cli]" -q

if [ -d "$MODEL_DIR/SoulX-FlashHead-1_3B/Model_Lite" ]; then
    echo "  FlashHead model already exists, skipping."
else
    echo "  Downloading FlashHead model (14GB)..."
    huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
        --local-dir "$MODEL_DIR/SoulX-FlashHead-1_3B"
fi

if [ -d "$MODEL_DIR/wav2vec2-base-960h" ]; then
    echo "  wav2vec2 model already exists, skipping."
else
    echo "  Downloading wav2vec2 (1.1GB)..."
    huggingface-cli download facebook/wav2vec2-base-960h \
        --local-dir "$MODEL_DIR/wav2vec2-base-960h"
fi

echo "  ✓ Models ready"

# ── 8. Apply WSL2 patches ──
echo ""
echo "[8/8] Applying WSL2 compatibility patches..."

MODEL_PY="$SCRIPT_DIR/flash_head/src/modules/flash_head_model.py"
if grep -q 'SAGE_ATTN_AVAILABLE = True' "$MODEL_PY" 2>/dev/null; then
    sed -i 's/SAGE_ATTN_AVAILABLE = True/SAGE_ATTN_AVAILABLE = False  # disabled on WSL2/' "$MODEL_PY"
    echo "  ✓ Disabled sageattention (WSL2 CUDA incompatible)"
else
    echo "  Already patched."
fi

# ── Done ──
echo ""
echo "============================================"
echo "  ✅ FlashHead setup complete!"
echo "============================================"
echo ""
echo "Quick start:"
echo "  conda activate $CONDA_ENV"
echo ""
echo "  # API server (for ai-mock-interview)"
echo "  python api_server.py"
echo ""
echo "  # WebUI (standalone)"
echo "  python webui.py"
echo ""
echo "  # CLI test"
echo "  python generate_video.py \\"
echo "    --ckpt_dir models/SoulX-FlashHead-1_3B \\"
echo "    --wav2vec_dir models/wav2vec2-base-960h \\"
echo "    --model_type lite \\"
echo "    --cond_image examples/girl.png \\"
echo "    --audio_path examples/podcast_sichuan_16k.wav \\"
echo "    --audio_encode_mode stream"
echo ""
