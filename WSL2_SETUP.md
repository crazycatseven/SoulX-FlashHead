# FlashHead WSL2 Setup Guide

Complete installation guide for SoulX-FlashHead on WSL2 with RTX 4090.

## Prerequisites
- WSL2 with Ubuntu 22.04/24.04
- NVIDIA GPU with CUDA support (tested on RTX 4090 24GB)
- Miniconda/Anaconda installed
- ~20GB disk space (models ~15GB + deps ~5GB)

## Step-by-Step Installation

### 1. Clone repo
```bash
cd ~
git clone https://github.com/Soul-AILab/SoulX-FlashHead.git
cd SoulX-FlashHead
```

### 2. Create conda environment
```bash
conda create -n flashhead python=3.10 -y
conda activate flashhead
```

### 3. Install PyTorch (CUDA 12.8)
```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install requirements (skip nccl conflict)
```bash
# nvidia-nccl-cu12==2.27.3 conflicts with torch 2.7.1 (needs 2.26.2)
# Safe to skip for single-GPU inference
grep -v 'nvidia-nccl' requirements.txt | pip install -r /dev/stdin
```

### 5. Install ninja (build tool)
```bash
pip install ninja
```

### 6. Install flash_attn (requires compilation, ~10-15 min)
```bash
# Option A: Compile from source (slow but reliable)
pip install flash_attn==2.8.0.post2 --no-build-isolation

# Option B: Download prebuilt wheel (faster)
# Find matching wheel at: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.0.post2
# Match: torch2.7, cu128, cp310, linux_x86_64
# pip install flash_attn-2.8.0.post2-cp310-cp310-linux_x86_64.whl
```

### 7. Install sageattention (requires compilation, ~3-5 min)
```bash
# PyPI doesn't have 2.2.0 wheels, must build from source
cd /tmp
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
pip install . --no-build-isolation
cd ~/SoulX-FlashHead
```
> ⚠️ **Note:** sageattention is installed but **disabled** on WSL2 due to CUDA compatibility issues. See [Compatibility Notes](#wsl2-compatibility-workarounds) below.

### 8. Install ffmpeg
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### 9. Download models (~15GB total)
```bash
pip install "huggingface_hub[cli]"

# FlashHead model (14GB) — takes a while
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
  --local-dir ./models/SoulX-FlashHead-1_3B

# wav2vec2 audio encoder (1.1GB)
huggingface-cli download facebook/wav2vec2-base-960h \
  --local-dir ./models/wav2vec2-base-960h
```

### 10. Apply WSL2 compatibility patches
```bash
# Disable sageattention (CUDA kernel incompatible with WSL2)
sed -i 's/SAGE_ATTN_AVAILABLE = True/SAGE_ATTN_AVAILABLE = False/' \
  flash_head/src/modules/flash_head_model.py
```

### 11. Verify installation
```bash
conda activate flashhead
python -c "
import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
from flash_attn import flash_attn_func; print('flash_attn OK')
from sageattention import sageattn; print('sageattention OK (installed but disabled)')
import ffmpeg; print('ffmpeg OK')
" 2>&1
# flash_attn and sageattention should both print OK
# sageattention is installed but disabled in model code
```

### 12. Test inference (CLI)
```bash
# Quick test with Lite model
CUDA_VISIBLE_DEVICES=0 python generate_video.py \
  --ckpt_dir models/SoulX-FlashHead-1_3B \
  --wav2vec_dir models/wav2vec2-base-960h \
  --model_type lite \
  --cond_image examples/girl.png \
  --audio_path examples/podcast_sichuan_16k.wav \
  --audio_encode_mode stream
```

### 13. Launch WebUI (optional)
```bash
CUDA_VISIBLE_DEVICES=0 python webui.py
# Open http://localhost:7860
```

---

## WSL2 Compatibility Workarounds

### ⚠️ SageAttention Disabled
**File:** `flash_head/src/modules/flash_head_model.py` → `SAGE_ATTN_AVAILABLE = False`

**Reason:** SageAttention's custom CUDA kernels are incompatible with WSL2's CUDA virtualization layer (`/usr/lib/wsl/lib/`), causing `RuntimeError: CUDA error: unspecified launch failure` during inference. This happens regardless of torch.compile state.

**Fallback:** `flash_attn` v2.8.0.post2 — identical output quality, ~5-10% slower at most.

### ⚠️ TorchDynamo Disabled
**File:** `webui.py` → `os.environ["TORCHDYNAMO_DISABLE"] = "1"`

**Reason:** `torch.compile` interacts poorly with custom CUDA kernels under WSL2, causing sporadic CUDA context corruption.

### nvidia-nccl-cu12 Skipped
**Reason:** `requirements.txt` specifies `==2.27.3` but torch 2.7.1 needs `==2.26.2`. NCCL is only for multi-GPU; single-GPU is unaffected.

---

## Performance (Single RTX 4090, flash_attn)
| Model | Speed | Notes |
|-------|-------|-------|
| Lite  | ~55ms/step | Real-time capable, 3 concurrent streams |
| Pro   | ~10.2 FPS | Higher quality, single stream |

Consistent with official benchmarks (Lite: 96 FPS, Pro: 10.8 FPS on native Linux).

---

## Restoring on Native Linux
If running on bare-metal Linux (not WSL2):
1. Change `SAGE_ATTN_AVAILABLE = False` → `True` in `flash_head/src/modules/flash_head_model.py`
2. Remove `TORCHDYNAMO_DISABLE=1` from `webui.py`
3. Test Lite model first

---

## Quick Reference
| Component | Version | Install Method |
|-----------|---------|----------------|
| Python | 3.10 | conda |
| PyTorch | 2.7.1+cu128 | pip (whl index) |
| flash_attn | 2.8.0.post2 | pip (compile ~15min) |
| sageattention | 2.2.0 | source (compile ~5min, disabled on WSL2) |
| xformers | 0.0.31 | pip (via requirements.txt) |
| diffusers | ≥0.34.0 | pip (via requirements.txt) |
| transformers | 4.57.3 | pip (via requirements.txt) |
| ffmpeg | system | apt-get |
| FlashHead model | 1.3B (14GB) | huggingface-cli |
| wav2vec2 | base-960h (1.1GB) | huggingface-cli |
