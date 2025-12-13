# GPU Training Setup Guide

## 🎮 Check if You Have GPU

```powershell
# Check NVIDIA GPU
nvidia-smi

# If command not found, you don't have NVIDIA GPU or drivers installed
```

---

## ⚡ GPU Setup Steps

### Step 1: Check Current PyTorch Installation

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

**If CUDA Available = False**, you need GPU-enabled PyTorch.

---

### Step 2: Install GPU-Enabled PyTorch

**First, check your CUDA version:**
```powershell
nvidia-smi
# Look for "CUDA Version" in the top-right
```

**Then install PyTorch with CUDA support:**

**For CUDA 11.8:**
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only (no GPU):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

### Step 3: Verify GPU Installation

```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'); print('GPU Count:', torch.cuda.device_count())"
```

**Should show:**
```
CUDA Available: True
GPU Name: NVIDIA GeForce RTX 3060 (or your GPU)
GPU Count: 1
```

---

## 🚀 Training with GPU

### Automatic (Default)
The training script automatically uses GPU if available:

```powershell
python scripts/train_video.py --config configs/video/efficientnet_b3.yaml
```

### Manual GPU Selection (if you have multiple GPUs)

```powershell
# Use GPU 0
$env:CUDA_VISIBLE_DEVICES="0"
python scripts/train_video.py --config configs/video/efficientnet_b3.yaml

# Use GPU 1
$env:CUDA_VISIBLE_DEVICES="1"
python scripts/train_video.py --config configs/video/efficientnet_b3.yaml

# Use multiple GPUs (0 and 1)
$env:CUDA_VISIBLE_DEVICES="0,1"
python scripts/train_video.py --config configs/video/efficientnet_b3.yaml
```

---

## 📊 Monitor GPU Usage During Training

**Real-time GPU monitoring:**
```powershell
# In a separate terminal
nvidia-smi -l 1  # Update every 1 second
```

**What to look for:**
- **GPU Utilization**: Should be 80-100%
- **Memory Usage**: Should be using most of VRAM
- **Temperature**: Should be 60-80°C (normal)
- **Power Usage**: Should be near max TDP

---

## ⚡ Performance Comparison

| Hardware | Training Time (50 epochs) | Speed |
|----------|---------------------------|-------|
| CPU (Intel i7) | 8-12 hours | 1x |
| GPU (RTX 3060) | 2-3 hours | 4x |
| GPU (RTX 3090) | 1-1.5 hours | 8x |
| GPU (A100) | 30-45 minutes | 16x |

---

## 🔧 Troubleshooting

### "CUDA out of memory"

Reduce batch size in config:

```yaml
# configs/video/efficientnet_b3.yaml
training:
  batch_size: 16  # Reduce from 32 to 16 or 8
```

### "GPU not detected"

1. Install/update NVIDIA drivers
2. Reinstall PyTorch with correct CUDA version
3. Restart computer

### "Training is slow on GPU"

Check:
```powershell
nvidia-smi
```
- GPU utilization should be 80-100%
- If low, increase batch size
- Check if data loading is bottleneck (increase num_workers)

---

## 💡 Optimization Tips

### 1. Mixed Precision Training (Already Enabled)
```yaml
# Already in config
training:
  precision: 16  # FP16 for 2x speed boost
```

### 2. Increase Batch Size (if you have VRAM)
```yaml
training:
  batch_size: 64  # Try 64 or 128 if you have 16GB+ VRAM
```

### 3. Faster Data Loading
```yaml
data:
  num_workers: 8  # Increase for faster data loading
```

### 4. Gradient Accumulation (for larger effective batch size)
```yaml
training:
  accumulate_grad_batches: 2  # Effective batch_size = 32 * 2 = 64
```

---

## 🎯 Quick GPU Check

Run this to verify everything:

```powershell
python -c "
import torch
print('='*50)
print('GPU SETUP CHECK')
print('='*50)
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print('✅ GPU Ready for Training!')
else:
    print('❌ GPU Not Available - Will use CPU')
    print('Install GPU PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118')
print('='*50)
"
```

---

## 🚀 Start Training on GPU

```powershell
# Make sure GPU is available
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# Start training (automatically uses GPU)
python scripts/train_video.py --config configs/video/efficientnet_b3.yaml

# Monitor in another terminal
nvidia-smi -l 1
```

---

## ⏱️ Expected Speed

With GPU (RTX 3060 or better):
- **Data Loading**: 30 seconds
- **First Epoch**: 3-5 minutes
- **Subsequent Epochs**: 2-3 minutes
- **Total (50 epochs)**: 2-3 hours

Without GPU (CPU only):
- **First Epoch**: 10-15 minutes
- **Total (50 epochs)**: 8-12 hours

---

## 📝 Current Training Command

```powershell
# With GPU auto-detection
D:/deepfake_detect/venv/Scripts/python.exe scripts/train_video.py --config configs/video/efficientnet_b3.yaml
```

GPU will be used automatically if available! 🚀
