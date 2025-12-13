# Getting Started with DeepGuard-X

Welcome to **DeepGuard-X**, a production-grade multi-modal deepfake detection system! This guide will help you get up and running quickly.

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed
- **CUDA 11.8+** (optional, for GPU acceleration)
- **8GB+ RAM** (16GB+ recommended)
- **Git** for cloning the repository

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourorg/deepguard-x.git
cd deepguard-x
```

### Step 2: Create Virtual Environment

**On Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Optional: Install with GPU support
pip install -e ".[gpu]"

# Optional: Install development tools
pip install -e ".[dev]"
```

## 🎯 Quick Start Examples

### Example 1: Simple Video Detection

```python
from src import DeepGuardXInference

# Initialize detector
detector = DeepGuardXInference(
    config_path="configs/ensemble_config.yaml",
    use_onnx=True,
    device="cuda"  # or "cpu"
)

# Detect deepfake
results = detector.predict(video_path="path/to/video.mp4")

# Print results
print(f"Prediction: {results['final_label']}")
print(f"Deepfake Score: {results['final_score']:.2%}")
print(f"Confidence: {results['confidence']:.2%}")

# Individual model scores
for model, score in results['individual_scores'].items():
    print(f"  {model}: {score:.4f}")
```

### Example 2: Launch Web Dashboard

```bash
# Start the Streamlit dashboard
streamlit run src/dashboard/app.py
```

Then open your browser to `http://localhost:8501`

### Example 3: Real-Time Webcam Detection

```bash
# Run real-time detection
python scripts/realtime_demo.py --config configs/realtime_config.yaml
```

Press 'q' or ESC to quit.

## 🎓 Training Your First Model

### Prepare Your Data

Organize your dataset in the following structure:

```
data/faceforensics/
├── train/
│   ├── real/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── fake/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### Configure Training

Edit `configs/video/efficientnet_b3.yaml`:

```yaml
data:
  train_path: data/faceforensics/train
  val_path: data/faceforensics/val
  test_path: data/faceforensics/test
  image_size: 224

training:
  batch_size: 32  # Adjust based on GPU memory
  learning_rate: 0.0001
  max_epochs: 50
  precision: 16  # Mixed precision
```

### Start Training

```bash
# Train video detection model
python scripts/train_video.py --config configs/video/efficientnet_b3.yaml

# Monitor with TensorBoard
tensorboard --logdir logs/

# Or use WandB
# Set logger: wandb in config, then:
wandb login
```

### Training Tips

**If you get CUDA out of memory:**
```yaml
training:
  batch_size: 16  # Reduce batch size
  accumulate_grad_batches: 2  # Accumulate gradients
```

**Resume training from checkpoint:**
```bash
python scripts/train_video.py \
  --config configs/video/efficientnet_b3.yaml \
  --resume checkpoints/video/last.ckpt
```

## 🔧 Common Use Cases

### Use Case 1: Analyze a Video File

```python
from src import DeepGuardXInference

detector = DeepGuardXInference()
results = detector.predict(
    video_path="suspicious_video.mp4",
    return_details=True
)

# Check if deepfake
if results['final_label'] == 'FAKE':
    print(f"⚠️  Deepfake detected with {results['confidence']:.1%} confidence")
    print(f"\nDetection details:")
    print(f"  Video model: {results['individual_scores']['video']:.2%}")
    print(f"  Audio model: {results['individual_scores']['audio']:.2%}")
else:
    print(f"✅ Video appears authentic")
```

### Use Case 2: Batch Process Multiple Files

```python
import glob
from src import DeepGuardXInference

detector = DeepGuardXInference()

# Get all videos
video_files = glob.glob("videos/*.mp4")

results_summary = []

for video_path in video_files:
    print(f"Processing: {video_path}")
    
    try:
        results = detector.predict(video_path=video_path)
        results_summary.append({
            'file': video_path,
            'prediction': results['final_label'],
            'score': results['final_score']
        })
    except Exception as e:
        print(f"  Error: {e}")

# Summary
fake_count = sum(1 for r in results_summary if r['prediction'] == 'FAKE')
print(f"\n📊 Summary: {fake_count}/{len(results_summary)} videos flagged as deepfakes")
```

### Use Case 3: Export Model to ONNX

```python
import torch
from src.models import create_video_model

# Load trained model
model = create_video_model({
    'architecture': 'efficientnet_b3',
    'num_classes': 2,
    'pretrained': False
})

# Load checkpoint
checkpoint = torch.load('checkpoints/video/best_model.ckpt')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    'models/video_efficientnet_b3.onnx',
    opset_version=14,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("✅ Model exported to ONNX!")
```

## 📊 Understanding Results

### Result Dictionary Structure

```python
{
    'final_score': 0.73,           # Overall deepfake probability (0-1)
    'final_label': 'FAKE',         # 'FAKE' or 'REAL'
    'confidence': 0.73,            # Confidence in prediction
    'threshold': 0.5,              # Decision threshold
    'individual_scores': {
        'video': 0.82,             # Video model score
        'audio': 0.68,             # Audio model score
        'lipsync': 0.71,           # Lip-sync score
        'micro_expression': 0.65,  # Micro-expression score
        'behavior': 0.79           # Behavior consistency score
    },
    'details': {
        'num_models_used': 5,
        'score_variance': 0.0034,
        'score_std': 0.058
    }
}
```

### Interpreting Scores

- **0.0 - 0.3**: Very likely authentic
- **0.3 - 0.5**: Probably authentic
- **0.5 - 0.7**: Suspicious, possible deepfake
- **0.7 - 0.9**: Likely deepfake
- **0.9 - 1.0**: Very likely deepfake

## 🐛 Troubleshooting

### Issue: Import Errors

```bash
# Make sure you're in the project root directory
cd deepguard-x

# Install in editable mode
pip install -e .
```

### Issue: CUDA Out of Memory

```python
# Reduce batch size in config
training:
  batch_size: 8  # Instead of 32
```

Or use CPU:
```python
detector = DeepGuardXInference(device="cpu")
```

### Issue: Slow Inference

1. **Use ONNX models** (3-5x faster):
```python
detector = DeepGuardXInference(use_onnx=True)
```

2. **Enable GPU acceleration**:
```bash
pip install onnxruntime-gpu
```

3. **Reduce frame processing**:
```yaml
# In realtime_config.yaml
processing:
  frame_skip: 3  # Process every 3rd frame
```

### Issue: Missing Models

If you see "Model not found" errors:

1. Train models first:
```bash
python scripts/train_video.py --config configs/video/efficientnet_b3.yaml
```

2. Or download pretrained models (if available):
```bash
# Download from your model repository
wget https://your-models.com/video_model.ckpt -O checkpoints/video/best_model.ckpt
```

## 📚 Next Steps

1. **Read the full documentation**: Check `README.md` for detailed information
2. **Explore examples**: See `examples/quickstart.py` for more use cases
3. **Train custom models**: Follow `DEVELOPMENT.md` for training guides
4. **Customize configurations**: Edit YAML files in `configs/`
5. **Run tests**: `pytest tests/ -v` to verify installation

## 🔗 Resources

- **GitHub**: https://github.com/yourorg/deepguard-x
- **Documentation**: See `README.md` and `DEVELOPMENT.md`
- **Issues**: Report bugs on GitHub Issues
- **Support**: support@deepguard-x.ai

## 💡 Pro Tips

1. **Start with pretrained models** before training from scratch
2. **Use mixed precision training** (precision: 16) for faster training
3. **Monitor training** with WandB or TensorBoard
4. **Use ONNX models** for production inference
5. **Batch process** when analyzing multiple files
6. **Adjust thresholds** based on your use case (lower for more sensitivity)

## 🎉 You're Ready!

You now have everything you need to start detecting deepfakes with DeepGuard-X. If you have questions:

1. Check the documentation
2. Look at the examples
3. Open an issue on GitHub
4. Contact support

Happy detecting! 🛡️
