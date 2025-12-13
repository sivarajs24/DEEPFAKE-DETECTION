# DeepGuard-X: AI Agent Instructions

## Project Overview
**DeepGuard-X** is a production-grade, multi-modal deepfake detection system combining video, audio, lip-sync, micro-expression, and behavior consistency analysis. It uses ensemble fusion with PyTorch Lightning training and real-time ONNX inference.

## Architecture Overview

### Core Modules
- **Video Detection** (`src/models/video_models.py`): EfficientNet/ViT/Xception models via timm library
- **Audio Detection** (`src/models/audio_models.py`): wav2vec2.0, ECAPA-TDNN, RawNet2
- **Lip-Sync Detection** (`src/models/syncnet_model.py`): Twin ResNet18+Conv1D network for AV synchronization
- **Micro-Expression** (`src/models/micro_expression.py`): MediaPipe FaceMesh + Optical Flow + LSTM
- **Ensemble Fusion** (`src/ensemble/fusion.py`): Weighted average or meta-learner (LogisticRegression/XGBoost)
- **Inference Pipeline** (`src/inference/pipeline.py`): Unified `DeepGuardXInference` class coordinating all modules
- **Real-Time Processing** (`src/realtime/detector.py`): `RealtimeDetector` with async inference via ThreadPoolExecutor

### Data Flow
1. **Input** → Video/audio files or webcam stream
2. **Preprocessing** → Face extraction (MediaPipe), frame sampling, normalization
3. **Modular Detection** → Each module generates independent predictions
4. **Ensemble Fusion** → Combines predictions using configured fusion method
5. **Output** → Final score (0-1) with per-module confidence and explanations

## Configuration System (Critical Pattern)

**All models and training use YAML configs via OmegaConf** (`src/utils/config.py`):
```python
config = load_config("configs/video/efficientnet_b3.yaml")
# Access: config.model.architecture, config.training.batch_size, etc.
OmegaConf.to_yaml(config)  # For debugging
```

Key config locations:
- `configs/video/*.yaml` → Video model training
- `configs/audio/*.yaml` → Audio model training
- `configs/ensemble_config.yaml` → Ensemble fusion weights and meta-learner
- `configs/realtime_config.yaml` → Real-time detector settings

## Training Pattern (PyTorch Lightning)

All training uses `pytorch-lightning` with standard pattern:
```python
from src.training.video_lightning import VideoDeepfakeModule
from pytorch_lightning import Trainer

config = load_config("configs/video/efficientnet_b3.yaml")
module = VideoDeepfakeModule(config)
trainer = Trainer(
    max_epochs=config.training.epochs,
    mixed_precision=config.training.mixed_precision,
    callbacks=[...]  # EarlyStopping, ModelCheckpoint, etc.
)
trainer.fit(module, train_dataloaders, val_dataloaders)
```

Training scripts: `scripts/train_video.py`, `scripts/train_audio.py`

## Data Handling Pattern

Datasets follow directory structure:
```
data/{audio|video}/{train|val}/
├── real/  # label 0
└── fake/  # label 1
```

**Dataset Classes**: `src/data/video_dataset.py`, `src/data/audio_dataset.py`
- Constructor: `data_path`, `image_size`, `augmentation` config, `is_training` flag
- Returns: `(input_tensor, label)` tuples
- Augmentation: albumentations (video), audiomentations (audio)

## Inference Workflow

```python
# Single-line initialization
detector = DeepGuardXInference(
    config_path="configs/ensemble_config.yaml",
    use_onnx=True,  # Prefer ONNX for production
    device="cuda"   # Falls back to CPU if unavailable
)

# Prediction
results = detector.predict(video_path="test.mp4")
# Returns: {
#   'final_score': 0.85,
#   'confidence': 0.92,
#   'per_module': {'video': 0.9, 'audio': 0.8, ...},
#   'uncertainty': 0.05,
#   'explanation': {...}
# }
```

## Logging Convention

**All modules use `loguru` via `get_logger()` utility**:
```python
from src.utils import get_logger
logger = get_logger(__name__)
logger.info(f"Starting training on {device}")
logger.error(f"Model load failed: {e}")
```

Logs are hierarchical and support structured data. Check `src/utils/logger.py` for setup.

## Model Export & Optimization

**ONNX Export**: Required for production deployment
```python
from src.inference.pipeline import export_to_onnx
export_to_onnx(
    model=model,
    sample_input=torch.randn(1, 3, 224, 224),
    output_path="checkpoints/model.onnx"
)
```

Inference uses `onnxruntime` with automatic provider selection (CUDA → CPU fallback).

## Metrics & Evaluation

**Custom MetricsCalculator** (`src/utils/metrics.py`):
- Tracks: accuracy, precision, recall, AUC-ROC, F1
- Used in validation/test steps of Lightning modules
- Call `.compute()` for aggregated metrics, `.reset()` after logging

## Real-Time Processing Details

`RealtimeDetector` (`src/realtime/detector.py`):
- Async inference via ThreadPoolExecutor (decouples frame capture from inference)
- Frame buffering and skipping for performance
- ONNX-only (no PyTorch models in production)
- Returns per-frame predictions with visual annotations
- Configurable inference intervals per model (e.g., video every 5 frames, audio every 30)

## Development Workflows

### Training a Model
```bash
# Install from source
pip install -e .

# Run training with config override
python scripts/train_video.py --config configs/video/vit_base.yaml --epochs 50 --batch_size 32
```

### Testing Inference
```bash
python -c "
from src.inference.pipeline import DeepGuardXInference
detector = DeepGuardXInference('configs/ensemble_config.yaml')
result = detector.predict('path/to/video.mp4')
print(result['final_score'])
"
```

### Real-Time Demo
```bash
python scripts/realtime_demo.py --config configs/realtime_config.yaml --device cuda
```

### Dashboard
```bash
streamlit run src/dashboard/app.py
```

## Code Quality Standards

- **Type Hints**: Required for all functions (e.g., `def forward(self, x: torch.Tensor) -> torch.Tensor`)
- **Docstrings**: Google-style docstrings for all classes/public functions
- **Error Handling**: Try-except with proper logging, never silent failures
- **Device Handling**: Always check `torch.cuda.is_available()` before CUDA operations
- **Path Handling**: Use `pathlib.Path` consistently, never string concatenation

## Key Files for Reference

| File | Purpose |
|------|---------|
| [src/models/video_models.py](src/models/video_models.py) | Architecture patterns (timm-based model creation) |
| [src/training/video_lightning.py](src/training/video_lightning.py) | Lightning module template with mixed precision |
| [src/data/video_dataset.py](src/data/video_dataset.py) | Dataset class pattern with augmentations |
| [src/ensemble/fusion.py](src/ensemble/fusion.py) | Ensemble fusion logic and meta-learner training |
| [src/inference/pipeline.py](src/inference/pipeline.py) | End-to-end inference orchestration |
| [src/utils/config.py](src/utils/config.py) | OmegaConf configuration loading/validation |
| [configs/ensemble_config.yaml](configs/ensemble_config.yaml) | Example: Multi-model fusion configuration |

## Common Pitfalls to Avoid

1. **Device Mismatch**: Always move model/data to same device before forward pass
2. **Metric Reset**: Remember to reset metrics after logging (critical in Lightning)
3. **Config Immutability**: OmegaConf freezes configs—use `OmegaConf.to_container()` to convert if modifying
4. **ONNX vs PyTorch**: ONNX models need `.onnx_path` config; PyTorch needs `.checkpoint`
5. **Real-Time FPS**: Don't assume 30 FPS—frame skipping configs are dataset-dependent
