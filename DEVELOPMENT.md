# DeepGuard-X Development Guide

## Development Setup

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/yourorg/deepguard-x.git
cd deepguard-x

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

## Project Structure

```
deepguard-x/
├── configs/              # YAML configurations
│   ├── video/           # Video model configs
│   ├── audio/           # Audio model configs
│   ├── lipsync/         # Lip-sync configs
│   ├── micro_expression/
│   ├── behavior/
│   ├── ensemble_config.yaml
│   └── realtime_config.yaml
├── src/
│   ├── data/            # Dataset loaders
│   ├── models/          # Model architectures
│   ├── training/        # Training modules
│   ├── inference/       # Inference pipelines
│   ├── ensemble/        # Ensemble fusion
│   ├── realtime/        # Real-time detection
│   ├── dashboard/       # Streamlit UI
│   └── utils/           # Utilities
├── scripts/             # Training scripts
├── tests/               # Unit tests
├── examples/            # Usage examples
├── notebooks/           # Jupyter notebooks
├── checkpoints/         # Model weights
├── logs/                # Training logs
└── data/                # Datasets
```

## Development Workflow

### 1. Training New Models

```bash
# Train video model
python scripts/train_video.py --config configs/video/efficientnet_b3.yaml

# Train audio model
python scripts/train_audio.py --config configs/audio/wav2vec2.yaml

# Resume from checkpoint
python scripts/train_video.py --config configs/video/vit_base.yaml --resume checkpoints/video/last.ckpt
```

### 2. Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest --cov=src tests/
```

### 3. Code Quality

```bash
# Format code with black
black src/ scripts/ tests/

# Check with flake8
flake8 src/ scripts/ tests/

# Type checking with mypy
mypy src/

# Run all checks
pre-commit run --all-files
```

### 4. Experiment Tracking

#### Using WandB

```python
# In config YAML
logging:
  logger: wandb
  project_name: deepguard-x
  experiment_name: my_experiment

# Login
wandb login
```

#### Using MLflow

```python
logging:
  logger: mlflow
  experiment_name: my_experiment

# View UI
mlflow ui
```

## Adding New Features

### Adding a New Model Architecture

1. **Create model class** in `src/models/`:

```python
# src/models/my_model.py
import torch.nn as nn

class MyNewModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize layers
    
    def forward(self, x):
        # Forward pass
        return output
```

2. **Add factory function**:

```python
def create_my_model(config):
    return MyNewModel(config)
```

3. **Create configuration** in `configs/`:

```yaml
# configs/my_model/config.yaml
model:
  architecture: my_model
  # ... parameters
```

4. **Add training script** or modify existing one

5. **Write tests**:

```python
# tests/test_my_model.py
def test_my_model_forward():
    model = MyNewModel(config)
    # Test forward pass
```

### Adding a New Dataset

1. **Create dataset class** in `src/data/`:

```python
# src/data/my_dataset.py
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_path, config):
        # Initialize
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load and return sample
        return data, label
```

2. **Create dataloader function**

3. **Add tests**

## Performance Optimization

### Mixed Precision Training

```python
# In config
training:
  precision: 16  # Use FP16
```

### Gradient Accumulation

```python
training:
  accumulate_grad_batches: 4  # Accumulate over 4 batches
```

### Multi-GPU Training

```bash
# Automatic with PyTorch Lightning
python scripts/train_video.py --config configs/video/efficientnet_b3.yaml
# Will use all available GPUs
```

### ONNX Export for Inference

```python
from src.inference.onnx_converter import export_to_onnx

export_to_onnx(
    model=model,
    onnx_path="models/my_model.onnx",
    input_shape=(1, 3, 224, 224)
)
```

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### PyTorch Lightning Debugging

```python
# Fast dev run (1 batch)
trainer = pl.Trainer(fast_dev_run=True)

# Overfit on small dataset
trainer = pl.Trainer(overfit_batches=10)

# Limit training batches
trainer = pl.Trainer(limit_train_batches=100)
```

### Memory Issues

```python
# Reduce batch size
training:
  batch_size: 8  # Instead of 32

# Enable gradient checkpointing (if supported)
model:
  gradient_checkpointing: true
```

## Contributing

### Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and commit: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/my-feature`
5. Create Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to all functions/classes
- Write unit tests for new features
- Update documentation
- Keep functions focused and modular
- Use type hints where appropriate

### Commit Messages

Follow conventional commits:

```
feat: Add new audio model architecture
fix: Correct bug in data preprocessing
docs: Update README with new examples
test: Add tests for ensemble fusion
refactor: Simplify config loading logic
```

## Resources

- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [WandB Documentation](https://docs.wandb.ai/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Getting Help

- Open an issue on GitHub
- Check existing issues and discussions
- Contact: support@deepguard-x.ai
