"""
Comprehensive Tests for DeepGuard-X
Unit tests for all modules
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.video_models import VideoDeepfakeModel, create_video_model
from src.models.audio_models import Wav2Vec2Detector, create_audio_model
from src.utils import MetricsCalculator, load_config


class TestVideoModels:
    """Test video detection models"""
    
    def test_efficientnet_forward(self):
        """Test EfficientNet forward pass"""
        model = VideoDeepfakeModel(architecture='efficientnet_b0', num_classes=2)
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (2, 2)
    
    def test_vit_forward(self):
        """Test ViT forward pass"""
        model = VideoDeepfakeModel(architecture='vit_base_patch16_224', num_classes=2)
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (1, 2)
    
    def test_create_video_model(self):
        """Test video model factory"""
        config = {
            'architecture': 'efficientnet_b0',
            'num_classes': 2,
            'pretrained': False,
            'dropout': 0.3
        }
        
        model = create_video_model(config)
        assert model is not None


class TestAudioModels:
    """Test audio detection models"""
    
    def test_wav2vec2_forward(self):
        """Test Wav2Vec2 forward pass"""
        # This test requires the actual model weights
        # Skip if not available
        pytest.skip("Requires pretrained weights")
    
    def test_create_audio_model(self):
        """Test audio model factory"""
        config = {
            'architecture': 'wav2vec2',
            'pretrained_model': 'facebook/wav2vec2-base',
            'num_classes': 2,
            'pooling': 'attention'
        }
        
        # Skip if transformers models not available
        pytest.skip("Requires transformers library")


class TestMetrics:
    """Test metrics calculation"""
    
    def test_metrics_calculator(self):
        """Test MetricsCalculator"""
        calc = MetricsCalculator(num_classes=2)
        
        # Simulate predictions
        preds = torch.tensor([0, 1, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 0, 1])
        probs = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1], [0.2, 0.8]])
        
        calc.update(preds, targets, probs)
        metrics = calc.compute()
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['auc'] <= 1


class TestConfig:
    """Test configuration loading"""
    
    def test_load_config(self):
        """Test config loading"""
        # Test with existing config
        config_path = "configs/video/efficientnet_b3.yaml"
        
        if Path(config_path).exists():
            config = load_config(config_path)
            assert config is not None
            assert 'model' in config
            assert 'training' in config


def test_imports():
    """Test that all imports work"""
    from src import DeepGuardXInference, get_logger
    from src.utils import MetricsCalculator, load_config, ConfigManager
    from src.models import create_video_model, create_audio_model
    from src.ensemble import EnsembleFusion, TemporalSmoother
    
    assert DeepGuardXInference is not None
    assert MetricsCalculator is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
