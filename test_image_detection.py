"""
Test script to verify image detection functionality
"""

import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, get_logger
from src.models.video_models import create_video_model
from src.data.video_dataset import VideoDeepfakeDataset, create_video_dataloaders
from src.training.video_lightning import VideoDeepfakeModule

logger = get_logger(__name__)


def test_config_loading():
    """Test 1: Can we load the image config?"""
    try:
        config = load_config("configs/image/efficientnet_b3.yaml")
        logger.info("✓ Config loaded successfully")
        logger.info(f"  Architecture: {config.model.architecture}")
        logger.info(f"  Batch size: {config.training.batch_size}")
        logger.info(f"  Image size: {config.data.image_size}")
        return True, config
    except Exception as e:
        logger.error(f"✗ Config loading failed: {e}")
        return False, None


def test_model_creation(config):
    """Test 2: Can we create the model?"""
    try:
        model = create_video_model(
            architecture=config.model.architecture,
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained,
            dropout=config.model.dropout
        )
        logger.info("✓ Model created successfully")
        logger.info(f"  Model type: {type(model).__name__}")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 3, config.data.image_size, config.data.image_size)
        with torch.no_grad():
            output = model(dummy_input)
        logger.info(f"  Output shape: {output.shape}")
        logger.info("✓ Forward pass successful")
        return True, model
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        return False, None


def test_dataset_loading(config):
    """Test 3: Can we load the dataset?"""
    try:
        # Check if data exists
        train_path = Path(config.data.train_path)
        val_path = Path(config.data.val_path)
        
        logger.info(f"Checking paths:")
        logger.info(f"  Train path: {train_path} (exists: {train_path.exists()})")
        logger.info(f"  Val path: {val_path} (exists: {val_path.exists()})")
        
        # Count files in each directory
        train_real = list((train_path / "real").glob("*.*")) if (train_path / "real").exists() else []
        train_fake = list((train_path / "fake").glob("*.*")) if (train_path / "fake").exists() else []
        val_real = list((val_path / "real").glob("*.*")) if (val_path / "real").exists() else []
        val_fake = list((val_path / "fake").glob("*.*")) if (val_path / "fake").exists() else []
        
        logger.info(f"  Train real images: {len(train_real)}")
        logger.info(f"  Train fake images: {len(train_fake)}")
        logger.info(f"  Val real images: {len(val_real)}")
        logger.info(f"  Val fake images: {len(val_fake)}")
        
        if len(train_real) + len(train_fake) == 0:
            logger.warning("✗ No training data found!")
            return False, None
        
        # Try to create dataset
        dataset = VideoDeepfakeDataset(
            data_path=str(train_path),
            image_size=config.data.image_size,
            augmentation=config.data.augmentation,
            is_training=True
        )
        
        logger.info(f"✓ Dataset created with {len(dataset)} samples")
        
        if len(dataset) > 0:
            # Test loading a sample
            sample, label = dataset[0]
            logger.info(f"  Sample shape: {sample.shape}")
            logger.info(f"  Label: {label}")
            logger.info("✓ Sample loading successful")
        
        return True, dataset
    except Exception as e:
        logger.error(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_lightning_module(config):
    """Test 4: Can we create a Lightning module?"""
    try:
        module = VideoDeepfakeModule(config)
        logger.info("✓ Lightning module created successfully")
        
        # Test forward pass
        dummy_batch = torch.randn(2, 3, config.data.image_size, config.data.image_size)
        dummy_labels = torch.tensor([0, 1])
        
        with torch.no_grad():
            loss = module.training_step((dummy_batch, dummy_labels), 0)
        
        logger.info(f"  Training step output: {loss}")
        logger.info("✓ Training step successful")
        return True, module
    except Exception as e:
        logger.error(f"✗ Lightning module creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_inference_on_image(model, config):
    """Test 5: Can we run inference on a sample image?"""
    try:
        # Create a dummy image (simulating real inference)
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Convert to tensor and normalize
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((config.data.image_size, config.data.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(dummy_image).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1)
        
        logger.info("✓ Inference successful")
        logger.info(f"  Logits: {output[0].tolist()}")
        logger.info(f"  Probabilities: {probs[0].tolist()}")
        logger.info(f"  Prediction: {'FAKE' if pred_class[0] == 1 else 'REAL'}")
        return True
    except Exception as e:
        logger.error(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("DeepGuard-X Image Detection Test Suite")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: Config loading
    logger.info("\n[Test 1] Testing config loading...")
    success, config = test_config_loading()
    results['config_loading'] = success
    if not success:
        logger.error("Cannot proceed without valid config")
        return
    
    # Test 2: Model creation
    logger.info("\n[Test 2] Testing model creation...")
    success, model = test_model_creation(config)
    results['model_creation'] = success
    
    # Test 3: Dataset loading
    logger.info("\n[Test 3] Testing dataset loading...")
    success, dataset = test_dataset_loading(config)
    results['dataset_loading'] = success
    
    # Test 4: Lightning module
    logger.info("\n[Test 4] Testing Lightning module...")
    success, module = test_lightning_module(config)
    results['lightning_module'] = success
    
    # Test 5: Inference
    if model is not None:
        logger.info("\n[Test 5] Testing inference...")
        success = test_inference_on_image(model, config)
        results['inference'] = success
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    logger.info("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    logger.info(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("\n🎉 All tests passed! Image detection is working correctly.")
    else:
        logger.warning("\n⚠️  Some tests failed. Check the errors above.")


if __name__ == '__main__':
    main()
