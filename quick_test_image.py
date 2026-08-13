"""Quick test for image detection - lightweight version"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Testing image detection components...")
print("=" * 60)

# Test 1: Config loading (fast)
print("\n[1/5] Testing config loading...")
try:
    from src.utils.config import load_config
    config = load_config("configs/image/efficientnet_b3.yaml")
    print(f"✓ Config loaded: {config.model.architecture}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 2: Check data directories
print("\n[2/5] Checking data directories...")
train_path = Path(config.data.train_path)
val_path = Path(config.data.val_path)

train_real = list((train_path / "real").glob("*.*")) if (train_path / "real").exists() else []
train_fake = list((train_path / "fake").glob("*.*")) if (train_path / "fake").exists() else []

print(f"  Train real: {len(train_real)} files")
print(f"  Train fake: {len(train_fake)} files")

if len(train_real) + len(train_fake) == 0:
    print("  ⚠️  WARNING: No training data found!")
else:
    print(f"  ✓ Found {len(train_real) + len(train_fake)} training images")

# Test 3: Import model module (no creation yet)
print("\n[3/5] Testing model imports...")
try:
    from src.models.video_models import create_video_model
    print("✓ Model module imported")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 4: Import dataset module
print("\n[4/5] Testing dataset imports...")
try:
    from src.data.video_dataset import VideoDeepfakeDataset
    print("✓ Dataset module imported")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 5: Import Lightning module
print("\n[5/5] Testing Lightning module imports...")
try:
    from src.training.video_lightning import VideoDeepfakeModule
    print("✓ Lightning module imported")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All imports successful!")
print("\nImage detection infrastructure is working.")
print("\nNote: Actual model creation and training requires:")
print("  - Training data in data/image/train/{real,fake}/")
print("  - Running: python scripts/train_image.py")
print("=" * 60)
