"""
Lightweight Dataset Generator for DeepGuard-X
Creates a small image-based dataset (< 10 GB) for quick training and testing
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter
import random


def create_real_image(image_id: int) -> np.ndarray:
    """
    Create a synthetic 'real' face-like image
    Returns RGB image as numpy array
    """
    width, height = 224, 224
    
    # Create a base image with skin tone gradient
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add skin tone background (realistic face color)
    skin_color = (200 + random.randint(-20, 20), 
                  150 + random.randint(-20, 20), 
                  130 + random.randint(-20, 20))
    image[:, :] = skin_color
    
    # Add natural variations
    for i in range(height):
        for j in range(width):
            noise = random.randint(-10, 10)
            image[i, j] = np.clip(image[i, j] + noise, 0, 255)
    
    # Add some facial features (circles for eyes, etc.)
    center_x, center_y = width // 2, height // 2
    
    # Left eye
    cv2.circle(image, (center_x - 40, center_y - 30), 15, (50, 50, 50), -1)
    cv2.circle(image, (center_x - 40, center_y - 30), 8, (100, 100, 200), -1)
    
    # Right eye
    cv2.circle(image, (center_x + 40, center_y - 30), 15, (50, 50, 50), -1)
    cv2.circle(image, (center_x + 40, center_y - 30), 8, (100, 100, 200), -1)
    
    # Mouth
    cv2.ellipse(image, (center_x, center_y + 50), (30, 15), 0, 0, 180, (150, 50, 50), 2)
    
    # Add realistic texture with slight blur
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def create_fake_image(image_id: int) -> np.ndarray:
    """
    Create a synthetic 'fake' (deepfake-like) image
    Introduces artifacts and inconsistencies
    """
    # Start with a real image
    image = create_real_image(image_id)
    
    # Add deepfake artifacts
    
    # 1. Unnatural smoothing/blurring in patches
    h, w = image.shape[:2]
    patch_size = 50
    for _ in range(random.randint(2, 5)):
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        patch = image[y:y+patch_size, x:x+patch_size]
        blurred_patch = cv2.GaussianBlur(patch, (15, 15), 0)
        image[y:y+patch_size, x:x+patch_size] = blurred_patch
    
    # 2. Compression artifacts (JPEG)
    for _ in range(random.randint(1, 3)):
        quality = random.randint(50, 75)
        ret, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. Inconsistent lighting/color shifts
    if random.random() > 0.5:
        # Brighten one side
        mask = np.ones_like(image, dtype=np.float32)
        mask[:, :w//2] = 1.2
        image = np.clip(image.astype(np.float32) * mask, 0, 255).astype(np.uint8)
    
    # 4. Add subtle noise that looks unnatural
    noise = np.random.normal(0, 5, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return image


def main():
    parser = argparse.ArgumentParser(
        description="Create lightweight dataset (< 10 GB) for DeepGuard-X",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/create_small_dataset.py                    # 100 samples (default)
  python scripts/create_small_dataset.py --num-samples 500  # 500 samples
  python scripts/create_small_dataset.py --num-samples 1000 # 1000 samples (full)
  
Note: Each sample is ~50-100 KB, so:
  100 samples = ~10-20 MB
  500 samples = ~50-100 MB
  1000 samples = ~100-200 MB
        """
    )
    
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of images per class (default: 100)")
    parser.add_argument("--dataset-type", choices=['image', 'mixed'], default='image',
                        help="Dataset type: image only or mixed with videos")
    
    args = parser.parse_args()
    
    base_path = Path("data/image")
    
    print("=" * 70)
    print("🎨 DeepGuard-X Lightweight Dataset Generator")
    print("=" * 70)
    print(f"\nGenerating {args.num_samples} samples per class...")
    print(f"Dataset type: {args.dataset_type}")
    
    # Create directories
    dirs = [
        base_path / "train" / "real",
        base_path / "train" / "fake",
        base_path / "val" / "real",
        base_path / "val" / "fake",
    ]
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\n" + "=" * 70)
    print("📸 Generating REAL images...")
    print("=" * 70)
    
    # Train real images (80%)
    num_train = int(args.num_samples * 0.8)
    num_val = args.num_samples - num_train
    
    for i in tqdm(range(num_train), desc="Training real"):
        image = create_real_image(i)
        output_path = base_path / "train" / "real" / f"real_{i:05d}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    for i in tqdm(range(num_val), desc="Validation real"):
        image = create_real_image(num_train + i)
        output_path = base_path / "val" / "real" / f"real_{i:05d}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    print("\n" + "=" * 70)
    print("🎭 Generating FAKE (deepfake) images...")
    print("=" * 70)
    
    for i in tqdm(range(num_train), desc="Training fake"):
        image = create_fake_image(i)
        output_path = base_path / "train" / "fake" / f"fake_{i:05d}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    for i in tqdm(range(num_val), desc="Validation fake"):
        image = create_fake_image(num_train + i)
        output_path = base_path / "val" / "fake" / f"fake_{i:05d}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    print("\n" + "=" * 70)
    print("✅ DATASET GENERATION COMPLETE!")
    print("=" * 70)
    
    total_samples = (num_train + num_val) * 2
    print(f"\n📊 Dataset Summary:")
    print(f"  Total images: {total_samples}")
    print(f"  Training set: {(num_train + num_val) * 2 // 2} real + {(num_train + num_val) * 2 // 2} fake")
    print(f"  Validation set: {num_val} real + {num_val} fake")
    print(f"  Location: {base_path}")
    
    # Estimate size
    size_per_image_mb = 0.1  # ~100 KB per JPEG
    total_size_mb = total_samples * size_per_image_mb
    
    print(f"\n💾 Estimated Size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
    
    print(f"\n🚀 Ready to train!")
    print(f"   Run: python scripts/train_image.py")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
