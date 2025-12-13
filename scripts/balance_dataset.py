"""
Dataset Balancer for Deepfake Detection
Balances real and fake video counts for better training
"""

import shutil
import random
from pathlib import Path
from tqdm import tqdm
import argparse


def balance_dataset(data_dir, strategy='duplicate'):
    """
    Balance real and fake video counts
    
    Args:
        data_dir: Path to data/video directory
        strategy: 'duplicate' (duplicate minority) or 'downsample' (remove majority)
    """
    
    data_path = Path(data_dir)
    
    print("=" * 70)
    print("⚖️  Dataset Balancer")
    print("=" * 70)
    
    # Check train split
    train_path = data_path / "train"
    real_path = train_path / "real"
    fake_path = train_path / "fake"
    
    real_videos = list(real_path.glob("*.mp4"))
    fake_videos = list(fake_path.glob("*.mp4"))
    
    real_count = len(real_videos)
    fake_count = len(fake_videos)
    
    print(f"\n📊 BEFORE BALANCING:")
    print(f"   Real videos: {real_count}")
    print(f"   Fake videos: {fake_count}")
    print(f"   Ratio: {fake_count/real_count:.2f}:1 (fake:real)")
    print(f"   Imbalance: {abs(real_count - fake_count)} videos")
    
    if real_count == fake_count:
        print("\n✅ Dataset already balanced!")
        return
    
    # Determine strategy
    if real_count < fake_count:
        minority = real_videos
        majority = fake_videos
        minority_label = "real"
        majority_label = "fake"
        minority_path = real_path
        majority_path = fake_path
    else:
        minority = fake_videos
        majority = real_videos
        minority_label = "fake"
        majority_label = "real"
        minority_path = fake_path
        majority_path = real_path
    
    target_count = len(majority)
    current_count = len(minority)
    needed = target_count - current_count
    
    print(f"\n📋 BALANCING STRATEGY: {strategy.upper()}")
    print(f"   Minority class: {minority_label} ({current_count})")
    print(f"   Majority class: {majority_label} ({target_count})")
    print(f"   Need to balance: {needed} videos")
    
    if strategy == 'duplicate':
        print(f"\n🔄 Duplicating {minority_label} videos...")
        
        # Randomly select videos to duplicate
        to_duplicate = random.choices(minority, k=needed)
        
        for i, video_path in enumerate(tqdm(to_duplicate, desc=f"Duplicating {minority_label}")):
            # Create copy with new name
            new_name = f"{video_path.stem}_dup_{i}{video_path.suffix}"
            dest_path = minority_path / new_name
            shutil.copy2(video_path, dest_path)
        
        print(f"✅ Duplicated {needed} {minority_label} videos")
    
    elif strategy == 'downsample':
        print(f"\n🗑️  Removing excess {majority_label} videos...")
        
        # Randomly remove videos from majority class
        to_remove = random.sample(majority, k=needed)
        
        for video_path in tqdm(to_remove, desc=f"Removing {majority_label}"):
            video_path.unlink()
        
        print(f"✅ Removed {needed} {majority_label} videos")
    
    # Verify
    print("\n" + "=" * 70)
    print("✅ AFTER BALANCING:")
    
    new_real = len(list(real_path.glob("*.mp4")))
    new_fake = len(list(fake_path.glob("*.mp4")))
    
    print(f"   Real videos: {new_real}")
    print(f"   Fake videos: {new_fake}")
    print(f"   Ratio: {new_fake/new_real:.2f}:1 (fake:real)")
    
    if new_real == new_fake:
        print(f"\n🎉 Dataset perfectly balanced!")
    else:
        print(f"   Imbalance: {abs(new_real - new_fake)} videos")
    
    # Also balance validation
    print("\n" + "=" * 70)
    print("⚖️  Balancing Validation Set...")
    
    val_path = data_path / "val"
    val_real_path = val_path / "real"
    val_fake_path = val_path / "fake"
    
    val_real_videos = list(val_real_path.glob("*.mp4"))
    val_fake_videos = list(val_fake_path.glob("*.mp4"))
    
    val_real_count = len(val_real_videos)
    val_fake_count = len(val_fake_videos)
    
    print(f"\nValidation Set BEFORE:")
    print(f"   Real: {val_real_count}, Fake: {val_fake_count}")
    
    if val_real_count < val_fake_count:
        val_to_duplicate = random.choices(val_real_videos, k=val_fake_count - val_real_count)
        for i, v in enumerate(val_to_duplicate):
            new_name = f"{v.stem}_dup_{i}{v.suffix}"
            shutil.copy2(v, val_real_path / new_name)
    elif val_fake_count < val_real_count:
        val_to_duplicate = random.choices(val_fake_videos, k=val_real_count - val_fake_count)
        for i, v in enumerate(val_to_duplicate):
            new_name = f"{v.stem}_dup_{i}{v.suffix}"
            shutil.copy2(v, val_fake_path / new_name)
    
    val_real_count = len(list(val_real_path.glob("*.mp4")))
    val_fake_count = len(list(val_fake_path.glob("*.mp4")))
    
    print(f"Validation Set AFTER:")
    print(f"   Real: {val_real_count}, Fake: {val_fake_count}")
    
    print("\n" + "=" * 70)
    print("🚀 Dataset balancing complete!")
    print("=" * 70)
    print("\n✅ Ready to train with balanced dataset!")
    print("Run: python scripts/train_video.py --config configs/video/efficientnet_b3.yaml")


def main():
    parser = argparse.ArgumentParser(description="Balance deepfake dataset")
    parser.add_argument(
        "--data-dir",
        default="data/video",
        help="Path to video data directory"
    )
    parser.add_argument(
        "--strategy",
        choices=['duplicate', 'downsample'],
        default='duplicate',
        help="Balancing strategy: duplicate minority class or downsample majority class"
    )
    
    args = parser.parse_args()
    
    balance_dataset(args.data_dir, args.strategy)


if __name__ == "__main__":
    main()
