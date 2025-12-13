"""
FaceForensics++ Dataset Organizer
Organize downloaded FaceForensics++ files into train/val/real/fake structure
"""

import os
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
import random


def organize_faceforensics(source_dir, output_dir, train_ratio=0.8):
    """
    Organize FaceForensics++ dataset
    
    Args:
        source_dir: Path where FaceForensics++ was downloaded
        output_dir: Output directory (data/video/)
        train_ratio: Ratio of training vs validation (0.8 = 80% train)
    """
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    print("=" * 70)
    print("🎬 FaceForensics++ Dataset Organizer")
    print("=" * 70)
    
    # Create output structure
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            (output_path / split / label).mkdir(parents=True, exist_ok=True)
    
    print("\n📁 Output structure created:")
    print(f"   {output_path}/train/real/")
    print(f"   {output_path}/train/fake/")
    print(f"   {output_path}/val/real/")
    print(f"   {output_path}/val/fake/")
    
    # Find real videos (original actors)
    print("\n🔍 Scanning for real videos...")
    
    # FaceForensics structure typically:
    # FaceForensics_video_frames/c23/videos/actors/
    
    original_videos_path = source_path / "original_sequences" / "videos"
    if not original_videos_path.exists():
        original_videos_path = source_path / "videos" / "original"
    
    if original_videos_path.exists():
        real_videos = list(original_videos_path.glob("**/*.mp4"))
        print(f"✅ Found {len(real_videos)} real videos")
        
        # Split real videos into train/val
        random.shuffle(real_videos)
        split_idx = int(len(real_videos) * train_ratio)
        train_real = real_videos[:split_idx]
        val_real = real_videos[split_idx:]
        
        # Copy real videos
        print("\n📋 Copying real videos to train...")
        for video in tqdm(train_real, desc="Real Train"):
            dest = output_path / "train" / "real" / video.name
            if not dest.exists():
                shutil.copy2(video, dest)
        
        print("📋 Copying real videos to val...")
        for video in tqdm(val_real, desc="Real Val"):
            dest = output_path / "val" / "real" / video.name
            if not dest.exists():
                shutil.copy2(video, dest)
    
    # Find fake videos (manipulated)
    print("\n🔍 Scanning for fake videos...")
    
    # FaceForensics manipulated structure typically:
    # FaceForensics_video_frames/c23/videos/manipulated_sequences/
    
    manipulated_paths = [
        source_path / "manipulated_sequences" / "videos",
        source_path / "videos" / "manipulated",
        source_path / "deepfakes" / "videos",  # Alternative name
    ]
    
    all_fake_videos = []
    for manip_path in manipulated_paths:
        if manip_path.exists():
            all_fake_videos.extend(manip_path.glob("**/*.mp4"))
    
    if all_fake_videos:
        print(f"✅ Found {len(all_fake_videos)} fake videos")
        
        # Split fake videos into train/val
        random.shuffle(all_fake_videos)
        split_idx = int(len(all_fake_videos) * train_ratio)
        train_fake = all_fake_videos[:split_idx]
        val_fake = all_fake_videos[split_idx:]
        
        # Copy fake videos
        print("\n📋 Copying fake videos to train...")
        for video in tqdm(train_fake, desc="Fake Train"):
            dest = output_path / "train" / "fake" / video.name
            if not dest.exists():
                shutil.copy2(video, dest)
        
        print("📋 Copying fake videos to val...")
        for video in tqdm(val_fake, desc="Fake Val"):
            dest = output_path / "val" / "fake" / video.name
            if not dest.exists():
                shutil.copy2(video, dest)
    
    # Statistics
    print("\n" + "=" * 70)
    print("📊 DATASET STATISTICS")
    print("=" * 70)
    
    train_real_count = len(list((output_path / "train" / "real").glob("*.mp4")))
    train_fake_count = len(list((output_path / "train" / "fake").glob("*.mp4")))
    val_real_count = len(list((output_path / "val" / "real").glob("*.mp4")))
    val_fake_count = len(list((output_path / "val" / "fake").glob("*.mp4")))
    
    print(f"\n✅ Training Set:")
    print(f"   Real: {train_real_count} videos")
    print(f"   Fake: {train_fake_count} videos")
    print(f"   Total: {train_real_count + train_fake_count} videos")
    
    print(f"\n✅ Validation Set:")
    print(f"   Real: {val_real_count} videos")
    print(f"   Fake: {val_fake_count} videos")
    print(f"   Total: {val_real_count + val_fake_count} videos")
    
    print(f"\n✅ GRAND TOTAL: {train_real_count + train_fake_count + val_real_count + val_fake_count} videos")
    
    print("\n" + "=" * 70)
    print("🚀 Ready to train!")
    print("=" * 70)
    print("\nRun: python scripts/train_video.py --config configs/video/efficientnet_b3.yaml")


def main():
    parser = argparse.ArgumentParser(description="Organize FaceForensics++ dataset")
    parser.add_argument(
        "--source",
        default="FaceForensics++",
        help="Source directory where FaceForensics++ was downloaded"
    )
    parser.add_argument(
        "--output",
        default="data/video",
        help="Output directory for organized dataset"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of training vs validation (default: 0.8 = 80/20)"
    )
    
    args = parser.parse_args()
    
    organize_faceforensics(args.source, args.output, args.train_ratio)


if __name__ == "__main__":
    main()
