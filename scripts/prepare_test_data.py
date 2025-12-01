"""
Test Data Preparation Script
Creates synthetic test data for development and testing
"""

import argparse
import cv2
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def create_test_video(output_path: Path, duration: int = 3, fps: int = 30):
    """Create a simple test video"""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    num_frames = duration * fps
    
    for i in range(num_frames):
        # Create a simple animated frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add moving shapes
        center_x = int(width / 2 + 100 * np.sin(2 * np.pi * i / num_frames))
        center_y = int(height / 2 + 100 * np.cos(2 * np.pi * i / num_frames))
        
        # Random color
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        cv2.circle(frame, (center_x, center_y), 50, color, -1)
        cv2.putText(frame, f"Frame {i}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()


def create_test_audio(output_path: Path, duration: int = 3, sample_rate: int = 16000):
    """Create a simple test audio file"""
    num_samples = duration * sample_rate
    
    # Generate a simple sine wave
    t = np.linspace(0, duration, num_samples)
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, num_samples)
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    sf.write(str(output_path), audio, sample_rate)


def main():
    parser = argparse.ArgumentParser(description="Create test dataset for development")
    parser.add_argument("--num-samples", type=int, default=10, 
                        help="Number of samples per class")
    parser.add_argument("--video-duration", type=int, default=3,
                        help="Video duration in seconds")
    parser.add_argument("--audio-duration", type=int, default=3,
                        help="Audio duration in seconds")
    parser.add_argument("--create-video", action="store_true", default=True,
                        help="Create video test data")
    parser.add_argument("--create-audio", action="store_true", default=True,
                        help="Create audio test data")
    
    args = parser.parse_args()
    
    base_path = Path("data")
    
    print("=" * 60)
    print("🎬 DeepGuard-X Test Data Generator")
    print("=" * 60)
    print(f"\nCreating {args.num_samples} samples per class...")
    
    # Create video test data
    if args.create_video:
        print("\n📹 Creating test videos...")
        
        video_paths = [
            base_path / "video" / "train" / "real",
            base_path / "video" / "train" / "fake",
            base_path / "video" / "val" / "real",
            base_path / "video" / "val" / "fake",
        ]
        
        for path in tqdm(video_paths, desc="Video directories"):
            path.mkdir(parents=True, exist_ok=True)
            
            num_samples = args.num_samples if "train" in str(path) else max(2, args.num_samples // 5)
            
            for i in range(num_samples):
                video_path = path / f"test_video_{i:03d}.mp4"
                if not video_path.exists():
                    create_test_video(video_path, duration=args.video_duration)
        
        print("✅ Video test data created!")
    
    # Create audio test data
    if args.create_audio:
        print("\n🔊 Creating test audio...")
        
        audio_paths = [
            base_path / "audio" / "train" / "real",
            base_path / "audio" / "train" / "fake",
            base_path / "audio" / "val" / "real",
            base_path / "audio" / "val" / "fake",
        ]
        
        for path in tqdm(audio_paths, desc="Audio directories"):
            path.mkdir(parents=True, exist_ok=True)
            
            num_samples = args.num_samples if "train" in str(path) else max(2, args.num_samples // 5)
            
            for i in range(num_samples):
                audio_path = path / f"test_audio_{i:03d}.wav"
                if not audio_path.exists():
                    create_test_audio(audio_path, duration=args.audio_duration)
        
        print("✅ Audio test data created!")
    
    print("\n" + "=" * 60)
    print("🎉 Test dataset created successfully!")
    print("=" * 60)
    print("\n📊 Dataset structure:")
    print(f"   Video Train: {len(list((base_path / 'video' / 'train' / 'real').glob('*.mp4')))} real, "
          f"{len(list((base_path / 'video' / 'train' / 'fake').glob('*.mp4')))} fake")
    print(f"   Video Val:   {len(list((base_path / 'video' / 'val' / 'real').glob('*.mp4')))} real, "
          f"{len(list((base_path / 'video' / 'val' / 'fake').glob('*.mp4')))} fake")
    print(f"   Audio Train: {len(list((base_path / 'audio' / 'train' / 'real').glob('*.wav')))} real, "
          f"{len(list((base_path / 'audio' / 'train' / 'fake').glob('*.wav')))} fake")
    print(f"   Audio Val:   {len(list((base_path / 'audio' / 'val' / 'real').glob('*.wav')))} real, "
          f"{len(list((base_path / 'audio' / 'val' / 'fake').glob('*.wav')))} fake")
    
    print("\n🚀 You can now train models with:")
    print("   python scripts/train_video.py --config configs/video/efficientnet_b3.yaml")
    print("   python scripts/train_audio.py --config configs/audio/wav2vec2.yaml")


if __name__ == "__main__":
    main()
