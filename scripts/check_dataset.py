"""
Dataset Statistics Checker
Displays information about your dataset
"""

from pathlib import Path
import cv2
import soundfile as sf
from collections import defaultdict


def check_video_properties(video_path: Path):
    """Check video properties"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return {
            'fps': fps,
            'resolution': (width, height),
            'frames': frame_count,
            'duration': duration,
            'size_mb': video_path.stat().st_size / (1024 * 1024)
        }
    except:
        return None


def check_audio_properties(audio_path: Path):
    """Check audio properties"""
    try:
        info = sf.info(str(audio_path))
        return {
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'duration': info.duration,
            'size_mb': audio_path.stat().st_size / (1024 * 1024)
        }
    except:
        return None


def main():
    print("=" * 70)
    print("📊 DeepGuard-X Dataset Statistics")
    print("=" * 70)
    
    base_path = Path("data")
    
    # Check video dataset
    print("\n🎥 VIDEO DATASET")
    print("-" * 70)
    
    video_stats = defaultdict(list)
    
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            path = base_path / "video" / split / label
            if path.exists():
                videos = list(path.glob("*.mp4")) + list(path.glob("*.avi")) + list(path.glob("*.mov"))
                print(f"\n{split.upper()} - {label.upper()}: {len(videos)} videos")
                
                if videos:
                    # Sample first video for properties
                    props = check_video_properties(videos[0])
                    if props:
                        print(f"  Sample properties:")
                        print(f"    Resolution: {props['resolution'][0]}x{props['resolution'][1]}")
                        print(f"    FPS: {props['fps']:.2f}")
                        print(f"    Duration: {props['duration']:.2f}s")
                        print(f"    Size: {props['size_mb']:.2f} MB")
                    
                    # Calculate total size
                    total_size = sum(v.stat().st_size for v in videos) / (1024 * 1024 * 1024)
                    print(f"  Total size: {total_size:.2f} GB")
                else:
                    print(f"  ⚠️ No videos found!")
            else:
                print(f"\n{split.upper()} - {label.upper()}: Directory not found!")
    
    # Check audio dataset
    print("\n" + "=" * 70)
    print("🔊 AUDIO DATASET")
    print("-" * 70)
    
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            path = base_path / "audio" / split / label
            if path.exists():
                audios = list(path.glob("*.wav")) + list(path.glob("*.mp3")) + list(path.glob("*.flac"))
                print(f"\n{split.upper()} - {label.upper()}: {len(audios)} audio files")
                
                if audios:
                    # Sample first audio for properties
                    props = check_audio_properties(audios[0])
                    if props:
                        print(f"  Sample properties:")
                        print(f"    Sample rate: {props['sample_rate']} Hz")
                        print(f"    Channels: {props['channels']}")
                        print(f"    Duration: {props['duration']:.2f}s")
                        print(f"    Size: {props['size_mb']:.2f} MB")
                    
                    # Calculate total size
                    total_size = sum(a.stat().st_size for a in audios) / (1024 * 1024 * 1024)
                    print(f"  Total size: {total_size:.2f} GB")
                else:
                    print(f"  ⚠️ No audio files found!")
            else:
                print(f"\n{split.upper()} - {label.upper()}: Directory not found!")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS")
    print("-" * 70)
    
    video_train_real = len(list((base_path / "video" / "train" / "real").glob("*.mp4")))
    video_train_fake = len(list((base_path / "video" / "train" / "fake").glob("*.mp4")))
    
    if video_train_real == 0 or video_train_fake == 0:
        print("❌ No video training data found!")
        print("   Run: python scripts/prepare_test_data.py --num-samples 20")
    elif video_train_real < 100 or video_train_fake < 100:
        print("⚠️  Small video dataset detected")
        print("   Recommended: 100+ videos per class for good performance")
    else:
        print("✅ Video dataset size looks good!")
    
    if abs(video_train_real - video_train_fake) > 0.2 * max(video_train_real, video_train_fake):
        print("⚠️  Class imbalance detected in video dataset")
        print("   Consider balancing real and fake samples")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
