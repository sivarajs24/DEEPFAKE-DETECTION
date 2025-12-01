"""
Dataset Downloader for DeepGuard-X
Automates downloading of popular deepfake detection datasets
"""

import argparse
import subprocess
import sys
from pathlib import Path
import urllib.request
import zipfile
import tarfile
import gdown
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file from URL with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_from_gdrive(file_id, output_path):
    """Download from Google Drive"""
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, str(output_path), quiet=False)


def extract_archive(archive_path, extract_to):
    """Extract zip or tar archive"""
    print(f"📦 Extracting {archive_path.name}...")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    
    print(f"✅ Extracted to {extract_to}")


def download_faceforensics_sample():
    """
    Download FaceForensics++ sample dataset
    Note: Full dataset requires registration at https://github.com/ondyari/FaceForensics
    """
    print("\n" + "="*70)
    print("📥 FaceForensics++ Sample Dataset")
    print("="*70)
    print("\n⚠️  NOTE: This downloads a small sample for testing.")
    print("For the full dataset, you need to:")
    print("1. Visit: https://github.com/ondyari/FaceForensics")
    print("2. Request access via the form")
    print("3. Use their download script")
    print("\nContinuing with sample download...\n")
    
    # Sample videos (these are public examples)
    sample_urls = {
        'real': [
            # Add public sample URLs here
            # Note: You'll need to find public samples or use the official download
        ],
        'fake': [
            # Add public sample URLs here
        ]
    }
    
    base_path = Path("data/video/train")
    
    print("⚠️  No public samples available - please use official FaceForensics++ download")
    print("Or create test data with: python scripts/prepare_test_data.py")
    return False


def download_dfdc_sample():
    """
    Download DFDC sample
    Note: Full dataset requires Kaggle account
    """
    print("\n" + "="*70)
    print("📥 Deepfake Detection Challenge (DFDC) Dataset")
    print("="*70)
    print("\n⚠️  DFDC requires Kaggle account and API key")
    print("\nTo download DFDC:")
    print("1. Create Kaggle account: https://www.kaggle.com/")
    print("2. Go to Account settings -> API -> Create New API Token")
    print("3. This downloads kaggle.json")
    print("4. Place kaggle.json in: ~/.kaggle/ (Linux/Mac) or C:\\Users\\<you>\\.kaggle\\ (Windows)")
    print("5. Run: kaggle competitions download -c deepfake-detection-challenge")
    
    try:
        import kaggle
        print("\n✅ Kaggle API detected!")
        
        response = input("\nDownload DFDC dataset? This is LARGE (~470GB). (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return False
        
        output_path = Path("data/downloads/dfdc")
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n📥 Downloading DFDC dataset...")
        kaggle.api.competition_download_files('deepfake-detection-challenge', path=str(output_path))
        
        print("✅ Download complete!")
        print(f"📁 Files saved to: {output_path}")
        print("\n⚠️  You'll need to extract and organize the files manually")
        return True
        
    except ImportError:
        print("\n❌ Kaggle API not installed")
        print("Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Kaggle account")
        print("2. API token (kaggle.json) in correct location")
        print("3. Accepted competition rules")
        return False


def download_celebdf():
    """
    Download Celeb-DF dataset
    """
    print("\n" + "="*70)
    print("📥 Celeb-DF Dataset")
    print("="*70)
    print("\nCeleb-DF download instructions:")
    print("1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics")
    print("2. Fill out the download request form")
    print("3. You'll receive download links via email")
    print("4. Download and extract to data/video/")
    
    print("\n⚠️  Manual download required - automated download not available")
    return False


def download_asvspoof():
    """
    Download ASVspoof 2019 dataset for audio
    """
    print("\n" + "="*70)
    print("📥 ASVspoof 2019 Dataset (Audio)")
    print("="*70)
    
    print("\nDownload ASVspoof 2019:")
    print("1. Visit: https://www.asvspoof.org/index2019.html")
    print("2. Download LA (Logical Access) partition")
    print("3. Extract to data/audio/")
    
    # Public download link for LA partition
    la_url = "https://datashare.ed.ac.uk/download/DS_10283_3336.zip"
    
    response = input("\nAttempt download of ASVspoof 2019 LA? (~3GB) (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return False
    
    try:
        output_path = Path("data/downloads/asvspoof")
        output_path.mkdir(parents=True, exist_ok=True)
        
        zip_path = output_path / "asvspoof2019_LA.zip"
        
        print(f"\n📥 Downloading ASVspoof 2019 LA...")
        download_url(la_url, zip_path)
        
        print("\n📦 Extracting...")
        extract_archive(zip_path, output_path)
        
        print("\n✅ Download complete!")
        print(f"📁 Files saved to: {output_path}")
        print("\n⚠️  Organize files into data/audio/train/real and data/audio/train/fake")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("Please download manually from: https://www.asvspoof.org/")
        return False


def download_sample_videos():
    """
    Download sample videos from public sources for testing
    """
    print("\n" + "="*70)
    print("📥 Sample Test Videos")
    print("="*70)
    
    print("\n📹 Downloading sample videos for testing...")
    
    # These would be public domain or creative commons videos
    # For production, use actual datasets
    
    print("⚠️  For development, use: python scripts/prepare_test_data.py")
    print("This creates synthetic test data for immediate use.")
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for DeepGuard-X",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_datasets.py --dataset faceforensics
  python scripts/download_datasets.py --dataset dfdc
  python scripts/download_datasets.py --dataset asvspoof
  python scripts/download_datasets.py --all
  
Note: Most datasets require registration and manual download.
For quick testing, use: python scripts/prepare_test_data.py
        """
    )
    
    parser.add_argument(
        "--dataset",
        choices=['faceforensics', 'dfdc', 'celebdf', 'asvspoof', 'samples'],
        help="Dataset to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show download instructions for all datasets"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("🎬 DeepGuard-X Dataset Downloader")
    print("="*70)
    
    if args.all:
        print("\n📚 All Dataset Download Information:\n")
        download_faceforensics_sample()
        download_dfdc_sample()
        download_celebdf()
        download_asvspoof()
        download_sample_videos()
        
    elif args.dataset == 'faceforensics':
        download_faceforensics_sample()
        
    elif args.dataset == 'dfdc':
        download_dfdc_sample()
        
    elif args.dataset == 'celebdf':
        download_celebdf()
        
    elif args.dataset == 'asvspoof':
        download_asvspoof()
        
    elif args.dataset == 'samples':
        download_sample_videos()
        
    else:
        print("\n⚠️  No dataset specified!")
        print("\nQuick start options:")
        print("\n1. 🚀 CREATE TEST DATA (Recommended for development):")
        print("   python scripts/prepare_test_data.py --num-samples 50")
        print("\n2. 📥 VIEW DATASET DOWNLOAD INFO:")
        print("   python scripts/download_datasets.py --all")
        print("\n3. 📥 DOWNLOAD SPECIFIC DATASET:")
        print("   python scripts/download_datasets.py --dataset asvspoof")
        print("\nFor production use, download real datasets:")
        print("  • FaceForensics++: https://github.com/ondyari/FaceForensics")
        print("  • DFDC: https://www.kaggle.com/c/deepfake-detection-challenge")
        print("  • Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics")
        print("  • ASVspoof: https://www.asvspoof.org/")
    
    print("\n" + "="*70)
    print("💡 TIP: Most datasets require registration/request forms")
    print("For immediate testing, use: python scripts/prepare_test_data.py")
    print("="*70)


if __name__ == "__main__":
    main()
