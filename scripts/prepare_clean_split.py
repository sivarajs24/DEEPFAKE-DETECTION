"""
Prepare Clean Train/Val/Test Split for Deepfake Detection
- Removes duplicate files (keeps one per source video)
- Splits by source video ID (no data leakage)
- Creates balanced splits with 70/15/15 ratio
"""

import shutil
import random
import hashlib
from pathlib import Path
from collections import defaultdict

random.seed(42)

DATA_ROOT = Path("data/video")
CLEAN_ROOT = Path("data/video_clean")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def get_base_id(filename: str) -> str:
    """Extract base video ID, stripping _dup_ suffixes."""
    name = Path(filename).stem
    if "_dup_" in name:
        return name.split("_dup_")[0]
    return name


def collect_unique_videos(directory: Path):
    """Return one file per unique base ID."""
    groups = defaultdict(list)
    for f in sorted(directory.glob("*.mp4")):
        base = get_base_id(f.name)
        groups[base].append(f)
    # Keep first file per group
    return {base: files[0] for base, files in groups.items()}


def split_by_ids(id_to_file: dict, train_r=0.70, val_r=0.15):
    """Split IDs into train/val/test with no overlap."""
    ids = sorted(id_to_file.keys())
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    return train_ids, val_ids, test_ids


def copy_files(ids, id_to_file, dest_dir):
    """Copy selected video files to destination."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for vid_id in ids:
        src = id_to_file[vid_id]
        dst = dest_dir / src.name
        shutil.copy2(src, dst)


def main():
    print("=" * 60)
    print("Preparing Clean Dataset Split (No Leakage)")
    print("=" * 60)

    # Clean output directory
    if CLEAN_ROOT.exists():
        print(f"\nRemoving existing {CLEAN_ROOT}...")
        shutil.rmtree(CLEAN_ROOT)

    # --- REAL videos ---
    print("\n--- REAL videos ---")
    real_dirs = [DATA_ROOT / "train" / "real", DATA_ROOT / "val" / "real"]
    all_real = {}
    for d in real_dirs:
        if d.exists():
            unique = collect_unique_videos(d)
            print(f"  {d}: {len(list(d.glob('*.mp4')))} files -> {len(unique)} unique")
            all_real.update(unique)
    print(f"  Total unique real videos: {len(all_real)}")

    real_train, real_val, real_test = split_by_ids(all_real)
    print(f"  Split: train={len(real_train)}, val={len(real_val)}, test={len(real_test)}")

    # --- FAKE videos ---
    print("\n--- FAKE videos ---")
    fake_dirs = [DATA_ROOT / "train" / "fake", DATA_ROOT / "val" / "fake"]
    all_fake = {}
    for d in fake_dirs:
        if d.exists():
            unique = collect_unique_videos(d)
            print(f"  {d}: {len(list(d.glob('*.mp4')))} files -> {len(unique)} unique")
            all_fake.update(unique)
    print(f"  Total unique fake videos: {len(all_fake)}")

    fake_train, fake_val, fake_test = split_by_ids(all_fake)
    print(f"  Split: train={len(fake_train)}, val={len(fake_val)}, test={len(fake_test)}")

    # --- Copy files ---
    print("\nCopying files to clean directory...")
    copy_files(real_train, all_real, CLEAN_ROOT / "train" / "real")
    copy_files(real_val, all_real, CLEAN_ROOT / "val" / "real")
    copy_files(real_test, all_real, CLEAN_ROOT / "test" / "real")

    copy_files(fake_train, all_fake, CLEAN_ROOT / "train" / "fake")
    copy_files(fake_val, all_fake, CLEAN_ROOT / "val" / "fake")
    copy_files(fake_test, all_fake, CLEAN_ROOT / "test" / "fake")

    # --- Verify no leakage ---
    print("\n--- Leakage Verification ---")
    for label in ["real", "fake"]:
        train_ids = set(get_base_id(f.name) for f in (CLEAN_ROOT / "train" / label).glob("*.mp4"))
        val_ids = set(get_base_id(f.name) for f in (CLEAN_ROOT / "val" / label).glob("*.mp4"))
        test_ids = set(get_base_id(f.name) for f in (CLEAN_ROOT / "test" / label).glob("*.mp4"))

        tv_overlap = train_ids & val_ids
        tt_overlap = train_ids & test_ids
        vt_overlap = val_ids & test_ids

        print(f"  {label}: train∩val={len(tv_overlap)}, train∩test={len(tt_overlap)}, val∩test={len(vt_overlap)}")
        if tv_overlap or tt_overlap or vt_overlap:
            print(f"    ❌ LEAKAGE DETECTED!")
        else:
            print(f"    ✅ No leakage")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        real_count = len(list((CLEAN_ROOT / split / "real").glob("*.mp4")))
        fake_count = len(list((CLEAN_ROOT / split / "fake").glob("*.mp4")))
        print(f"  {split:5s}: {real_count:4d} real + {fake_count:4d} fake = {real_count + fake_count:5d} total")

    total_real = len(all_real)
    total_fake = len(all_fake)
    print(f"\n  Class ratio (real:fake) = 1:{total_fake/total_real:.1f}")
    print(f"  Recommended class weights for CrossEntropyLoss:")
    total = total_real + total_fake
    w_real = total / (2 * total_real)
    w_fake = total / (2 * total_fake)
    print(f"    Real weight: {w_real:.4f}")
    print(f"    Fake weight: {w_fake:.4f}")
    print(f"\n  Use: CrossEntropyLoss(weight=torch.tensor([{w_real:.4f}, {w_fake:.4f}]))")

    print(f"\nClean data saved to: {CLEAN_ROOT.resolve()}")
    print("Done! ✅")


if __name__ == "__main__":
    main()
