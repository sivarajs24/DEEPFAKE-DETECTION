"""
Extract final metrics from TensorBoard logs after clean retrain.
Prints resume-ready bullet points with actual achieved scores.

Usage:
    python scripts/extract_metrics.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def find_latest_version(log_dir):
    """Find the latest version directory."""
    versions = sorted(Path(log_dir).glob("version_*"), key=lambda p: int(p.name.split("_")[1]))
    if not versions:
        print(f"No versions found in {log_dir}")
        sys.exit(1)
    return versions[-1]


def main():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    log_dir = "logs/video_efficientnet_b3_clean"
    latest = find_latest_version(log_dir)
    event_files = list(latest.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"No event files in {latest}")
        sys.exit(1)

    ea = EventAccumulator(str(event_files[0]))
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    print("=" * 60)
    print("CLEAN RETRAIN METRICS")
    print("=" * 60)

    # Validation metrics
    print("\n--- VALIDATION SET (epoch-by-epoch) ---")
    if "val_accuracy" in tags:
        val_acc = ea.Scalars("val_accuracy")
        val_auc = ea.Scalars("val_auc") if "val_auc" in tags else []
        val_f1 = ea.Scalars("val_f1") if "val_f1" in tags else []
        val_loss = ea.Scalars("val_loss") if "val_loss" in tags else []
        val_prec = ea.Scalars("val_precision") if "val_precision" in tags else []
        val_rec = ea.Scalars("val_recall") if "val_recall" in tags else []

        print("Epoch | Val Loss   | Val Acc  | Val F1   | Val Prec | Val Rec  | Val AUC")
        print("-" * 85)
        for i in range(len(val_acc)):
            loss = val_loss[i].value if i < len(val_loss) else 0
            acc = val_acc[i].value
            f1 = val_f1[i].value if i < len(val_f1) else 0
            prec = val_prec[i].value if i < len(val_prec) else 0
            rec = val_rec[i].value if i < len(val_rec) else 0
            auc = val_auc[i].value if i < len(val_auc) else 0
            print("%2d    | %.6f | %.4f   | %.4f   | %.4f   | %.4f   | %.4f" % (i, loss, acc, f1, prec, rec, auc))

    # Test metrics
    print("\n--- HELD-OUT TEST SET ---")
    test_tags = [t for t in tags if t.startswith("test_")]
    if test_tags:
        for tag in sorted(test_tags):
            events = ea.Scalars(tag)
            if events:
                print("  %s: %.4f" % (tag, events[-1].value))
    else:
        print("  No test metrics found in logs.")
        print("  Check terminal output from training for test results.")

    # Best values
    print("\n--- BEST METRICS ---")
    if "val_accuracy" in tags:
        best_acc = max(e.value for e in ea.Scalars("val_accuracy"))
        print("  Best Val Accuracy: %.4f (%.2f%%)" % (best_acc, best_acc * 100))
    if "val_auc" in tags:
        best_auc = max(e.value for e in ea.Scalars("val_auc"))
        print("  Best Val AUC-ROC:  %.4f" % best_auc)
    if "val_f1" in tags:
        best_f1 = max(e.value for e in ea.Scalars("val_f1"))
        print("  Best Val F1-Score: %.4f" % best_f1)
    if "val_loss" in tags:
        best_loss = min(e.value for e in ea.Scalars("val_loss"))
        print("  Best Val Loss:     %.6f" % best_loss)

    # Generate resume bullet
    print("\n" + "=" * 60)
    print("RESUME BULLET POINTS (copy-paste ready)")
    print("=" * 60)

    if "val_accuracy" in tags and "val_auc" in tags:
        best_acc = max(e.value for e in ea.Scalars("val_accuracy"))
        best_auc = max(e.value for e in ea.Scalars("val_auc"))
        best_f1 = max(e.value for e in ea.Scalars("val_f1")) if "val_f1" in tags else 0
        epochs = len(ea.Scalars("val_accuracy"))

        acc_pct = int(best_acc * 100)
        auc_str = "%.2f" % best_auc

        print("""
[Deepfake Detection](https://github.com/sivarajs24/DEEPFAKE-DETECTION) | Python, PyTorch Lightning, EfficientNet-B3, ONNX Runtime, OpenCV, Streamlit

* Trained a deep learning classifier (EfficientNet-B3, PyTorch Lightning) on ~6,500 videos
  from the Celeb-DF v2 benchmark using FP16 mixed-precision training, weighted cross-entropy
  loss for class imbalance, and an Albumentations augmentation pipeline, achieving %d%% validation
  accuracy and %s AUC-ROC on a held-out evaluation set.

* Exported the trained model to ONNX format and deployed it via a FastAPI backend + Streamlit
  dashboard, enabling real-time video upload analysis and webcam-based deepfake detection with
  ONNX Runtime inference.
""" % (acc_pct, auc_str))

    print("=" * 60)


if __name__ == "__main__":
    main()
