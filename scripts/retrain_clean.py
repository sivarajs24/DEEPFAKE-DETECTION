"""
Retrain EfficientNet-B3 on clean data split.
Computes exact class weights from the data, trains, tests, and exports to ONNX.

Usage:
    python scripts/retrain_clean.py
"""

import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger
import warnings

from src.utils import load_config, get_logger
from src.data.video_dataset import create_video_dataloaders
from src.training.video_lightning import VideoDeepfakeModule

logger = get_logger(__name__)
warnings.filterwarnings('ignore')

CONFIG_PATH = "configs/video/efficientnet_b3_clean.yaml"
CLEAN_DATA = Path("data/video_clean")


def compute_class_weights(train_path: Path):
    """Compute exact class weights from training data."""
    real_count = len(list((train_path / "real").glob("*.mp4")))
    fake_count = len(list((train_path / "fake").glob("*.mp4")))
    total = real_count + fake_count

    w_real = total / (2 * real_count)
    w_fake = total / (2 * fake_count)

    logger.info(f"Training data: {real_count} real + {fake_count} fake = {total} total")
    logger.info(f"Class weights: real={w_real:.4f}, fake={w_fake:.4f}")
    return [w_real, w_fake]


def main():
    # Check clean data exists
    if not CLEAN_DATA.exists():
        logger.error("Clean data not found! Run 'python scripts/prepare_clean_split.py' first.")
        sys.exit(1)

    # Load config
    config = load_config(CONFIG_PATH)

    # Compute exact weights from data
    weights = compute_class_weights(CLEAN_DATA / "train")
    config['training']['class_weights'] = weights

    # Seed
    pl.seed_everything(42, workers=True)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_video_dataloaders(
        train_path=str(CLEAN_DATA / "train"),
        val_path=str(CLEAN_DATA / "val"),
        test_path=str(CLEAN_DATA / "test"),
        config={
            'batch_size': config['training']['batch_size'],
            'image_size': config['data']['image_size'],
            'augmentation': config['data'].get('augmentation', {}),
            'pin_memory': config['data'].get('pin_memory', True)
        },
        num_workers=config['data'].get('num_workers', 4)
    )

    # Create model
    logger.info("Initializing model...")
    model = VideoDeepfakeModule(config)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/video_clean",
            filename="efficientnet_b3_clean_{epoch}_{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            mode="min",
            verbose=True
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar()
    ]

    # Logger
    pl_logger = TensorBoardLogger(
        save_dir="logs",
        name="video_efficientnet_b3_clean"
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training'].get('max_epochs', 50),
        accelerator='auto',
        devices='auto',
        precision=config['training'].get('precision', 16),
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        callbacks=callbacks,
        logger=pl_logger,
        log_every_n_steps=10,
        deterministic=True,
    )

    # Train
    logger.info("=" * 60)
    logger.info("Starting clean retrain...")
    logger.info("=" * 60)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test on held-out test set
    logger.info("=" * 60)
    logger.info("Evaluating on held-out test set...")
    logger.info("=" * 60)
    trainer.test(model, test_loader, ckpt_path="best")

    # Export to ONNX
    logger.info("Exporting to ONNX...")
    best_model = VideoDeepfakeModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, config=config
    )
    best_model.eval()
    best_model.model.cpu()

    onnx_path = "models/video_efficientnet_b3.onnx"
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        best_model.model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    logger.info(f"ONNX model saved to {onnx_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {trainer.checkpoint_callback.best_model_score:.6f}")
    print(f"ONNX model: {onnx_path}")
    print("\nRun this to extract your final metrics:")
    print("  python scripts/extract_metrics.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
