"""
Training script for Image Deepfake Detection
Adapted from video training pipeline
"""

import os
import sys
from pathlib import Path
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import warnings

# Add project root (so src is importable)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, get_logger
from src.data.video_dataset import create_video_dataloaders
from src.training.video_lightning import VideoDeepfakeModule

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


def setup_callbacks(config: dict) -> list:
    """
    Setup training callbacks
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpoint
    checkpoint_config = config.get('checkpoint', {})
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_config.get('dirpath', 'checkpoints/image'),
        filename=checkpoint_config.get('filename', 'image_{epoch}_{val_loss:.4f}'),
        monitor=checkpoint_config.get('monitor', 'val_loss'),
        mode=checkpoint_config.get('mode', 'min'),
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=checkpoint_config.get('save_last', True),
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if 'early_stopping' in config.get('training', {}):
        es_config = config['training']['early_stopping']
        early_stop_callback = EarlyStopping(
            monitor=es_config.get('monitor', 'val_loss'),
            patience=es_config.get('patience', 7),
            mode=es_config.get('mode', 'min'),
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Rich progress bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    logger.info(f"Setup {len(callbacks)} callbacks")
    return callbacks


def setup_logger_pl(config: dict):
    """
    Setup PyTorch Lightning logger
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Logger instance
    """
    logging_config = config.get('logging', {})
    logger_type = logging_config.get('logger', 'tensorboard').lower()
    
    if logger_type == 'wandb':
        pl_logger = WandbLogger(
            project=logging_config.get('project_name', 'deepguard-x'),
            name=logging_config.get('experiment_name', 'image_training'),
            log_model=True
        )
        logger.info("Using WandB logger")
        
    elif logger_type == 'tensorboard':
        pl_logger = TensorBoardLogger(
            save_dir='logs',
            name=logging_config.get('experiment_name', 'image_training'),
            default_hp_metric=False
        )
        logger.info("Using TensorBoard logger")
        
    else:
        logger.warning(f"Unknown logger type: {logger_type}, using TensorBoard")
        pl_logger = TensorBoardLogger(save_dir='logs', name='image_training')
        
    return pl_logger


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='DeepGuard-X Image Training')
    parser.add_argument('--config', type=str, default='configs/image/efficientnet_b3.yaml',
                        help='Path to configuration file')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with args
    if args.epochs:
        config['training']['max_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
        
    # Set seed for reproducibility
    pl.seed_everything(42)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_video_dataloaders(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'],
        test_path=config['data'].get('test_path'),
        config=config['data'],
        num_workers=config['data'].get('num_workers', 4)
    )
    
    # Initialize model
    logger.info(f"Initializing model: {config['model']['architecture']}")
    model = VideoDeepfakeModule(config)
    
    # Setup callbacks and logger
    callbacks = setup_callbacks(config)
    pl_logger = setup_logger_pl(config)
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=config['training'].get('precision', 16),
        callbacks=callbacks,
        logger=pl_logger,
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        log_every_n_steps=10
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    if test_loader:
        logger.info("Starting testing...")
        trainer.test(model, test_loader)
        
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
