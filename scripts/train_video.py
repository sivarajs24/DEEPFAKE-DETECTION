"""
Training script for Video Deepfake Detection
Production-grade training with all bells and whistles
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
        dirpath=checkpoint_config.get('dirpath', 'checkpoints/video'),
        filename=checkpoint_config.get('filename', 'video_{epoch}_{val_loss:.4f}'),
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
            name=logging_config.get('experiment_name', 'video_training'),
            log_model=True
        )
        logger.info("Using WandB logger")
        
    elif logger_type == 'tensorboard':
        pl_logger = TensorBoardLogger(
            save_dir='logs',
            name=logging_config.get('experiment_name', 'video_training')
        )
        logger.info("Using TensorBoard logger")
        
    else:
        logger.warning(f"Unknown logger type: {logger_type}, using TensorBoard")
        pl_logger = TensorBoardLogger(save_dir='logs', name='video_training')
    
    return pl_logger


def train(config_path: str, resume_from: str = None, test_only: bool = False):
    """
    Main training function
    
    Args:
        config_path: Path to configuration YAML
        resume_from: Path to checkpoint to resume from
        test_only: Whether to only run testing
    """
    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_video_dataloaders(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'],
        test_path=config['data'].get('test_path'),
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
    
    # Setup callbacks and logger
    callbacks = setup_callbacks(config)
    pl_logger = setup_logger_pl(config)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training'].get('max_epochs', 50),
        accelerator='auto',
        devices='auto',
        precision=config['training'].get('precision', 16),
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        callbacks=callbacks,
        logger=pl_logger,
        log_every_n_steps=config.get('logging', {}).get('log_every_n_steps', 10),
        deterministic=True,
        benchmark=False,  # Set to True for faster training if input sizes are fixed
    )
    
    if test_only:
        # Test only
        if not test_loader:
            logger.error("No test data provided")
            return
        
        logger.info("Running test only...")
        if resume_from:
            model = VideoDeepfakeModule.load_from_checkpoint(resume_from, config=config)
        trainer.test(model, test_loader)
        
    else:
        # Training
        logger.info("Starting training...")
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_from
        )
        
        # Test after training
        if test_loader:
            logger.info("Running final test...")
            trainer.test(model, test_loader, ckpt_path='best')
    
    logger.info("Training completed!")
    
    # Export to ONNX if configured
    if 'export' in config and not test_only:
        export_config = config['export']
        onnx_path = export_config.get('onnx_path')
        
        if onnx_path:
            logger.info(f"Exporting model to ONNX: {onnx_path}")
            try:
                export_to_onnx(
                    model=model.model,
                    onnx_path=onnx_path,
                    input_size=(1, 3, config['data']['image_size'], config['data']['image_size']),
                    opset_version=export_config.get('opset_version', 14)
                )
            except Exception as e:
                logger.error(f"Failed to export to ONNX: {e}")


def export_to_onnx(model, onnx_path: str, input_size: tuple, opset_version: int = 14):
    """
    Export PyTorch model to ONNX
    
    Args:
        model: PyTorch model
        onnx_path: Output ONNX path
        input_size: Input tensor size (B, C, H, W)
        opset_version: ONNX opset version
    """
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    model.cpu()
    
    dummy_input = torch.randn(*input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Model exported to {onnx_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train Video Deepfake Detection Model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only run testing, no training'
    )
    
    args = parser.parse_args()
    
    train(args.config, args.resume, args.test_only)


if __name__ == '__main__':
    main()
