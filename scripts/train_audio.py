"""
Train Audio Deepfake Detection Model
Similar structure to video training
"""

import os
import sys
from pathlib import Path
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, get_logger
from src.data.audio_dataset import create_audio_dataloaders
from src.models.audio_models import create_audio_model

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


class AudioDeepfakeModule(pl.LightningModule):
    """PyTorch Lightning module for audio deepfake detection"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Create model
        self.model = create_audio_model(config['model'])
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        audio, labels = batch
        logits = self(audio)
        loss = self.criterion(logits, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        audio, labels = batch
        logits = self(audio)
        loss = self.criterion(logits, labels)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.0001)
        )
        return optimizer


def train(config_path: str):
    """Train audio model"""
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Set seed
    pl.seed_everything(42, workers=True)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_audio_dataloaders(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'],
        test_path=config['data'].get('test_path'),
        config={
            'batch_size': config['training']['batch_size'],
            'sample_rate': config['data']['sample_rate'],
            'max_duration': config['data']['max_duration'],
            'features': config['data']['features'],
            'augmentation': config['data'].get('augmentation', {}),
            'noise_filtering': config['data'].get('noise_filtering', {}),
            'pin_memory': config['data'].get('pin_memory', True)
        },
        num_workers=config['data'].get('num_workers', 4)
    )
    
    # Create model
    logger.info("Initializing model...")
    model = AudioDeepfakeModule(config)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config['checkpoint']['dirpath'],
            filename=config['checkpoint']['filename'],
            monitor=config['checkpoint']['monitor'],
            mode=config['checkpoint']['mode'],
            save_top_k=config['checkpoint']['save_top_k']
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='auto',
        devices='auto',
        precision=config['training'].get('precision', 16),
        callbacks=callbacks,
        log_every_n_steps=10
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train Audio Deepfake Detection Model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML')
    args = parser.parse_args()
    
    train(args.config)


if __name__ == '__main__':
    main()
