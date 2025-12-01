"""
PyTorch Lightning Module for Video Deepfake Detection
Production-grade training with mixed precision, logging, and callbacks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR
)
from typing import Dict, Optional, Any
import wandb

from ..models.video_models import create_video_model
from ..utils import MetricsCalculator, get_logger

logger = get_logger(__name__)


class VideoDeepfakeModule(pl.LightningModule):
    """
    Lightning module for video deepfake detection training
    Includes mixed precision, logging, and comprehensive metrics
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Complete configuration dictionary
        """
        super().__init__()
        
        self.save_hyperparameters(config)
        self.config = config
        
        # Create model
        self.model = create_video_model(config['model'])
        
        # Loss function with class weights (handle imbalanced data)
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics calculators
        self.train_metrics = MetricsCalculator()
        self.val_metrics = MetricsCalculator()
        self.test_metrics = MetricsCalculator()
        
        logger.info("VideoDeepfakeModule initialized")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Calculate predictions
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Update metrics
        self.train_metrics.update(preds, labels, probs)
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Compute and log training metrics at epoch end"""
        metrics = self.train_metrics.compute()
        
        for name, value in metrics.items():
            self.log(f'train_{name}', value, prog_bar=True)
        
        logger.info(f"Train Epoch {self.current_epoch} - " +
                   f"Loss: {self.trainer.callback_metrics.get('train_loss_epoch', 0):.4f}, " +
                   f"Acc: {metrics.get('accuracy', 0):.4f}, " +
                   f"AUC: {metrics.get('auc', 0):.4f}")
        
        # Reset metrics
        self.train_metrics.reset()
    
    def validation_step(self, batch: tuple, batch_idx: int):
        """
        Validation step
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index
        """
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Calculate predictions
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Update metrics
        self.val_metrics.update(preds, labels, probs)
        
        # Log loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self):
        """Compute and log validation metrics at epoch end"""
        metrics = self.val_metrics.compute()
        
        for name, value in metrics.items():
            self.log(f'val_{name}', value, prog_bar=True)
        
        logger.info(f"Val Epoch {self.current_epoch} - " +
                   f"Loss: {self.trainer.callback_metrics.get('val_loss', 0):.4f}, " +
                   f"Acc: {metrics.get('accuracy', 0):.4f}, " +
                   f"AUC: {metrics.get('auc', 0):.4f}")
        
        # Log confusion matrix to wandb
        if self.logger and hasattr(self.logger, 'experiment'):
            cm = self.val_metrics.get_confusion_matrix()
            if cm.size > 0:
                try:
                    if isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
                        self.logger.experiment.log({
                            "val_confusion_matrix": wandb.plot.confusion_matrix(
                                probs=None,
                                y_true=self.val_metrics.targets,
                                preds=self.val_metrics.predictions,
                                class_names=['Real', 'Fake']
                            )
                        })
                except Exception as e:
                    logger.warning(f"Could not log confusion matrix: {e}")
        
        # Reset metrics
        self.val_metrics.reset()
    
    def test_step(self, batch: tuple, batch_idx: int):
        """
        Test step
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index
        """
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Calculate predictions
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Update metrics
        self.test_metrics.update(preds, labels, probs)
        
        # Log loss
        self.log('test_loss', loss, on_epoch=True)
    
    def on_test_epoch_end(self):
        """Compute and log test metrics at epoch end"""
        metrics = self.test_metrics.compute()
        
        for name, value in metrics.items():
            self.log(f'test_{name}', value)
        
        logger.info(f"Test Results - " +
                   f"Loss: {self.trainer.callback_metrics.get('test_loss', 0):.4f}, " +
                   f"Acc: {metrics.get('accuracy', 0):.4f}, " +
                   f"AUC: {metrics.get('auc', 0):.4f}")
        
        # Print classification report
        report = self.test_metrics.get_classification_report()
        logger.info(f"\nClassification Report:\n{report}")
        
        # Reset metrics
        self.test_metrics.reset()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        config = self.config['training']
        
        # Optimizer
        optimizer_name = config.get('optimizer', 'adamw').lower()
        lr = config.get('learning_rate', 0.0001)
        weight_decay = config.get('weight_decay', 0.0001)
        
        if optimizer_name == 'adam':
            optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        logger.info(f"Using {optimizer_name} optimizer with lr={lr}")
        
        # Learning rate scheduler
        if 'scheduler' not in config:
            return {'optimizer': optimizer}
        
        scheduler_config = config['scheduler']
        scheduler_type = scheduler_config.get('type', 'reduce_lr_on_plateau')
        
        if scheduler_type == 'reduce_lr_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get('mode', 'min'),
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 3),
                min_lr=scheduler_config.get('min_lr', 1e-7)
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        elif scheduler_type == 'cosine_annealing_warm_restarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=scheduler_config.get('t_0', 10),
                T_mult=scheduler_config.get('t_mult', 2),
                eta_min=scheduler_config.get('eta_min', 1e-7)
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        elif scheduler_type == 'step_lr':
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, using no scheduler")
            return {'optimizer': optimizer}
