"""
Metrics and Evaluation Utilities
Production-grade metrics for deepfake detection
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    """
    Calculate comprehensive metrics for binary classification
    """
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and targets"""
        self.predictions: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.probabilities: List[np.ndarray] = []
    
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probs: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with new batch
        
        Args:
            preds: Predicted class labels
            targets: Ground truth labels
            probs: Prediction probabilities (optional)
        """
        self.predictions.append(preds.cpu().numpy())
        self.targets.append(targets.cpu().numpy())
        
        if probs is not None:
            self.probabilities.append(probs.detach().float().cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary of metric names and values
        """
        if not self.predictions:
            logger.warning("No predictions to compute metrics")
            return {}
        
        # Concatenate all batches
        y_pred = np.concatenate(self.predictions)
        y_true = np.concatenate(self.targets)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        }
        
        # Add AUC if probabilities are available
        if self.probabilities:
            y_prob = np.concatenate(self.probabilities)
            
            # For binary classification, use positive class probability
            if y_prob.ndim > 1 and y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]
            
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except ValueError as e:
                logger.warning(f"Could not compute AUC: {e}")
                metrics['auc'] = 0.0
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix
        
        Returns:
            Confusion matrix as numpy array
        """
        if not self.predictions:
            return np.array([])
        
        y_pred = np.concatenate(self.predictions)
        y_true = np.concatenate(self.targets)
        
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self) -> str:
        """
        Get detailed classification report
        
        Returns:
            Classification report string
        """
        if not self.predictions:
            return "No predictions available"
        
        y_pred = np.concatenate(self.predictions)
        y_true = np.concatenate(self.targets)
        
        return classification_report(
            y_true, y_pred,
            target_names=['Real', 'Fake'],
            digits=4
        )
    
    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        normalize: bool = False
    ):
        """
        Plot confusion matrix
        
        Args:
            save_path: Path to save plot (optional)
            normalize: Whether to normalize the matrix
        """
        cm = self.get_confusion_matrix()
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, save_path: Optional[str] = None):
        """
        Plot ROC curve
        
        Args:
            save_path: Path to save plot (optional)
        """
        if not self.probabilities:
            logger.warning("Probabilities not available for ROC curve")
            return
        
        y_true = np.concatenate(self.targets)
        y_prob = np.concatenate(self.probabilities)
        
        # For binary classification
        if y_prob.ndim > 1 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to: {save_path}")
        
        plt.close()


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    logger.info(f"Optimal threshold for {metric}: {best_threshold:.3f} (score: {best_score:.4f})")
    return best_threshold, best_score
