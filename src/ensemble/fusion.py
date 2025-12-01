"""
Ensemble Fusion Model
Combines predictions from all detection modules
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from typing import Dict, List, Optional, Any
import joblib
from pathlib import Path
import json

from ..utils import get_logger

logger = get_logger(__name__)


class EnsembleFusion:
    """
    Ensemble fusion for combining multiple deepfake detectors
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Ensemble configuration
        """
        self.config = config
        self.method = config['ensemble']['method']
        self.weights = config['ensemble'].get('weights', {})
        self.models = config['ensemble']['models']
        
        # Meta-learner for fusion
        self.meta_learner = None
        
        logger.info(f"EnsembleFusion initialized with method: {self.method}")
    
    def weighted_average(self, predictions: Dict[str, float]) -> float:
        """
        Weighted average of model predictions
        
        Args:
            predictions: Dictionary of {model_name: prediction_score}
            
        Returns:
            Ensemble score
        """
        total_score = 0.0
        total_weight = 0.0
        
        for model_name, score in predictions.items():
            weight = self.weights.get(model_name, 1.0)
            total_score += score * weight
            total_weight += weight
        
        ensemble_score = total_score / (total_weight + 1e-8)
        return ensemble_score
    
    def train_meta_learner(
        self,
        train_predictions: np.ndarray,
        train_labels: np.ndarray,
        val_predictions: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None
    ):
        """
        Train meta-learner for ensemble fusion
        
        Args:
            train_predictions: Training predictions from all models (N, num_models)
            train_labels: Training labels (N,)
            val_predictions: Validation predictions (optional)
            val_labels: Validation labels (optional)
        """
        logger.info(f"Training meta-learner: {self.method}")
        
        if self.method == 'logistic_regression':
            self.meta_learner = LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                class_weight='balanced',
                random_state=42
            )
        elif self.method == 'xgboost':
            from xgboost import XGBClassifier
            xgb_params = self.config['ensemble'].get('xgb_params', {})
            self.meta_learner = XGBClassifier(
                max_depth=xgb_params.get('max_depth', 5),
                learning_rate=xgb_params.get('learning_rate', 0.1),
                n_estimators=xgb_params.get('n_estimators', 100),
                objective='binary:logistic',
                random_state=42
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Train
        self.meta_learner.fit(train_predictions, train_labels)
        
        # Validate
        if val_predictions is not None and val_labels is not None:
            val_score = self.meta_learner.score(val_predictions, val_labels)
            logger.info(f"Meta-learner validation accuracy: {val_score:.4f}")
    
    def predict(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        Make ensemble prediction
        
        Args:
            predictions: Dictionary of {model_name: prediction_score}
            
        Returns:
            Dictionary with final score and individual scores
        """
        if self.method == 'weighted_average':
            final_score = self.weighted_average(predictions)
            
        elif self.method in ['logistic_regression', 'xgboost']:
            if self.meta_learner is None:
                raise ValueError("Meta-learner not trained")
            
            # Prepare input
            model_order = ['video', 'audio', 'lipsync', 'micro_expression', 'behavior']
            input_scores = np.array([
                predictions.get(m, 0.0) for m in model_order
            ]).reshape(1, -1)
            
            # Predict probability
            final_score = self.meta_learner.predict_proba(input_scores)[0, 1]
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Prepare result
        result = {
            'final_score': float(final_score),
            'final_label': 'FAKE' if final_score > 0.5 else 'REAL',
            'confidence': float(max(final_score, 1 - final_score)),
            'individual_scores': predictions,
            'threshold': self.config['inference'].get('threshold', 0.5)
        }
        
        return result
    
    def predict_with_uncertainty(
        self,
        predictions: Dict[str, float],
        uncertainties: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Make prediction with uncertainty quantification
        
        Args:
            predictions: Dictionary of {model_name: prediction_score}
            uncertainties: Dictionary of {model_name: uncertainty_score}
            
        Returns:
            Dictionary with final score, confidence intervals, and explanations
        """
        result = self.predict(predictions)
        
        if uncertainties:
            # Weighted uncertainty
            total_uncertainty = 0.0
            for model_name, uncertainty in uncertainties.items():
                weight = self.weights.get(model_name, 1.0)
                total_uncertainty += uncertainty * weight
            
            total_uncertainty /= sum(self.weights.values())
            
            # Confidence interval (simplified)
            result['uncertainty'] = float(total_uncertainty)
            result['confidence_interval'] = [
                max(0.0, result['final_score'] - 1.96 * total_uncertainty),
                min(1.0, result['final_score'] + 1.96 * total_uncertainty)
            ]
        
        # Add explanations
        if self.config['inference']['output'].get('include_explanations', True):
            result['explanation'] = self._generate_explanation(predictions, result)
        
        return result
    
    def _generate_explanation(
        self,
        predictions: Dict[str, float],
        result: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable explanation
        
        Args:
            predictions: Individual model predictions
            result: Final result dictionary
            
        Returns:
            Explanation string
        """
        label = result['final_label']
        score = result['final_score']
        
        # Find most influential model
        sorted_preds = sorted(predictions.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
        
        explanation = f"Prediction: {label} (confidence: {result['confidence']:.2%})\n"
        explanation += f"Final Score: {score:.4f}\n\n"
        explanation += "Individual Model Contributions:\n"
        
        for model_name, pred_score in sorted_preds:
            model_label = 'FAKE' if pred_score > 0.5 else 'REAL'
            explanation += f"  - {model_name}: {model_label} ({pred_score:.4f})\n"
        
        return explanation
    
    def save(self, save_path: str):
        """
        Save ensemble model
        
        Args:
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.meta_learner:
            joblib.dump(self.meta_learner, save_path)
            logger.info(f"Ensemble model saved to: {save_path}")
        else:
            logger.warning("No meta-learner to save")
    
    def load(self, load_path: str):
        """
        Load ensemble model
        
        Args:
            load_path: Path to load model from
        """
        load_path = Path(load_path)
        
        if load_path.exists():
            self.meta_learner = joblib.load(load_path)
            logger.info(f"Ensemble model loaded from: {load_path}")
        else:
            raise FileNotFoundError(f"Model not found: {load_path}")


class TemporalSmoother:
    """
    Temporal smoothing for video predictions
    """
    
    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: Size of smoothing window
        """
        self.window_size = window_size
        self.history: List[float] = []
    
    def smooth(self, score: float) -> float:
        """
        Apply temporal smoothing
        
        Args:
            score: Current prediction score
            
        Returns:
            Smoothed score
        """
        self.history.append(score)
        
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Moving average
        smoothed = np.mean(self.history)
        
        return smoothed
    
    def reset(self):
        """Reset history"""
        self.history = []


def create_ensemble_fusion(config_path: str) -> EnsembleFusion:
    """
    Factory function to create ensemble fusion from config
    
    Args:
        config_path: Path to ensemble configuration
        
    Returns:
        Initialized EnsembleFusion
    """
    from ..utils import load_config
    
    config = load_config(config_path)
    ensemble = EnsembleFusion(config)
    
    # Load trained meta-learner if exists
    model_path = config['ensemble'].get('ensemble_model_path')
    if model_path and Path(model_path).exists():
        ensemble.load(model_path)
    
    return ensemble
