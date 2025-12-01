"""
Video Deepfake Detection Models
Supports EfficientNet, ViT, and Xception architectures
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Dict
from torchvision import models

from ..utils import get_logger

logger = get_logger(__name__)


class VideoDeepfakeModel(nn.Module):
    """
    Base model for video deepfake detection
    Supports multiple backbone architectures
    """
    
    def __init__(
        self,
        architecture: str = "efficientnet_b3",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Args:
            architecture: Model architecture name
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        super().__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        
        logger.info(f"Initializing {architecture} model (pretrained={pretrained})")
        
        # Load backbone using timm (supports EfficientNet, ViT, etc.)
        if architecture in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
                           'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                           'efficientnet_b6', 'efficientnet_b7']:
            self.backbone = timm.create_model(
                architecture,
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                global_pool='avg'
            )
            in_features = self.backbone.num_features
            
        elif 'vit' in architecture.lower():
            # Vision Transformer
            self.backbone = timm.create_model(
                architecture,
                pretrained=pretrained,
                num_classes=0,
                global_pool='token'
            )
            in_features = self.backbone.num_features
            
        elif architecture == 'xception':
            # Xception (popular for deepfake detection)
            self.backbone = timm.create_model(
                'xception',
                pretrained=pretrained,
                num_classes=0,
                global_pool='avg'
            )
            in_features = self.backbone.num_features
            
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        logger.info(f"Model initialized with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Logits (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Feature tensor (B, feature_dim)
        """
        return self.backbone(x)


class EfficientNetB3Detector(VideoDeepfakeModel):
    """EfficientNet-B3 specialized for deepfake detection"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__(
            architecture='efficientnet_b3',
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )


class ViTDetector(VideoDeepfakeModel):
    """Vision Transformer specialized for deepfake detection"""
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        attention_dropout: float = 0.1
    ):
        super().__init__(
            architecture='vit_base_patch16_224',
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
        # Note: attention_dropout would need to be set in the backbone config


class XceptionDetector(VideoDeepfakeModel):
    """Xception specialized for deepfake detection"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__(
            architecture='xception',
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )


def create_video_model(config: Dict) -> nn.Module:
    """
    Factory function to create video model from config
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    architecture = config.get('architecture', 'efficientnet_b3')
    num_classes = config.get('num_classes', 2)
    pretrained = config.get('pretrained', True)
    dropout = config.get('dropout', 0.3)
    
    model = VideoDeepfakeModel(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    
    return model


class GradCAMExtractor:
    """
    Grad-CAM for visualizing model attention
    Useful for explaining predictions
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        """
        Args:
            model: The model to extract Grad-CAM from
            target_layer: Name of target layer (e.g., 'backbone.blocks.6')
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Get target layer
        target = dict(self.model.named_modules())[self.target_layer]
        target.register_forward_hook(forward_hook)
        target.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for CAM (if None, uses predicted class)
            
        Returns:
            CAM heatmap (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Compute CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam
