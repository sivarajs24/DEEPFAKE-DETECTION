"""
Lip-Sync Detection Module (SyncNet-style)
Twin network architecture for audio-visual synchronization detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple, Optional

from ..utils import get_logger

logger = get_logger(__name__)


class SyncNetModel(nn.Module):
    """
    SyncNet for lip-sync detection
    Twin network: video encoder + audio encoder
    """
    
    def __init__(
        self,
        video_encoder_type: str = "resnet18",
        video_output_dim: int = 512,
        audio_input_features: int = 13,  # MFCC features
        audio_output_dim: int = 512,
        fusion_type: str = "contrastive"
    ):
        super().__init__()
        
        logger.info("Initializing SyncNet model")
        
        # Video encoder (processes mouth ROI)
        if video_encoder_type == "resnet18":
            resnet = models.resnet18(pretrained=True)
            self.video_encoder = nn.Sequential(
                *list(resnet.children())[:-1],  # Remove final FC layer
                nn.Flatten(),
                nn.Linear(512, video_output_dim),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unknown video encoder: {video_encoder_type}")
        
        # Audio encoder (processes MFCC features)
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(audio_input_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            
            nn.Flatten(),
            nn.Linear(256, audio_output_dim),
            nn.ReLU()
        )
        
        self.fusion_type = fusion_type
        
        # L2 normalization for embeddings
        self.normalize = nn.functional.normalize
        
        logger.info(f"SyncNet initialized with {fusion_type} fusion")
    
    def forward_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video (mouth ROI)
        
        Args:
            video: Mouth ROI images (B, T, C, H, W)
            
        Returns:
            Video embeddings (B, T, video_output_dim)
        """
        B, T, C, H, W = video.shape
        
        # Reshape for batch processing
        video = video.view(B * T, C, H, W)
        
        # Encode
        embeddings = self.video_encoder(video)
        
        # Reshape back
        embeddings = embeddings.view(B, T, -1)
        
        # L2 normalize
        embeddings = self.normalize(embeddings, p=2, dim=2)
        
        return embeddings
    
    def forward_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio (MFCC features)
        
        Args:
            audio: MFCC features (B, T, n_mfcc, audio_frames)
            
        Returns:
            Audio embeddings (B, T, audio_output_dim)
        """
        B, T, n_mfcc, audio_frames = audio.shape
        
        # Reshape for batch processing
        audio = audio.view(B * T, n_mfcc, audio_frames)
        
        # Encode
        embeddings = self.audio_encoder(audio)
        
        # Reshape back
        embeddings = embeddings.view(B, T, -1)
        
        # L2 normalize
        embeddings = self.normalize(embeddings, p=2, dim=2)
        
        return embeddings
    
    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        compute_distance: bool = True
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            video: Mouth ROI images (B, T, C, H, W)
            audio: MFCC features (B, T, n_mfcc, audio_frames)
            compute_distance: Whether to compute sync distance
            
        Returns:
            Sync distances (B, T) if compute_distance, else (video_emb, audio_emb)
        """
        video_emb = self.forward_video(video)
        audio_emb = self.forward_audio(audio)
        
        if compute_distance:
            # Euclidean distance
            distance = torch.sqrt(torch.sum((video_emb - audio_emb) ** 2, dim=2))
            return distance
        else:
            return video_emb, audio_emb


class SyncNetLoss(nn.Module):
    """
    Contrastive loss for SyncNet training
    """
    
    def __init__(self, margin: float = 0.5, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        video_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            video_emb: Video embeddings (B, T, D)
            audio_emb: Audio embeddings (B, T, D)
            labels: Binary labels (B, T) - 1 for in-sync, 0 for out-of-sync
            
        Returns:
            Loss scalar
        """
        # Cosine similarity
        similarity = F.cosine_similarity(video_emb, audio_emb, dim=2)
        
        # Contrastive loss
        positive_loss = labels * (1 - similarity)
        negative_loss = (1 - labels) * F.relu(similarity - self.margin)
        
        loss = (positive_loss + negative_loss).mean()
        
        return loss


def create_syncnet_model(config: Dict) -> nn.Module:
    """
    Factory function to create SyncNet model from config
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    video_config = config.get('video_encoder', {})
    audio_config = config.get('audio_encoder', {})
    
    model = SyncNetModel(
        video_encoder_type=video_config.get('type', 'resnet18'),
        video_output_dim=video_config.get('output_dim', 512),
        audio_input_features=audio_config.get('input_features', 13),
        audio_output_dim=audio_config.get('output_dim', 512),
        fusion_type=config.get('fusion', {}).get('type', 'contrastive')
    )
    
    return model


from typing import Dict
