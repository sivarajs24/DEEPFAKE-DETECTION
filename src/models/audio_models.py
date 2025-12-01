"""
Audio Deepfake Detection Models
Supports wav2vec2, ECAPA-TDNN, and RawNet2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config
from typing import Optional, Dict

from ..utils import get_logger

logger = get_logger(__name__)


class Wav2Vec2Detector(nn.Module):
    """
    Audio deepfake detector using wav2vec2
    """
    
    def __init__(
        self,
        pretrained_model: str = "facebook/wav2vec2-base",
        num_classes: int = 2,
        freeze_feature_extractor: bool = False,
        pooling: str = "attention",
        dropout: float = 0.1
    ):
        super().__init__()
        
        logger.info(f"Initializing Wav2Vec2Detector with {pretrained_model}")
        
        # Load pretrained wav2vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model)
        
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()
            logger.info("Froze feature extractor")
        
        hidden_size = self.wav2vec2.config.hidden_size
        
        # Pooling layer
        self.pooling_type = pooling
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        logger.info(f"Model initialized with {pooling} pooling")
    
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_values: Raw audio waveform (B, T)
            
        Returns:
            Logits (B, num_classes)
        """
        # Extract features
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state  # (B, T, hidden_size)
        
        # Pooling
        if self.pooling_type == "mean":
            pooled = hidden_states.mean(dim=1)
        elif self.pooling_type == "attention":
            attention_weights = F.softmax(self.attention(hidden_states), dim=1)  # (B, T, 1)
            pooled = (hidden_states * attention_weights).sum(dim=1)  # (B, hidden_size)
        else:
            pooled = hidden_states[:, 0]  # Use first token
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


class ECAPATDNNDetector(nn.Module):
    """
    ECAPA-TDNN for audio deepfake detection
    Emphasizing Channel Attention, Propagation and Aggregation in TDNN
    """
    
    def __init__(
        self,
        input_size: int = 80,  # Mel filterbank features
        channels: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_size, channels, kernel_size=5, padding=2)
        
        # SE-Res2Blocks
        self.layer1 = SERes2Block(channels, channels, kernel_size=3, dilation=2)
        self.layer2 = SERes2Block(channels, channels, kernel_size=3, dilation=3)
        self.layer3 = SERes2Block(channels, channels, kernel_size=3, dilation=4)
        
        # Attention aggregation
        self.attention = AttentiveStatisticsPooling(channels * 3)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(channels * 3 * 2, 512),  # *2 for mean and std
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        logger.info("ECAPA-TDNN initialized")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features (B, input_size, T)
            
        Returns:
            Logits (B, num_classes)
        """
        x = F.relu(self.conv1(x))
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        # Concatenate outputs
        x = torch.cat([x1, x2, x3], dim=1)
        
        # Attention pooling
        x = self.attention(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class SERes2Block(nn.Module):
    """Squeeze-Excitation Res2Block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Squeeze-Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(out_channels // 8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # SE attention
        se_weight = self.se(x)
        x = x * se_weight
        
        return x + residual


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, channels, kernel_size=1),
            nn.Softmax(dim=2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, C, T)
            
        Returns:
            Pooled output (B, C*2)
        """
        attn_weights = self.attention(x)
        
        # Weighted mean and std
        mean = (x * attn_weights).sum(dim=2)
        std = torch.sqrt((x ** 2 * attn_weights).sum(dim=2) - mean ** 2 + 1e-8)
        
        return torch.cat([mean, std], dim=1)


class RawNet2Detector(nn.Module):
    """
    RawNet2 for end-to-end audio deepfake detection from raw waveform
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        first_conv_channels: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Sinc convolution (learnable filterbank)
        self.sinc_conv = SincConv(out_channels=first_conv_channels, kernel_size=251)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(first_conv_channels, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
        )
        
        # GRU for temporal modeling
        self.gru = nn.GRU(512, 512, num_layers=2, batch_first=True, dropout=dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        logger.info("RawNet2 initialized")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Raw waveform (B, T)
            
        Returns:
            Logits (B, num_classes)
        """
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.sinc_conv(x)
        x = F.max_pool1d(x, 3)
        x = torch.abs(x)  # Abs pooling
        
        x = self.res_blocks(x)
        
        # Transpose for GRU
        x = x.transpose(1, 2)  # (B, T, C)
        
        # GRU
        x, _ = self.gru(x)
        x = x[:, -1, :]  # Last time step
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class SincConv(nn.Module):
    """Sinc-based convolution (learnable filterbank)"""
    
    def __init__(self, out_channels: int, kernel_size: int, sample_rate: int = 16000):
        super().__init__()
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        
        # Learnable parameters for band-pass filters
        self.low_hz = nn.Parameter(torch.linspace(0, sample_rate / 2, out_channels + 1)[:-1])
        self.band_hz = nn.Parameter(torch.ones(out_channels) * 50)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sinc filters"""
        # Create sinc filters
        n = torch.arange(0, self.kernel_size).float()
        n = (n - (self.kernel_size - 1) / 2).unsqueeze(0).to(x.device)
        
        low_hz = torch.abs(self.low_hz).unsqueeze(1)
        high_hz = torch.abs(low_hz + self.band_hz.unsqueeze(1))
        
        band = (high_hz - low_hz) / self.sample_rate
        f_times_t_low = 2 * np.pi * low_hz * n / self.sample_rate
        f_times_t_high = 2 * np.pi * high_hz * n / self.sample_rate
        
        # Sinc function
        bp = (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (2 * np.pi * n + 1e-9)
        bp[:, (self.kernel_size - 1) // 2] = band.squeeze()
        
        # Hamming window
        window = 0.54 - 0.46 * torch.cos(2 * np.pi * torch.arange(self.kernel_size) / (self.kernel_size - 1))
        window = window.unsqueeze(0).to(x.device)
        
        filters = bp * window
        filters = filters.unsqueeze(1)  # (out_channels, 1, kernel_size)
        
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)


class ResidualBlock(nn.Module):
    """Residual block for RawNet2"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        return F.relu(x + residual)


# Fix import
import numpy as np


def create_audio_model(config: Dict) -> nn.Module:
    """
    Factory function to create audio model from config
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    architecture = config.get('architecture', 'wav2vec2')
    
    if architecture == 'wav2vec2':
        model = Wav2Vec2Detector(
            pretrained_model=config.get('pretrained_model', 'facebook/wav2vec2-base'),
            num_classes=config.get('num_classes', 2),
            freeze_feature_extractor=config.get('freeze_feature_extractor', False),
            pooling=config.get('pooling', 'attention'),
            dropout=config.get('dropout', 0.1)
        )
    elif architecture == 'ecapa_tdnn':
        model = ECAPATDNNDetector(
            input_size=config.get('input_size', 80),
            channels=config.get('channels', 512),
            num_classes=config.get('num_classes', 2),
            dropout=config.get('dropout', 0.1)
        )
    elif architecture == 'rawnet2':
        model = RawNet2Detector(
            num_classes=config.get('num_classes', 2),
            first_conv_channels=config.get('first_conv_channels', 128),
            dropout=config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    logger.info(f"Created {architecture} model")
    return model
