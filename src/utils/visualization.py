"""
Visualization Utilities
Advanced plotting and visualization for deepfake detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from pathlib import Path
import torch

from .logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def visualize_predictions(
    image: np.ndarray,
    prediction: float,
    label: str,
    confidence: float,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize prediction on image
    
    Args:
        image: Input image (RGB)
        prediction: Prediction score (0-1)
        label: Predicted label ('Real' or 'Fake')
        confidence: Confidence score
        save_path: Path to save visualization
        
    Returns:
        Annotated image
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    
    # Determine color based on prediction
    color = (0, 255, 0) if label == "Real" else (255, 0, 0)
    
    # Add text overlay
    text = f"{label}: {confidence:.2%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Get text size for background rectangle
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(
        annotated,
        (10, 10),
        (20 + text_w, 30 + text_h),
        color,
        -1
    )
    
    # Draw text
    cv2.putText(
        annotated,
        text,
        (15, 25 + text_h),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )
    
    # Draw confidence bar
    bar_width = int(w * 0.3)
    bar_height = 20
    bar_x = 10
    bar_y = 50 + text_h
    
    # Background bar
    cv2.rectangle(
        annotated,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (200, 200, 200),
        -1
    )
    
    # Confidence bar
    conf_width = int(bar_width * confidence)
    cv2.rectangle(
        annotated,
        (bar_x, bar_y),
        (bar_x + conf_width, bar_y + bar_height),
        color,
        -1
    )
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        logger.info(f"Visualization saved to: {save_path}")
    
    return annotated


def create_heatmap(
    image: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.5,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create attention/artifact heatmap overlay
    
    Args:
        image: Original image (RGB)
        attention_map: Attention weights (H x W)
        alpha: Blending factor
        save_path: Path to save heatmap
        
    Returns:
        Heatmap overlay
    """
    # Resize attention map to image size
    h, w = image.shape[:2]
    attention_resized = cv2.resize(attention_map, (w, h))
    
    # Normalize to 0-255
    attention_normalized = ((attention_resized - attention_resized.min()) / 
                           (attention_resized.max() - attention_resized.min() + 1e-8) * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(attention_normalized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        logger.info(f"Heatmap saved to: {save_path}")
    
    return overlay


def plot_temporal_features(
    features: np.ndarray,
    feature_names: List[str],
    title: str = "Temporal Features",
    save_path: Optional[str] = None
):
    """
    Plot temporal features over time
    
    Args:
        features: Feature array (time_steps x num_features)
        feature_names: Names of features
        title: Plot title
        save_path: Path to save plot
    """
    n_features = features.shape[1]
    time_steps = np.arange(features.shape[0])
    
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features))
    
    if n_features == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        ax.plot(time_steps, features[:, i], linewidth=2)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.set_title(title)
        if i == n_features - 1:
            ax.set_xlabel('Frame')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Temporal plot saved to: {save_path}")
    
    plt.close()


def plot_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    title: str = "Spectrogram",
    save_path: Optional[str] = None
):
    """
    Plot audio spectrogram
    
    Args:
        audio: Audio signal
        sr: Sample rate
        title: Plot title
        save_path: Path to save plot
    """
    import librosa
    import librosa.display
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Spectrogram saved to: {save_path}")
    
    plt.close()


def plot_lipsync_alignment(
    sync_distances: np.ndarray,
    threshold: float = 7.5,
    title: str = "Lip-Sync Alignment",
    save_path: Optional[str] = None
):
    """
    Plot lip-sync alignment curve
    
    Args:
        sync_distances: Synchronization distances over time
        threshold: Threshold for out-of-sync detection
        title: Plot title
        save_path: Path to save plot
    """
    time_steps = np.arange(len(sync_distances))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, sync_distances, linewidth=2, label='Sync Distance')
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    
    # Highlight out-of-sync regions
    out_of_sync = sync_distances > threshold
    if out_of_sync.any():
        plt.fill_between(
            time_steps,
            0,
            sync_distances.max(),
            where=out_of_sync,
            alpha=0.3,
            color='red',
            label='Out of Sync'
        )
    
    plt.xlabel('Frame')
    plt.ylabel('Sync Distance')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Lip-sync plot saved to: {save_path}")
    
    plt.close()


def draw_facial_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    connections: Optional[List[Tuple[int, int]]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 2
) -> np.ndarray:
    """
    Draw facial landmarks on image
    
    Args:
        image: Input image (RGB)
        landmarks: Landmark coordinates (N x 2)
        connections: List of landmark pairs to connect
        color: Drawing color
        radius: Point radius
        
    Returns:
        Image with landmarks drawn
    """
    annotated = image.copy()
    
    # Draw landmarks
    for (x, y) in landmarks:
        cv2.circle(annotated, (int(x), int(y)), radius, color, -1)
    
    # Draw connections
    if connections:
        for (i, j) in connections:
            pt1 = tuple(landmarks[i].astype(int))
            pt2 = tuple(landmarks[j].astype(int))
            cv2.line(annotated, pt1, pt2, color, 1)
    
    return annotated
