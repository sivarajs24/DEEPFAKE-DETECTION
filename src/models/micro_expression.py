"""
Micro-Expression Analysis Module
MediaPipe FaceMesh + Optical Flow for facial behavior analysis
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple

from ..utils import get_logger

logger = get_logger(__name__)


class MicroExpressionExtractor:
    """
    Extract micro-expression features using MediaPipe and Optical Flow
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize MediaPipe FaceMesh
        face_mesh_config = config.get('face_mesh', {})
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=face_mesh_config.get('static_image_mode', False),
            max_num_faces=face_mesh_config.get('max_num_faces', 1),
            refine_landmarks=face_mesh_config.get('refine_landmarks', True),
            min_detection_confidence=face_mesh_config.get('min_detection_confidence', 0.5),
            min_tracking_confidence=face_mesh_config.get('min_tracking_confidence', 0.5)
        )
        
        # Optical flow config
        self.optical_flow_config = config.get('optical_flow', {})
        
        # Previous frame for optical flow
        self.prev_gray = None
        self.prev_landmarks = None
        
        logger.info("MicroExpressionExtractor initialized")
    
    def extract_blink_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract blink-related features (Eye Aspect Ratio)
        
        Args:
            landmarks: Facial landmarks (468 x 3)
            
        Returns:
            Dictionary of blink features
        """
        # Eye landmark indices (MediaPipe)
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        def eye_aspect_ratio(eye_points):
            # Vertical distances
            v1 = np.linalg.norm(eye_points[1] - eye_points[5])
            v2 = np.linalg.norm(eye_points[2] - eye_points[4])
            # Horizontal distance
            h = np.linalg.norm(eye_points[0] - eye_points[3])
            
            ear = (v1 + v2) / (2.0 * h + 1e-6)
            return ear
        
        left_eye_points = landmarks[LEFT_EYE, :2]
        right_eye_points = landmarks[RIGHT_EYE, :2]
        
        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        return {
            'left_eye_ear': left_ear,
            'right_eye_ear': right_ear,
            'avg_ear': avg_ear,
            'is_blink': avg_ear < 0.21  # Threshold
        }
    
    def extract_eyebrow_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract eyebrow movement features
        
        Args:
            landmarks: Facial landmarks (468 x 3)
            
        Returns:
            Dictionary of eyebrow features
        """
        # Eyebrow landmark indices
        LEFT_EYEBROW = [70, 63, 105, 66, 107]
        RIGHT_EYEBROW = [336, 296, 334, 293, 300]
        
        left_eyebrow = landmarks[LEFT_EYEBROW, :2]
        right_eyebrow = landmarks[RIGHT_EYEBROW, :2]
        
        # Eyebrow height (distance from eye)
        left_eye_center = landmarks[[33, 133], :2].mean(axis=0)
        right_eye_center = landmarks[[362, 263], :2].mean(axis=0)
        
        left_eyebrow_height = left_eyebrow[:, 1].mean() - left_eye_center[1]
        right_eyebrow_height = right_eyebrow[:, 1].mean() - right_eye_center[1]
        
        return {
            'left_eyebrow_height': left_eyebrow_height,
            'right_eyebrow_height': right_eyebrow_height,
            'avg_eyebrow_height': (left_eyebrow_height + right_eyebrow_height) / 2
        }
    
    def extract_mouth_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract mouth movement features
        
        Args:
            landmarks: Facial landmarks (468 x 3)
            
        Returns:
            Dictionary of mouth features
        """
        # Mouth landmark indices
        MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        
        mouth_points = landmarks[MOUTH_OUTER, :2]
        
        # Mouth aspect ratio
        vertical = np.linalg.norm(mouth_points[3] - mouth_points[9])
        horizontal = np.linalg.norm(mouth_points[0] - mouth_points[6])
        
        mouth_ar = vertical / (horizontal + 1e-6)
        
        # Mouth area
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(mouth_points)
            mouth_area = hull.volume  # 2D area
        except:
            mouth_area = 0.0
        
        return {
            'mouth_aspect_ratio': mouth_ar,
            'mouth_area': mouth_area
        }
    
    def compute_optical_flow(
        self,
        current_frame: np.ndarray,
        current_landmarks: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute optical flow between frames
        
        Args:
            current_frame: Current grayscale frame
            current_landmarks: Current landmarks
            
        Returns:
            Dictionary of optical flow features
        """
        if self.prev_gray is None or self.prev_landmarks is None:
            self.prev_gray = current_frame
            self.prev_landmarks = current_landmarks
            return {'flow_magnitude': 0.0, 'flow_angle': 0.0}
        
        # Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            current_frame,
            None,
            pyr_scale=self.optical_flow_config.get('pyr_scale', 0.5),
            levels=self.optical_flow_config.get('levels', 3),
            winsize=self.optical_flow_config.get('winsize', 15),
            iterations=self.optical_flow_config.get('iterations', 3),
            poly_n=self.optical_flow_config.get('poly_n', 5),
            poly_sigma=self.optical_flow_config.get('poly_sigma', 1.2),
            flags=0
        )
        
        # Compute flow at landmark points
        landmark_coords = current_landmarks[:, :2].astype(int)
        
        # Ensure coordinates are within bounds
        h, w = current_frame.shape
        landmark_coords[:, 0] = np.clip(landmark_coords[:, 0], 0, w - 1)
        landmark_coords[:, 1] = np.clip(landmark_coords[:, 1], 0, h - 1)
        
        flows = flow[landmark_coords[:, 1], landmark_coords[:, 0]]
        
        # Flow magnitude and angle
        magnitude = np.linalg.norm(flows, axis=1).mean()
        angle = np.arctan2(flows[:, 1], flows[:, 0]).mean()
        
        # Update previous
        self.prev_gray = current_frame
        self.prev_landmarks = current_landmarks
        
        return {
            'flow_magnitude': magnitude,
            'flow_angle': angle
        }
    
    def extract_features(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Extract all micro-expression features from a frame
        
        Args:
            frame: RGB frame
            
        Returns:
            Dictionary of all features or None if face not detected
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
        
        # Detect face landmarks
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array
        h, w = frame.shape[:2]
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z]
            for lm in face_landmarks.landmark
        ])
        
        # Extract features
        features = {}
        
        # Blink features
        blink_features = self.extract_blink_features(landmarks)
        features.update(blink_features)
        
        # Eyebrow features
        eyebrow_features = self.extract_eyebrow_features(landmarks)
        features.update(eyebrow_features)
        
        # Mouth features
        mouth_features = self.extract_mouth_features(landmarks)
        features.update(mouth_features)
        
        # Optical flow features
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        flow_features = self.compute_optical_flow(gray_frame, landmarks)
        features.update(flow_features)
        
        return features


class MicroExpressionLSTM(nn.Module):
    """
    LSTM model for temporal micro-expression analysis
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        logger.info(f"MicroExpressionLSTM initialized with {num_layers} layers")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features (B, T, input_dim)
            
        Returns:
            Logits (B, num_classes)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use last time step
        last_out = lstm_out[:, -1, :]
        
        # Classification
        logits = self.classifier(last_out)
        
        return logits
