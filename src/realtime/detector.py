"""
Real-Time Detection Pipeline
Webcam capture, frame batching, async inference with ONNX
"""

import cv2
import torch
import numpy as np
import onnxruntime as ort
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Any, Callable
import time

from ..utils import get_logger

logger = get_logger(__name__)


class RealtimeDetector:
    """
    Real-time deepfake detection from webcam or video stream
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Real-time configuration
        """
        self.config = config
        self.running = False
        
        # Input settings
        input_config = config['realtime']['input']
        self.source = input_config['source']
        self.webcam_id = input_config.get('webcam_id', 0)
        self.resolution = tuple(input_config.get('resolution', [640, 480]))
        self.target_fps = input_config.get('fps', 30)
        
        # Processing settings
        proc_config = config['realtime']['processing']
        self.frame_skip = proc_config.get('frame_skip', 2)
        self.batch_size = proc_config.get('batch_size', 8)
        self.buffer_size = proc_config.get('buffer_size', 30)
        
        # Async settings
        async_config = config['realtime']['async']
        self.async_enabled = async_config.get('enabled', True)
        self.max_workers = async_config.get('max_workers', 4)
        
        # Queues
        self.frame_queue = Queue(maxsize=self.buffer_size)
        self.result_queue = Queue(maxsize=self.buffer_size)
        
        # Thread pool
        if self.async_enabled:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # ONNX models
        self.models = {}
        self._load_models()
        
        # Display settings
        self.display_config = config['realtime']['display']
        
        # Alert settings
        self.alert_config = config['realtime']['alerts']
        self.alert_threshold = self.alert_config.get('threshold', 0.7)
        
        logger.info("RealtimeDetector initialized")
    
    def _load_models(self):
        """Load ONNX models for inference"""
        models_config = self.config['realtime']['models']
        onnx_config = self.config['realtime']['onnx']
        
        providers = onnx_config.get('providers', ['CPUExecutionProvider'])
        
        for model_name, model_cfg in models_config.items():
            if not model_cfg.get('enabled', False):
                continue
            
            onnx_path = model_cfg.get('onnx_path')
            if not onnx_path:
                logger.warning(f"No ONNX path for {model_name}")
                continue
            
            try:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = onnx_config.get('intra_op_num_threads', 4)
                sess_options.inter_op_num_threads = onnx_config.get('inter_op_num_threads', 4)
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                session = ort.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=providers
                )
                
                self.models[model_name] = {
                    'session': session,
                    'interval': model_cfg.get('inference_interval', 1),
                    'counter': 0
                }
                
                logger.info(f"Loaded ONNX model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
    
    def start_webcam_detection(self):
        """Start real-time detection from webcam"""
        logger.info("Starting webcam detection...")
        
        # Open webcam
        cap = cv2.VideoCapture(self.webcam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            return
        
        self.running = True
        
        # Start inference thread
        if self.async_enabled:
            inference_thread = Thread(target=self._inference_worker, daemon=True)
            inference_thread.start()
        
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        try:
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
                
                frame_count += 1
                
                # Calculate FPS
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                
                # Process frame
                if frame_count % self.frame_skip == 0:
                    if self.async_enabled:
                        # Add to queue for async processing
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame.copy())
                    else:
                        # Synchronous processing
                        result = self._process_frame(frame)
                        self._display_result(frame, result, fps)
                
                # Get async results
                if self.async_enabled and not self.result_queue.empty():
                    result = self.result_queue.get()
                    self._display_result(frame, result, fps)
                else:
                    # Display frame without result
                    if self.display_config.get('show_video', True):
                        display_frame = self._annotate_frame(frame, None, fps)
                        cv2.imshow('DeepGuard-X Real-Time Detection', display_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break
                
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Stopped webcam detection")
    
    def _inference_worker(self):
        """Worker thread for async inference"""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                result = self._process_frame(frame)
                self.result_queue.put(result)
            else:
                time.sleep(0.01)  # Avoid busy waiting
    
    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame through all models
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Dictionary of predictions
        """
        predictions = {}
        
        # Video model
        if 'video' in self.models:
            model_info = self.models['video']
            model_info['counter'] += 1
            
            if model_info['counter'] >= model_info['interval']:
                model_info['counter'] = 0
                
                try:
                    # Preprocess
                    input_tensor = self._preprocess_video(frame)
                    
                    # Inference
                    outputs = model_info['session'].run(None, {'input': input_tensor})
                    logits = outputs[0][0]
                    
                    # Softmax
                    probs = np.exp(logits) / np.sum(np.exp(logits))
                    predictions['video'] = float(probs[1])  # Fake probability
                    
                except Exception as e:
                    logger.error(f"Video inference error: {e}")
                    predictions['video'] = 0.0
        
        # Audio model (would need audio input)
        # For real-time, audio would be captured separately
        
        # Compute ensemble score
        if predictions:
            ensemble_score = np.mean(list(predictions.values()))
        else:
            ensemble_score = 0.0
        
        result = {
            'predictions': predictions,
            'ensemble_score': ensemble_score,
            'label': 'FAKE' if ensemble_score > 0.5 else 'REAL',
            'timestamp': time.time()
        }
        
        # Check for alert
        if self.alert_config.get('enabled', True):
            if ensemble_score > self.alert_threshold:
                self._trigger_alert(result)
        
        return result
    
    def _preprocess_video(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for video model
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Preprocessed tensor
        """
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        resized = cv2.resize(rgb, (224, 224))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # Transpose to CHW
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        tensor = np.expand_dims(transposed, axis=0)
        
        return tensor
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        result: Optional[Dict],
        fps: float
    ) -> np.ndarray:
        """
        Annotate frame with predictions
        
        Args:
            frame: Input frame
            result: Detection result
            fps: Current FPS
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Show FPS
        if self.display_config.get('show_fps', True):
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        # Show predictions
        if result and self.display_config.get('show_predictions', True):
            label = result['label']
            score = result['ensemble_score']
            color = (0, 0, 255) if label == 'FAKE' else (0, 255, 0)
            
            cv2.putText(
                annotated,
                f"{label}: {score:.2%}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2
            )
            
            # Draw confidence bar
            if self.display_config.get('show_scores', True):
                bar_width = 200
                bar_height = 20
                bar_x = 10
                bar_y = 90
                
                # Background
                cv2.rectangle(
                    annotated,
                    (bar_x, bar_y),
                    (bar_x + bar_width, bar_y + bar_height),
                    (200, 200, 200),
                    -1
                )
                
                # Score bar
                score_width = int(bar_width * score)
                cv2.rectangle(
                    annotated,
                    (bar_x, bar_y),
                    (bar_x + score_width, bar_y + bar_height),
                    color,
                    -1
                )
        
        return annotated
    
    def _display_result(
        self,
        frame: np.ndarray,
        result: Dict,
        fps: float
    ):
        """Display frame with result"""
        if self.display_config.get('show_video', True):
            display_frame = self._annotate_frame(frame, result, fps)
            cv2.imshow('DeepGuard-X Real-Time Detection', display_frame)
    
    def _trigger_alert(self, result: Dict):
        """Trigger alert for deepfake detection"""
        if self.alert_config.get('visual_alert', True):
            logger.warning(f"ALERT: Deepfake detected! Score: {result['ensemble_score']:.2%}")
        
        if self.alert_config.get('log_alert', True):
            logger.warning(f"Detection alert: {result}")
    
    def stop(self):
        """Stop detection"""
        self.running = False


def create_realtime_detector(config_path: str) -> RealtimeDetector:
    """
    Factory function to create realtime detector
    
    Args:
        config_path: Path to configuration
        
    Returns:
        Initialized RealtimeDetector
    """
    from ..utils import load_config
    
    config = load_config(config_path)
    detector = RealtimeDetector(config)
    
    return detector
