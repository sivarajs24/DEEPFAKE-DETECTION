"""DeepGuard-X: Production-Grade Multi-Modal Deepfake Detection System"""

__version__ = "1.0.0"
__author__ = "DeepGuard-X Team"
__license__ = "MIT"

from .utils import get_logger
from .inference.pipeline import DeepGuardXInference

__all__ = [
    'get_logger',
    'DeepGuardXInference',
]
