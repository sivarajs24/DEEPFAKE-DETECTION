"""
Real-Time Detection Demo
Run webcam deepfake detection
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.realtime import create_realtime_detector
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepGuard-X Real-Time Detection')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/realtime_config.yaml',
        help='Path to real-time configuration'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting DeepGuard-X Real-Time Detection")
    logger.info("Press 'q' or ESC to quit")
    
    # Create detector
    detector = create_realtime_detector(args.config)
    
    # Start detection
    try:
        detector.start_webcam_detection()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        detector.stop()
        logger.info("Detection stopped")


if __name__ == '__main__':
    main()
