"""
Quick Start Example
Demonstrates basic usage of DeepGuard-X
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src import DeepGuardXInference, get_logger

logger = get_logger(__name__)


def example_video_detection():
    """Example: Video deepfake detection"""
    logger.info("="*60)
    logger.info("Example 1: Video Deepfake Detection")
    logger.info("="*60)
    
    # Initialize detector
    detector = DeepGuardXInference(
        config_path="configs/ensemble_config.yaml",
        use_onnx=True,
        device="cuda"
    )
    
    # Detect from video file
    video_path = "path/to/your/video.mp4"
    
    logger.info(f"Analyzing video: {video_path}")
    
    results = detector.predict(video_path=video_path)
    
    # Print results
    logger.info(f"Prediction: {results['final_label']}")
    logger.info(f"Deepfake Score: {results['final_score']:.4f}")
    logger.info(f"Confidence: {results['confidence']:.4f}")
    
    logger.info("\nIndividual Model Scores:")
    for model, score in results['individual_scores'].items():
        logger.info(f"  {model}: {score:.4f}")


def example_audio_detection():
    """Example: Audio deepfake detection"""
    logger.info("\n" + "="*60)
    logger.info("Example 2: Audio Deepfake Detection")
    logger.info("="*60)
    
    detector = DeepGuardXInference()
    
    audio_path = "path/to/your/audio.wav"
    
    logger.info(f"Analyzing audio: {audio_path}")
    
    results = detector.predict(audio_path=audio_path)
    
    logger.info(f"Prediction: {results['final_label']}")
    logger.info(f"Deepfake Score: {results['final_score']:.4f}")


def example_multimodal_detection():
    """Example: Multi-modal detection (video + audio)"""
    logger.info("\n" + "="*60)
    logger.info("Example 3: Multi-Modal Detection")
    logger.info("="*60)
    
    detector = DeepGuardXInference()
    
    video_path = "path/to/your/video.mp4"
    audio_path = "path/to/your/audio.wav"
    
    logger.info("Analyzing video and audio...")
    
    results = detector.predict(
        video_path=video_path,
        audio_path=audio_path,
        return_details=True
    )
    
    logger.info(f"\nFinal Prediction: {results['final_label']}")
    logger.info(f"Ensemble Score: {results['final_score']:.4f}")
    logger.info(f"Confidence: {results['confidence']:.4f}")
    
    logger.info("\nAll Module Scores:")
    for model, score in results['individual_scores'].items():
        status = "FAKE" if score > 0.5 else "REAL"
        logger.info(f"  {model:20s}: {score:.4f} ({status})")
    
    if 'details' in results:
        logger.info(f"\nAnalysis Details:")
        logger.info(f"  Models Used: {results['details']['num_models_used']}")
        logger.info(f"  Score Variance: {results['details']['score_variance']:.6f}")
        logger.info(f"  Score Std Dev: {results['details']['score_std']:.6f}")


def example_batch_processing():
    """Example: Batch processing multiple files"""
    logger.info("\n" + "="*60)
    logger.info("Example 4: Batch Processing")
    logger.info("="*60)
    
    detector = DeepGuardXInference()
    
    video_files = [
        "video1.mp4",
        "video2.mp4",
        "video3.mp4"
    ]
    
    results_list = []
    
    for video_path in video_files:
        logger.info(f"\nProcessing: {video_path}")
        
        try:
            results = detector.predict(video_path=video_path)
            results_list.append({
                'file': video_path,
                'prediction': results['final_label'],
                'score': results['final_score']
            })
            
            logger.info(f"  Result: {results['final_label']} ({results['final_score']:.4f})")
            
        except Exception as e:
            logger.error(f"  Error processing {video_path}: {e}")
    
    # Summary
    logger.info("\n" + "-"*60)
    logger.info("Batch Processing Summary")
    logger.info("-"*60)
    
    fake_count = sum(1 for r in results_list if r['prediction'] == 'FAKE')
    real_count = len(results_list) - fake_count
    
    logger.info(f"Total Files: {len(results_list)}")
    logger.info(f"Fake: {fake_count}")
    logger.info(f"Real: {real_count}")
    logger.info(f"Average Score: {sum(r['score'] for r in results_list) / len(results_list):.4f}")


def main():
    """Run all examples"""
    logger.info("DeepGuard-X Quick Start Examples")
    logger.info("="*60)
    
    # Note: Replace file paths with actual files to run examples
    logger.info("\n⚠️  Note: Update file paths in the examples before running")
    logger.info("The examples below show the API usage patterns\n")
    
    # Uncomment to run examples with real files
    # example_video_detection()
    # example_audio_detection()
    # example_multimodal_detection()
    # example_batch_processing()
    
    logger.info("\n✅ Examples completed!")
    logger.info("\nFor more information, see:")
    logger.info("  - README.md: General overview")
    logger.info("  - docs/: Detailed documentation")
    logger.info("  - configs/: Configuration examples")


if __name__ == '__main__':
    main()
