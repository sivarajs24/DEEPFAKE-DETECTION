"""
Test the trained EfficientNet-B3 model on images
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.training.video_lightning import VideoDeepfakeModule

# Configuration
CHECKPOINT_PATH = "checkpoints/video/efficientnet_b3_epoch=16_val_loss=0.0616.ckpt"  # Best checkpoint
CONFIG_PATH = "configs/video/efficientnet_b3.yaml"

print("=" * 60)
print("Testing Trained EfficientNet-B3 Model")
print("=" * 60)

# Load config and model
print(f"\nLoading model from: {CHECKPOINT_PATH}")
config = load_config(CONFIG_PATH)

# Load trained model
model = VideoDeepfakeModule.load_from_checkpoint(
    CHECKPOINT_PATH,
    config=config
)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"✓ Model loaded successfully on {device}")
print(f"  Architecture: {config.model.architecture}")
print(f"  Training validation loss: 0.0616")

# Prepare transform
transform = T.Compose([
    T.Resize((config.data.image_size, config.data.image_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path_or_array):
    """Predict if an image is real or fake"""
    # Load image
    if isinstance(image_path_or_array, (str, Path)):
        image = Image.open(image_path_or_array).convert('RGB')
        print(f"\nTesting image: {Path(image_path_or_array).name}")
    else:
        # Numpy array (for testing)
        image = Image.fromarray(image_path_or_array.astype('uint8')).convert('RGB')
        print(f"\nTesting random image")
    
    # Transform and predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1)
    
    real_prob = probs[0][0].item()
    fake_prob = probs[0][1].item()
    prediction = "FAKE" if pred_class[0] == 1 else "REAL"
    confidence = max(real_prob, fake_prob)
    
    print(f"  Prediction: {prediction}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Real probability: {real_prob:.2%}")
    print(f"  Fake probability: {fake_prob:.2%}")
    
    return prediction, confidence, real_prob, fake_prob

# Test 1: Random synthetic image
print("\n" + "-" * 60)
print("Test 1: Random Synthetic Image")
print("-" * 60)
random_image = np.random.randint(0, 255, (224, 224, 3))
predict_image(random_image)

# Test 2: Check for any real test images
print("\n" + "-" * 60)
print("Test 2: Looking for test images in dataset...")
print("-" * 60)

test_dirs = [
    Path("data/video/test/real"),
    Path("data/video/test/fake"),
    Path("data/video/val/real"),
    Path("data/video/val/fake"),
    Path("data/image/val/real"),
    Path("data/image/val/fake"),
]

found_images = []
for test_dir in test_dirs:
    if test_dir.exists():
        images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        if images:
            found_images.extend([(img, test_dir.name) for img in images[:2]])  # Max 2 per dir

if found_images:
    print(f"Found {len(found_images)} test images")
    for img_path, label in found_images[:5]:  # Test max 5 images
        print(f"\n  Ground truth label: {label.upper()}")
        predict_image(img_path)
else:
    print("No test images found in dataset directories")
    print("\nTo test on your own image, use:")
    print("  python -c \"from test_trained_model import predict_image; predict_image('your_image.jpg')\"")

print("\n" + "=" * 60)
print("✓ Model is working and ready for inference!")
print("=" * 60)
