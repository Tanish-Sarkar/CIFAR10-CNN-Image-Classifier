import os
import argparse
from typing import Dict

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

from src.model import build_model
from src.dataset import CIFAR10_MEAN, CIFAR10_STD

# CIFAR-10 fixed class names (in order)
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

def get_inference_transform():
    """
    This should mimic the *eval/test* transform used during training:
    - Resize / ensure 32x32
    - ToTensor
    - Normalize with CIFAR10 mean & std
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

def load_image(image_path: str, device: torch.device) -> torch.Tensor:
    """
    Load an image from disk and preprocess it into a batch tensor of shape [1, 3, 32, 32].
    Resolves image_path relative to the project root (folder containing src/).
    """

    # Resolve project root: .../CIFAR10 CNN Image Classifier
    project_root = Path(__file__).resolve().parent.parent

    # Allow both absolute and relative paths:
    img_path = Path(image_path)
    if not img_path.is_absolute():
        img_path = project_root / img_path

    if not img_path.exists():
        raise FileNotFoundError(f"Image path '{img_path}' does not exist.")

    img = Image.open(img_path).convert("RGB")
    transform = get_inference_transform()
    tensor = transform(img)          # [3, 32, 32]
    tensor = tensor.unsqueeze(0)     # [1, 3, 32, 32]
    return tensor.to(device)  

def load_trained_model(model_name: str, device: torch.device) -> nn.Module:
    """
    Build the same architecture and load the best weights.
    Keras equivalent:
        model = build_model()
        model.load_weights("best_model.h5")
    """
    model = build_model(model_name=model_name, num_classes=len(CLASS_NAMES)).to(device)
    weights_path = f"models/best_model_{model_name}.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file '{weights_path}' does not exist.")

    state_dict = torch.load(weights_path, map_location=device)  
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_image(
    image_path: str,
    model_name: str = "simple_cnn",
) -> Dict:
    """
    High-level inference function:
    - load model
    - load & preprocess image
    - forward pass
    - return best class and probabilities
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trained model
    model = load_trained_model(model_name, device)

    # Load and preprocess image
    img_tensor = load_image(image_path, device)  # [1, 3, 32, 32]

    # --- IMPORTANT: define logits here ---
    with torch.no_grad():
        logits = model(img_tensor)              # <-- FIXED
        probs = torch.softmax(logits, dim=1)[0] # 1D array of 10 classes

    # Convert to CPU numpy
    probs_np = probs.cpu().numpy()
    top_idx = int(np.argmax(probs_np))
    top_class = CLASS_NAMES[top_idx]
    top_prob = float(probs_np[top_idx])

    return {
        "image_path": image_path,
        "predicted_class": top_class,
        "predicted_index": top_idx,
        "confidence": top_prob,
        "all_probabilities": {
            CLASS_NAMES[i]: float(p) for i, p in enumerate(probs_np)
        }
    }



def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="simple_cnn",
        help="Model name to use (simple_cnn or resnet18).",
    )

    args = parser.parse_args()

    result = predict_image(args.image, model_name=args.model_name)
    print("\nPrediction result:")
    print(f"Image: {result['image_path']}")
    print(f"Predicted class: {result['predicted_class']} (index {result['predicted_index']})")
    print(f"Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()