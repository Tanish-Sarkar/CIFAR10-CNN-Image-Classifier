import os
import argparse
from typing import Dict

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

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
    Keras equivalent:
        img = tf.keras.preprocessing.image.load_img(...)
        x = img_to_array(...)
        x = preprocess(x)
        x = np.expand_dims(x, axis=0)
    """
    if os.path.exists(image_path):
        raise FileNotFoundError(f"Image path '{image_path}' does not exist.")

    image = Image.open(image_path).convert("RGB")
    transform = get_inference_transform()
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
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

def predict_image(image_path: str, model_name: str = "simple_cnn") -> Dict:
    """
    High-level function:
    - load model
    - load & preprocess image
    - run forward pass
    - return top prediction + probabilities
    """
    

