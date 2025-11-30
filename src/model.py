import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleCNN(nn.Module):
    """
    A small CNN for CIFAR-10 (32x32 RGB).
    Architecture:
      - Conv -> BN -> ReLU -> MaxPool
      - Conv -> BN -> ReLU -> MaxPool
      - Conv -> BN -> ReLU -> MaxPool
      - FC -> Dropout -> FC (num_classes)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3x32x32 -> 32x16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32x16x16 -> 64x8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64x8x8 -> 128x4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
def build_simple_cnn(num_classes: int = 10) -> nn.Module:
    """
    Return the SimpleCNN model.    
    """
    return SimpleCNN(num_classes=num_classes)

def build_resnet18(num_classes: int = 10, pretrained: bool = True) -> nn.Module:

    """
    Builds a ResNet18-based classifier adapted for CIFAR-10.
    - If pretrained=True, uses ImageNet-pretrained weights and replaces the final layer.
    - Adjusts the first conv layer if needed for CIFAR-10 (not strictly necessary since CIFAR is also 3-channel).
    """
    try:
        # Newer Api
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        models = models.resnet18(weights=weights)
    except Exception:
        # fallback to the older Api
        models = models.resnet18(pretrained=pretrained)

    # ResNet18 expects 3x224x224; we will handle resizing in transforms if we use this.
    in_features = models.fc.in_features
    models.fc = nn.Linear(in_features, num_classes)
    return models

def build_model(model_name: str = "simple_cnn", num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Factory function to build a model by name.
    Supported models:
      - "simple_cnn": A small CNN for CIFAR-10.
      - "resnet18": ResNet18 adapted for CIFAR-10.
    """
    if model_name == "simple_cnn":
        return build_simple_cnn(num_classes=num_classes)
    elif model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    

if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    models = build_model("simple_cnn").to(device)
    x = torch.randn(8,3,32,32).to(device)
    logits = models(x)
    print("SimpleCNN output shape:", logits.shape)


