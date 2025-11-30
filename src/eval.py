import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.dataset import get_dataloaders
from src.model import build_model

def evaluate_model(model_name: str = "simple_cnn", batch_size: int = 128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _,_, test_loader, class_names = get_dataloaders(batch_size=batch_size)
    model = build_model(model_name, num_classes=len(class_names)).to(device)

    weights_path = f"models/best_model_{model_name}.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Evaluation loop
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)


    # Matrix
    overall_acc = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {overall_acc:.4f}")

    # sklearn classification report
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=4,
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    os.makedirs("reports/figures", exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"reports/figures/confusion_matrix_{model_name}.png")
    plt.close()

    # per-class accuracy
    per_class_acc = {}
    for idx, class_name in enumerate(class_names):
        mask = (all_labels == idx)
        if mask.sum() == 0:
            acc = 0.0
        else:
            acc = (all_preds[mask] == all_labels[mask]).mean()
        per_class_acc[class_name] = float(acc)

    # Bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_names)), [per_class_acc[c] for c in class_names])
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(f"Per-class Accuracy - {model_name}")
    plt.tight_layout()
    plt.savefig(f"reports/figures/per_class_accuracy_{model_name}.png")
    plt.close()

    metrics = {
        "test_accuracy": float(overall_acc),
        "per_class_accuracy": per_class_acc,
    }

    os.makedirs("reports", exist_ok=True)
    with open(f"reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nSaved:")
    print(f"- Confusion matrix: reports/figures/confusion_matrix_{model_name}.png")
    print(f"- Per-class accuracy: reports/figures/per_class_accuracy_{model_name}.png")
    print(f"- Metrics JSON: reports/test_metrics_{model_name}.json")


if __name__ == "__main__":
    evaluate_model(
        model_name="simple_cnn",
        batch_size=128,
    )

