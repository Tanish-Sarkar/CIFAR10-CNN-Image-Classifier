import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.dataset import get_dataloaders
from src.model import build_model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Validate the model for one epoch
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # ✅ accumulate validation loss too
            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plot training and validation curves.
    """
    os.makedirs("reports/figures", exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.savefig("reports/loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(train_accuracies, label="Train Acc")
    plt.plot(val_accuracies, label="Val Acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.savefig("reports/accuracy_curve.png")
    plt.close()


def train_model(model_name="simple_cnn", batch_size=128, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(batch_size=batch_size)

    # model
    model = build_model(model_name=model_name, num_classes=10).to(device)

    # LOSS + OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")

        # ✅ these are floats for THIS epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        # ✅ append to the history lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # ✅ print the epoch values, not the lists
        print(f"Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}    |  Val Acc:   {val_acc:.4f}")

        # ✅ Save the best model (compare val_acc, not the list)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/best_model_{model_name}.pth")
            print("Best model saved!")

    # Plot training curves
    plot_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    print("Training complete.")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train_model(model_name="simple_cnn", batch_size=128, epochs=10, lr=0.001)

