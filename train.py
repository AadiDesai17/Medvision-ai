"""
MedVision Disease Prediction - Training Pipeline
Author: K. Gnana Saaketh (ROLE: Deep Learning / Disease Prediction)
Project: Medvision — AI Medical Imaging Assistant
"""

import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import (
    NUM_CLASSES,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED,
    BEST_MODEL_PATH,
    MODEL_CONFIG_PATH,
    MODELS_DIR,
    DEVICE,
    CLASS_NAMES
)
from src.utils import seed_everything, save_json, plot_training_curves
from src.dataset import prepare_dataset, get_data_loaders, compute_class_weights
from src.model import build_model

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Executes a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Executes validation across the validation split."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_model(
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    seed=RANDOM_SEED,
    device=DEVICE,
    checkpoint_path=BEST_MODEL_PATH
):
    """
    Main training controller.
    Loads dataset, initializes ResNet50, optimizes with Adam, validates, and saves best model.
    """
    seed_everything(seed)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("      MEDVISION DISEASE PREDICTION — TRAINING PIPELINE      ")
    print("=" * 60)
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Architecture  : ResNet50 Transfer Learning")
    print(f"Epochs        : {epochs}")
    print(f"Batch Size    : {batch_size}")
    print(f"Learning Rate : {lr}")
    print(f"Random Seed   : {seed}")
    print("=" * 60)

    # 1. Dataset Preparation
    train_df, val_df, test_df = prepare_dataset(random_seed=seed)
    train_loader, val_loader, _ = get_data_loaders(train_df, val_df, test_df, batch_size=batch_size)

    # 2. Build Model & Loss
    model = build_model(num_classes=NUM_CLASSES, pretrained=True, freeze_early_layers=True, device=device)
    class_weights = compute_class_weights(train_df, device=device)
    print(f"[Training] Class weights (Normal vs Pneumonia): {class_weights.cpu().numpy().round(3)}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Only optimize trainable parameters (layer3, layer4, fc)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    start_time = time.time()

    print("\nStarting model training...")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}%"
        )

        # Checkpoint Best Model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> [CHECKPOINT SAVED] New best validation loss: {val_loss:.4f} to {checkpoint_path.name}")

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.4f} | Accuracy: {best_val_acc*100:.2f}%")

    # 3. Save Plots
    loss_path, acc_path = plot_training_curves(history)
    print(f"[Visualizations] Saved curves to:\n  - {loss_path}\n  - {acc_path}")

    # 4. Save Model Configuration & Metadata
    model_config = {
        **model.get_config(),
        "classes": CLASS_NAMES,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "training_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "device": str(device),
        "dataset_samples": {
            "total": len(train_df) + len(val_df) + len(test_df),
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        }
    }
    save_json(model_config, MODEL_CONFIG_PATH)
    print(f"[Config] Saved model metadata to {MODEL_CONFIG_PATH}")

    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MedVision ResNet50 Disease Classifier")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
