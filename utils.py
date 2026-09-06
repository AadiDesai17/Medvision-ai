"""
MedVision Utility Functions & Visualization Tools
Author: K. Gnana Saaketh (ROLE: Deep Learning / Disease Prediction)
Project: Medvision — AI Medical Imaging Assistant
"""

import os
import json
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.config import RANDOM_SEED, PLOTS_DIR

def seed_everything(seed=RANDOM_SEED):
    """
    Enforces strict reproducibility across random, numpy, and PyTorch (CPU & CUDA).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_json(data, file_path, indent=2):
    """Saves dictionary data to a formatted JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)

def load_json(file_path):
    """Loads and parses data from a JSON file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_training_curves(history, output_dir=PLOTS_DIR):
    """
    Generates and saves clean training vs. validation loss and accuracy curves.
    
    Parameters:
        history (dict): Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        output_dir (Path): Output directory for plot images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "b-o", label="Training Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", linewidth=2)
    plt.title("MedVision ResNet50 — Training & Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    loss_plot_path = output_dir / "loss_curve.png"
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    
    # 2. Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Training Accuracy", linewidth=2)
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], "g--s", label="Validation Accuracy", linewidth=2)
    plt.title("MedVision ResNet50 — Training & Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xticks(epochs)
    plt.ylim([0, 105])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    acc_plot_path = output_dir / "accuracy_curve.png"
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    
    return str(loss_plot_path), str(acc_plot_path)

def plot_confusion_matrix(cm, class_names, output_path=None):
    """
    Renders and saves a clean, annotated Confusion Matrix.
    
    Parameters:
        cm (np.ndarray): 2D Confusion Matrix array.
        class_names (list): List of class names.
        output_path (Path): Path to save the confusion matrix image.
    """
    if output_path is None:
        output_path = PLOTS_DIR / "confusion_matrix.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("MedVision Test Confusion Matrix", fontsize=13, fontweight="bold")
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13,
                fontweight="bold"
            )
            
    plt.ylabel("True Clinical Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return str(output_path)
