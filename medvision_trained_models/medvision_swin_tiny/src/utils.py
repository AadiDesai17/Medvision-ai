"""
Utility functions for reproducibility, evaluation metrics, visualization, and checkpoints.
"""
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(y_true: List[int], y_pred: List[int], class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Computes classification metrics including Accuracy, Macro/Weighted F1,
    and per-class precision, recall, F1, and support.
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    acc = float(accuracy_score(y_true_np, y_pred_np))
    macro_p = float(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true_np, y_pred_np).tolist()

    per_class_p = precision_score(y_true_np, y_pred_np, average=None, zero_division=0).tolist()
    per_class_r = recall_score(y_true_np, y_pred_np, average=None, zero_division=0).tolist()
    per_class_f1 = f1_score(y_true_np, y_pred_np, average=None, zero_division=0).tolist()

    per_class_metrics = {}
    if class_names:
        for idx, name in enumerate(class_names):
            supp = int(np.sum(y_true_np == idx))
            per_class_metrics[name] = {
                "precision": float(per_class_p[idx]) if idx < len(per_class_p) else 0.0,
                "recall": float(per_class_r[idx]) if idx < len(per_class_r) else 0.0,
                "f1": float(per_class_f1[idx]) if idx < len(per_class_f1) else 0.0,
                "support": supp,
            }

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "per_class": per_class_metrics,
    }


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], save_path: Path, title: str = "Confusion Matrix") -> None:
    """Plots and saves a styled confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_arr[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_training_curves(history: Dict[str, List[float]], save_path: Path) -> None:
    """Plots training and validation loss and Macro F1 curves over epochs."""
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if "train_loss" in history and "val_loss" in history:
        ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
        ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="s")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.6)

    if "val_macro_f1" in history:
        ax2.plot(epochs, history["val_macro_f1"], label="Val Macro F1", color="green", marker="^")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Macro F1")
        ax2.set_title("Validation Macro F1 Score")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def save_checkpoint(state: Dict[str, Any], filepath: Path) -> None:
    """Saves model checkpoint safely."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Loads checkpoint weights into model and optional training components."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint
