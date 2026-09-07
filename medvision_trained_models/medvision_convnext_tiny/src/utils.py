"""
MedVision Utility Functions & Evaluation Metrics
Chest X-Ray 3-Class Disease Prediction (NORMAL, PNEUMONIA, TUBERCULOSIS)

Provides:
- Reproducibility seeding (Seed: 42)
- Comprehensive multi-class metrics (Accuracy, Macro/Weighted F1, Precision, Recall, per-class stats)
- Visualization routines (Confusion matrix, training loss/metric curves)
- Checkpoint & JSON serialization utilities
"""

import os
import random
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)


def set_seed(seed: int = 42) -> None:
    """
    Configures deterministic seeds across all random number generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes rigorous multi-class evaluation metrics.
    Primary model selection metric is Validation Macro F1.
    """
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    
    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    # Weighted metrics
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1": float(f1_per[i]),
            "support": int(sup_per[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist()

    metrics = {
        "accuracy": acc,
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "macro_precision": float(p_macro),
        "weighted_precision": float(p_weighted),
        "macro_recall": float(r_macro),
        "weighted_recall": float(r_weighted),
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
    }

    # Multiclass ROC AUC (One-vs-Rest)
    if y_probs is not None and len(np.unique(y_true)) == len(class_names):
        try:
            roc_macro = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
            roc_weighted = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted"))
            metrics["macro_roc_auc"] = roc_macro
            metrics["weighted_roc_auc"] = roc_weighted
        except Exception:
            metrics["macro_roc_auc"] = None
            metrics["weighted_roc_auc"] = None

    return metrics


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """
    Renders and saves a clean, publication-grade confusion matrix visualization.
    """
    cm_array = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.matshow(cm_array, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            val = cm_array[i, j]
            ax.text(
                j, i, f"{val}",
                ha="center", va="center",
                color="white" if val > cm_array.max() / 2 else "black",
                fontsize=11, fontweight="bold",
            )

    fig.colorbar(cax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="left")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontweight="bold", labelpad=10)
    ax.set_title(title, fontweight="bold", pad=15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


def plot_training_curves(history: Dict[str, List[float]], save_path: str) -> None:
    """
    Plots training & validation loss and Macro F1 trajectories across epochs.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e")
    axes[0].set_title("Training & Validation Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Macro F1 plot
    axes[1].plot(epochs, history["val_macro_f1"], "^-", label="Val Macro F1", color="#2ca02c")
    if "train_macro_f1" in history:
        axes[1].plot(epochs, history["train_macro_f1"], "x--", label="Train Macro F1", color="#9467bd")
    axes[1].set_title("Validation Macro F1", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


def save_json(data: Any, path: str) -> None:
    """
    Saves serializable dictionary to JSON with indentation.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
