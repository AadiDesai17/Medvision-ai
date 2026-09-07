"""
Utility functions for reproducibility, evaluation metrics, visualization,
and checkpoint handling for the MedVision ResNet50 project.
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
    """Sets random seeds for full reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes classification metrics including Accuracy, Macro/Weighted Precision,
    Recall, F1, and per-class breakdown with support.
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    acc = float(accuracy_score(y_true_np, y_pred_np))
    macro_p = float(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    weighted_p = float(precision_score(y_true_np, y_pred_np, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true_np, y_pred_np, average="weighted", zero_division=0))
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
                "precision": float(round(per_class_p[idx], 4)) if idx < len(per_class_p) else 0.0,
                "recall": float(round(per_class_r[idx], 4)) if idx < len(per_class_r) else 0.0,
                "f1_score": float(round(per_class_f1[idx], 4)) if idx < len(per_class_f1) else 0.0,
                "support": supp,
            }

    return {
        "accuracy": float(round(acc, 4)),
        "macro_precision": float(round(macro_p, 4)),
        "macro_recall": float(round(macro_r, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "weighted_precision": float(round(weighted_p, 4)),
        "weighted_recall": float(round(weighted_r, 4)),
        "weighted_f1": float(round(weighted_f1, 4)),
        "confusion_matrix": cm,
        "per_class": per_class_metrics,
    }


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix (ResNet50)"
) -> None:
    """Plots and saves raw and normalized confusion matrices side-by-side."""
    cm_arr = np.array(cm)
    cm_norm = cm_arr.astype(float) / np.maximum(cm_arr.sum(axis=1, keepdims=True), 1e-12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    im1 = ax1.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.figure.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=f"{title} (Counts)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh1 = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax1.text(
                j, i, f"{cm_arr[i, j]:d}",
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh1 else "black",
            )

    # Normalized percentages
    im2 = ax2.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax2.figure.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set(
        xticks=np.arange(cm_norm.shape[1]),
        yticks=np.arange(cm_norm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=f"{title} (Recall %)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh2 = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax2.text(
                j, i, f"{cm_norm[i, j]*100:.1f}%",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh2 else "black",
            )

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(state: Dict[str, Any], filepath: Path) -> None:
    """Saves model checkpoint safely."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    map_location: str = "cpu"
) -> Dict[str, Any]:
    """
    Loads checkpoint weights into model. Handles both raw state_dict and dict wrappers.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    import io
    with open(filepath, "rb") as f:
        buffer = io.BytesIO(f.read())
    checkpoint = torch.load(buffer, map_location=map_location, weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert len(missing) == 0, f"Missing keys in state_dict: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys in state_dict: {unexpected}"

    return checkpoint
