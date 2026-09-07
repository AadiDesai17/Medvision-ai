import os
import random
import yaml
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

def set_seed(seed=42):
    """Set random seeds across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path="configs/vit_b16.yaml"):
    """Load YAML configuration dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def calculate_class_weights(targets, num_classes=3, method="sqrt_inverse"):
    """
    Calculate class weights strictly from training targets.
    
    method: 'sqrt_inverse' computes normalized square-root inverse-frequency weights:
      w_c = sqrt(N / count_c), normalized so mean(w) == 1.0.
      This gently balances the minority classes (TB) without introducing extreme gradient scale.
    """
    targets = np.array(targets)
    counts = np.bincount(targets, minlength=num_classes)
    total_samples = len(targets)
    
    if method == "sqrt_inverse":
        inv_freq = total_samples / (counts.astype(np.float32) + 1e-6)
        raw_weights = np.sqrt(inv_freq)
        normalized_weights = raw_weights / np.mean(raw_weights)
    elif method == "inverse":
        raw_weights = total_samples / (num_classes * counts.astype(np.float32) + 1e-6)
        normalized_weights = raw_weights
    else:
        normalized_weights = np.ones(num_classes, dtype=np.float32)
        
    return torch.tensor(normalized_weights, dtype=torch.float32)

def calculate_metrics(y_true, y_pred, class_names=None):
    """
    Calculate classification metrics:
    - Overall Accuracy
    - Macro Precision, Recall, F1
    - Weighted F1
    - Per-class Precision, Recall, F1, Support
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": {}
    }
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(per_class_p))]
        
    for i, name in enumerate(class_names):
        metrics["per_class"][name] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(per_class_supp[i])
        }
        
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"
            
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax
    )
    ax.set_title("MedVision ViT-B/16 Confusion Matrix (Test Set)", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=8)
    ax.set_ylabel("True Label", fontsize=12, labelpad=8)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_training_curves(history, save_path):
    """Plot and save Loss and Macro F1 curves over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#ff7f0e")
    ax1.set_title("Cross-Entropy Loss vs Epoch", fontsize=13)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    
    # Macro F1 plot
    ax2.plot(epochs, history["train_macro_f1"], "o-", label="Train Macro F1", color="#2ca02c")
    ax2.plot(epochs, history["val_macro_f1"], "s-", label="Val Macro F1", color="#d62728")
    
    # Mark best epoch
    best_epoch = np.argmax(history["val_macro_f1"]) + 1
    best_f1 = np.max(history["val_macro_f1"])
    ax2.scatter([best_epoch], [best_f1], color="gold", s=150, zorder=5, edgecolor="black", label=f"Best Val F1 ({best_f1:.4f} @ Ep {best_epoch})")
    
    ax2.set_title("Macro F1 vs Epoch (Primary Selection Metric)", fontsize=13)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Macro F1", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def get_gpu_memory():
    """Return allocated and reserved GPU memory in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
        return f"Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB"
    return "GPU not active"
