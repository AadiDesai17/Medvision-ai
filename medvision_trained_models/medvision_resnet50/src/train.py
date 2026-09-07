"""
Standardized training pipeline for ResNet50 on Chest X-ray 3-class classification.
Aligned with the standardized experimental protocol used for ViT-B/16 and Swin-Tiny:
- Architecture: torchvision.models.resnet50 with ImageNet pretrained weights (ResNet50_Weights.DEFAULT)
- Custom Classification Head: Dropout(p=0.2) + Linear(2048, 3)
- Optimizer: AdamW(lr=3e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
- Scheduler: CosineAnnealingLR(T_max=15, eta_min=1e-6)
- Loss: Weighted CrossEntropyLoss [NORMAL: 0.8630, PNEUMONIA: 0.9902, TUBERCULOSIS: 1.1468]
- Label Smoothing: STRICTLY 0.0 (NO label smoothing)
- Batching: Physical batch size = 16, Gradient Accumulation = 2 (Effective batch size = 32)
- Mixed Precision: CUDA FP16 AMP with GradScaler(init_scale=1024)
- Gradient Clipping: max_norm = 1.0
- Model Selection: Highest Validation Macro F1 ONLY
- Max Epochs: 15, Early Stopping Patience: 5
- Seed: 42
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from dataset import CLASSES, get_dataloaders
from model import create_resnet50, count_parameters
from utils import set_seed, compute_metrics, save_checkpoint


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler = None,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 2,
) -> Dict[str, float]:
    """
    Executes a single training epoch with gradient accumulation and FP16 AMP.
    """
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    optimizer.zero_grad()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
                loss_scaled = loss / grad_accum_steps
            scaler.scale(loss_scaled).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        running_loss += loss.item() * len(targets)
        preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().tolist())

    metrics = compute_metrics(all_targets, all_preds, class_names=CLASSES)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    return metrics


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluates model on validation split. Model selection is strictly based on val Macro F1.
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * len(targets)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().tolist())

    metrics = compute_metrics(all_targets, all_preds, class_names=CLASSES)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    return metrics


def plot_training_curves(history: Dict[str, List], save_path: Path) -> None:
    """
    Generates training and validation curves for loss, accuracy, macro F1, and learning rate.
    """
    epochs = history["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss Curve
    axes[0, 0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0, 0].plot(epochs, history["val_loss"], "r-s", label="Val Loss")
    axes[0, 0].set_title("Cross-Entropy Loss Trajectory")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    axes[0, 0].legend()

    # 2. Accuracy Curve
    axes[0, 1].plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc (%)")
    axes[0, 1].plot(epochs, [a * 100 for a in history["val_acc"]], "g-^", label="Val Acc (%)")
    axes[0, 1].set_title("Classification Accuracy (%)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    axes[0, 1].legend()

    # 3. Validation Macro F1 Curve
    best_f1 = max(history["val_macro_f1"])
    best_ep = epochs[history["val_macro_f1"].index(best_f1)]
    axes[1, 0].plot(epochs, history["val_macro_f1"], "m-D", label="Val Macro F1")
    axes[1, 0].axvline(best_ep, color="gold", linestyle="--", label=f"Best Ep {best_ep} ({best_f1:.4f})")
    axes[1, 0].set_title("Validation Macro F1 (Model Selection Metric)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Macro F1")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    axes[1, 0].legend()

    # 4. Learning Rate Schedule
    axes[1, 1].plot(epochs, history["lr"], "c-x", label="Cosine Annealing LR")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    axes[1, 1].legend()

    plt.suptitle("MedVision ResNet50 Standardized Training Convergence", fontsize=15, y=0.99)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to: {save_path}", flush=True)


def run_training(config_path: str, allow_training: bool = False):
    """
    Executes standardized ResNet50 training pipeline.
    Guarded by allow_training flag.
    """
    if not allow_training:
        print("[SAFETY GUARD] Full model training requires explicit --execute-training flag.", flush=True)
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Reproducibility
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 2. DataLoaders (Train and Validation ONLY; Test set remains untouched)
    dataset_root = Path(cfg["dataset"]["dataset_root"])
    batch_size = cfg["training"].get("batch_size", 16)
    grad_accum_steps = cfg["training"].get("gradient_accumulation_steps", 2)
    num_workers = cfg["dataset"].get("num_workers", 2)
    pin_memory = cfg["dataset"].get("pin_memory", True)
    img_size = cfg["dataset"].get("img_size", 224)

    train_loader, val_loader, _, class_weights = get_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        img_size=img_size,
    )
    print(f"Dataset Loaded: Train = {len(train_loader.dataset):,} | Val = {len(val_loader.dataset):,}", flush=True)
    print(f"Calculated Class Weights: {class_weights.tolist()}", flush=True)
    
    # Standardized Loss: Weighted CrossEntropy, NO label smoothing
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.0
    )

    # 3. Model Architecture
    # Always starts from ImageNet pretrained weights ResNet50_Weights.DEFAULT
    model = create_resnet50(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("head_dropout", 0.2),
    ).to(device)

    total_p, train_p, _ = count_parameters(model)
    print(f"Model: ResNet50 | Total Parameters: {total_p:,} | Trainable: {train_p:,}", flush=True)

    # 4. Optimizer & Scheduler
    lr = float(cfg["optimizer"].get("lr", 3e-5))
    weight_decay = float(cfg["optimizer"].get("weight_decay", 0.01))
    betas = tuple(cfg["optimizer"].get("betas", [0.9, 0.999]))
    eps = float(cfg["optimizer"].get("eps", 1e-8))

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    max_epochs = cfg["training"].get("max_epochs", 15)
    eta_min = float(cfg["scheduler"].get("eta_min", 1e-6))
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=eta_min)

    # 5. AMP GradScaler
    use_amp = cfg["training"].get("amp", True) and device.type == "cuda"
    scaler = None
    if use_amp:
        init_scale = float(cfg["training"].get("grad_scaler_init_scale", 1024))
        scaler = torch.amp.GradScaler("cuda", init_scale=init_scale)
        print(f"CUDA FP16 AMP enabled with GradScaler(init_scale={init_scale})", flush=True)

    grad_clip = float(cfg["training"].get("gradient_clipping", 1.0))
    patience = cfg["training"].get("early_stopping_patience", 5)

    model_dir = Path(cfg["paths"]["checkpoint_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save training configuration
    training_config_path = results_dir / "training_config.json"
    with open(training_config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved training configuration to: {training_config_path}", flush=True)

    best_model_path = Path(cfg["paths"]["best_model_path"])

    # Training state history
    history = {
        "epoch": [],
        "lr": [],
        "train_loss": [],
        "train_acc": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_p": [],
        "val_macro_r": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "epoch_duration_sec": [],
        "gpu_memory_mb": [],
    }

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_val_acc = 0.0
    best_val_loss = 0.0
    patience_counter = 0
    t_start = time.time()
    peak_gpu_memory_overall = 0.0

    print("\n" + "=" * 70, flush=True)
    print(f"STARTING STANDARDIZED RETRAINING (Max Epochs: {max_epochs}, Patience: {patience})", flush=True)
    print(f"Physical Batch: {batch_size}, Grad Accum: {grad_accum_steps} (Effective: {batch_size * grad_accum_steps})", flush=True)
    print("=" * 70, flush=True)

    for epoch in range(1, max_epochs + 1):
        ep_t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        tr_res = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=grad_clip,
            grad_accum_steps=grad_accum_steps,
        )
        val_res = evaluate_validation(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        scheduler.step()

        ep_duration = round(time.time() - ep_t0, 2)
        gpu_mem = 0.0
        if device.type == "cuda":
            gpu_mem = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2)
            peak_gpu_memory_overall = max(peak_gpu_memory_overall, gpu_mem)

        history["epoch"].append(epoch)
        history["lr"].append(current_lr)
        history["train_loss"].append(round(tr_res["loss"], 4))
        history["train_acc"].append(round(tr_res["accuracy"], 4))
        history["train_macro_f1"].append(round(tr_res["macro_f1"], 4))
        history["val_loss"].append(round(val_res["loss"], 4))
        history["val_acc"].append(round(val_res["accuracy"], 4))
        history["val_macro_p"].append(round(val_res["macro_precision"], 4))
        history["val_macro_r"].append(round(val_res["macro_recall"], 4))
        history["val_macro_f1"].append(round(val_res["macro_f1"], 4))
        history["val_weighted_f1"].append(round(val_res["weighted_f1"], 4))
        history["epoch_duration_sec"].append(ep_duration)
        history["gpu_memory_mb"].append(gpu_mem)

        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} ({ep_duration:.1f}s, LR: {current_lr:.2e}) | "
            f"Train Loss: {tr_res['loss']:.4f} Acc: {tr_res['accuracy']*100:.2f}% F1: {tr_res['macro_f1']:.4f} | "
            f"Val Loss: {val_res['loss']:.4f} Acc: {val_res['accuracy']*100:.2f}% Macro F1: {val_res['macro_f1']:.4f} (Peak VRAM: {gpu_mem:.1f}MB)",
            flush=True
        )

        # Model selection: Highest validation Macro F1 ONLY
        if val_res["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_res["macro_f1"]
            best_epoch = epoch
            best_val_acc = val_res["accuracy"]
            best_val_loss = val_res["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved NEW best checkpoint at epoch {epoch:02d} (Val Macro F1: {best_val_macro_f1:.4f})", flush=True)
        else:
            patience_counter += 1
            print(f"  --> No Macro F1 improvement. Early stopping patience: {patience_counter}/{patience}", flush=True)
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch:02d} (patience={patience}).", flush=True)
                break

    total_duration_min = round((time.time() - t_start) / 60.0, 2)
    print("\n" + "=" * 70, flush=True)
    print(f"TRAINING COMPLETE in {total_duration_min:.2f} mins.", flush=True)
    print(f"Best Checkpoint: Epoch {best_epoch:02d} | Val Macro F1: {best_val_macro_f1:.4f} | Val Acc: {best_val_acc*100:.2f}%", flush=True)
    print(f"Saved to: {best_model_path}", flush=True)
    print("=" * 70, flush=True)

    # Save training history
    history_path = results_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}", flush=True)

    # Save training summary
    summary = {
        "model": "resnet50",
        "torchvision_constructor": "torchvision.models.resnet50",
        "pretrained_weights": "ResNet50_Weights.DEFAULT",
        "total_parameters": total_p,
        "trainable_parameters": train_p,
        "total_epochs_trained": len(history["epoch"]),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 4),
        "early_stopping_triggered": (patience_counter >= patience),
        "early_stopping_patience": patience,
        "total_duration_minutes": total_duration_min,
        "peak_gpu_memory_mb": peak_gpu_memory_overall,
        "best_model_path": str(best_model_path),
        "training_history_path": str(history_path),
        "held_out_test_evaluated": False,
    }
    summary_path = results_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training summary to: {summary_path}", flush=True)

    # Generate training curves
    curves_path = results_dir / "training_curves.png"
    plot_training_curves(history, curves_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardized ResNet50 Training")
    parser.add_argument("--config", type=str, default="configs/resnet50.yaml")
    parser.add_argument("--execute-training", action="store_true", help="Explicitly enable training execution (Stage 3)")
    args = parser.parse_args()

    run_training(args.config, allow_training=args.execute_training)
