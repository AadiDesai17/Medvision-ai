"""
Training pipeline for Swin-Tiny on Chest X-ray dataset.
Executes Stage 3 full training run. Incorporates AMP (init_scale=1024),
gradient accumulation (effective batch size 32), gradient clipping (1.0),
and early stopping on Validation Macro F1.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).parent))
from dataset import CLASSES, get_dataloaders
from model import count_parameters, create_swin_tiny
from utils import compute_metrics, plot_training_curves, save_checkpoint, set_seed


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 2,
    grad_clip: float = 1.0,
) -> float:
    """Runs one training epoch with gradient accumulation and AMP."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    optimizer.zero_grad()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_scaled = loss / grad_accum_steps

        scaler.scale(loss_scaled).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, any]:
    """Runs validation evaluation and computes Accuracy and Macro F1."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().tolist())

    epoch_loss = running_loss / max(total_samples, 1)
    metrics = compute_metrics(all_targets, all_preds, class_names=CLASSES)
    metrics["loss"] = epoch_loss
    return metrics


def run_training(config_path: str):
    """Executes model training pipeline based on provided config file."""
    start_time_all = time.time()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Reproducibility
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Using device: {device} ({gpu_name})")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    # 2. DataLoaders & Class Weights (strictly train & val splits)
    dataset_root = Path(cfg["dataset"]["dataset_root"])
    batch_size = cfg["training"]["physical_batch_size"]
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
    print(f"Calculated Class Weights: {class_weights.tolist()}")
    for idx, (cls_name, w) in enumerate(zip(CLASSES, class_weights.tolist())):
        print(f"  - {cls_name:<15}: {w:.4f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 3. Model
    model = create_swin_tiny(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
    ).to(device)

    total_p, train_p = count_parameters(model)
    print(f"Model: Swin-Tiny | Total Params: {total_p:,} | Trainable Params: {train_p:,}")

    # 4. Optimizer & Scheduler
    opt_cfg = cfg["optimizer"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        betas=tuple(opt_cfg["betas"]),
        eps=float(opt_cfg["eps"]),
    )

    max_epochs = cfg["training"]["max_epochs"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["scheduler"]["T_max"],
        eta_min=float(cfg["scheduler"]["eta_min"]),
    )

    # 5. AMP GradScaler
    init_scale = float(cfg["training"].get("amp_init_scale", 1024.0))
    scaler = torch.amp.GradScaler("cuda", init_scale=init_scale)

    # 6. Training State Tracking
    grad_accum_steps = cfg["training"].get("gradient_accumulation_steps", 2)
    grad_clip = float(cfg["training"].get("gradient_clipping", 1.0))
    patience = cfg["training"].get("early_stopping_patience", 5)

    best_macro_f1 = -1.0
    best_epoch = -1
    best_metrics = {}
    no_improve_count = 0
    early_stopped = False

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_precision": [],
        "val_macro_recall": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "lr": [],
        "epoch_duration_sec": [],
    }

    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = Path(cfg["paths"]["best_model_path"])
    history_path = Path(cfg["paths"]["history_path"])

    print("\n" + "=" * 80)
    print(f"STARTING SWIN-TINY TRAINING: {max_epochs} Epochs, Batch Size {batch_size} (Eff: {batch_size * grad_accum_steps})")
    print("=" * 80)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_accum_steps=grad_accum_steps,
            grad_clip=grad_clip,
        )

        val_metrics = evaluate_validation(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()
        epoch_time = time.time() - t0

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]
        val_p = val_metrics["macro_precision"]
        val_r = val_metrics["macro_recall"]
        val_wf1 = val_metrics["weighted_f1"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_precision"].append(val_p)
        history["val_macro_recall"].append(val_r)
        history["val_macro_f1"].append(val_f1)
        history["val_weighted_f1"].append(val_wf1)
        history["lr"].append(current_lr)
        history["epoch_duration_sec"].append(epoch_time)

        print(
            f"Epoch [{epoch:02d}/{max_epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f} | LR: {current_lr:.2e}"
        )

        # Checkpoint on best Validation Macro F1
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_epoch = epoch
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_precision": val_p,
                "val_macro_recall": val_r,
                "val_macro_f1": val_f1,
                "val_weighted_f1": val_wf1,
            }
            no_improve_count = 0
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_metrics": val_metrics,
                    "config": cfg,
                    "classes": CLASSES,
                },
                filepath=best_model_path,
            )
            print(f"  --> Saved new best checkpoint at Epoch {epoch:02d} (Val Macro F1: {val_f1:.4f})")
        else:
            no_improve_count += 1
            print(f"  --> No improvement for {no_improve_count} epoch(s). Best Val Macro F1: {best_macro_f1:.4f} (Epoch {best_epoch:02d})")

        # Save history JSON each epoch
        with open(history_path, "w") as hf:
            json.dump(history, hf, indent=2)

        if no_improve_count >= patience:
            early_stopped = True
            print(f"\n[Early Stopping] Triggered after {patience} epochs without improvement.")
            break

    total_training_time = time.time() - start_time_all
    peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0.0

    # Save training curves plot
    plot_training_curves(history, results_dir / "training_curves.png")

    # Save final summary JSON
    summary = {
        "status": "COMPLETED",
        "architecture": "Swin-Tiny",
        "total_epochs_completed": len(history["epoch"]),
        "early_stopped": early_stopped,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "total_training_time_sec": total_training_time,
        "peak_vram_mb": peak_vram_mb,
        "device": gpu_name,
        "final_lr": optimizer.param_groups[0]["lr"],
        "class_weights": {cls: w for cls, w in zip(CLASSES, class_weights.tolist())},
        "best_checkpoint_path": str(best_model_path),
    }
    with open(results_dir / "training_summary.json", "w") as sf:
        json.dump(summary, sf, indent=2)

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE in {total_training_time / 60:.2f} minutes.")
    print(f"Best Epoch: {best_epoch:02d} with Validation Macro F1: {best_macro_f1:.4f}")
    print(f"Best Checkpoint: {best_model_path} ({best_model_path.stat().st_size / (1024 ** 2):.2f} MB)")
    print(f"Peak VRAM: {peak_vram_mb:.2f} MB ({peak_vram_mb / 1024:.2f} GB)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Swin-Tiny on Chest X-ray dataset")
    parser.add_argument("--config", type=str, default="configs/swin_tiny.yaml", help="Path to config YAML")
    args = parser.parse_args()
    run_training(args.config)
