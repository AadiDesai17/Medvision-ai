"""
MedVision ConvNeXt-Tiny Standardized Training Pipeline
Chest X-Ray 3-Class Disease Prediction (NORMAL, PNEUMONIA, TUBERCULOSIS)

Standardized Experimental Protocol:
- Architecture: torchvision.models.convnext_tiny (ImageNet-1K pretrained)
- 3-Class linear classifier head: LayerNorm2d(768) -> Flatten() -> Linear(768, 3)
- Weighted CrossEntropyLoss ([0.8630, 0.9902, 1.1468]), label_smoothing=0.0
- AdamW (lr=3e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
- CosineAnnealingLR (T_max=15, eta_min=1e-6)
- Physical batch size: 16, Gradient accumulation: 2 -> Effective batch size: 32
- PyTorch CUDA FP16 AMP, GradScaler (init_scale=1024), gradient clipping (max_norm=1.0)
- Early stopping patience: 5 epochs on VALIDATION MACRO F1 ONLY
- Max epochs: 15, Seed: 42
"""

import os
import sys
import time
import argparse
import yaml
import json
import torch
import torch.nn as nn
from typing import Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import build_convnext_tiny, count_parameters
from src.dataset import get_dataloaders, get_standardized_class_weights, CLASS_NAMES
from src.utils import (
    set_seed,
    compute_metrics,
    plot_training_curves,
    save_json,
)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 2,
    max_grad_norm: float = 1.0,
) -> Dict[str, Any]:
    model.train()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_scaled = loss / grad_accum_steps

        scaler.scale(loss_scaled).backward()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    epoch_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds, class_names=CLASS_NAMES)
    metrics["loss"] = epoch_loss
    return metrics


def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[List[float]] = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    epoch_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds, y_probs=all_probs, class_names=CLASS_NAMES)
    metrics["loss"] = epoch_loss
    return metrics


def run_training(config_path: str, execute_training: bool = False) -> None:
    """
    Main training entrypoint with strict safety guard for Stage 1.
    """
    if not execute_training:
        print("=" * 70)
        print("SAFETY GUARD ACTIVE: STAGE 1 AUDIT & PREPARATION ONLY")
        print("Multi-epoch training is disabled in this mode.")
        print("To authorize full standardized training in Stage 3, pass:")
        print("  python src/train.py --execute-training")
        print("=" * 70)
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Reproducibility
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)
    print(f"[Setup] Deterministic seed set to {seed}")

    # 2. Hardware / Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Device] Using {device}: {gpu_name}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # 3. Model Architecture
    model = build_convnext_tiny(
        num_classes=cfg["model"]["num_classes"],
        pretrained=True,
    ).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"[Model] ConvNeXt-Tiny instantiated. Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # 4. DataLoaders
    loaders = get_dataloaders(
        dataset_root=cfg["dataset"]["root"],
        physical_batch_size=cfg["training"]["physical_batch_size"],
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
    )
    print(f"[Data] Train: {len(loaders['train'].dataset):,} samples, Val: {len(loaders['val'].dataset):,} samples")

    # 5. Loss with Standardized Class Weights (NO label smoothing)
    class_weights = get_standardized_class_weights(device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)
    print(f"[Loss] Weighted CrossEntropyLoss (weights={class_weights.tolist()}, label_smoothing=0.0)")

    # 6. Optimizer (AdamW)
    opt_cfg = cfg["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        betas=tuple(opt_cfg["betas"]),
        eps=float(opt_cfg["eps"]),
    )

    # 7. Scheduler (CosineAnnealingLR)
    sched_cfg = cfg["training"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=sched_cfg["T_max"],
        eta_min=float(sched_cfg["eta_min"]),
    )

    # 8. AMP GradScaler
    scaler = torch.amp.GradScaler("cuda", init_scale=cfg["training"]["amp"]["init_scale"])
    grad_accum_steps = cfg["training"]["gradient_accumulation_steps"]
    max_grad_norm = cfg["training"]["amp"]["gradient_clipping"]

    # 9. Paths & Checkpoints
    checkpoint_dir = os.path.join(PROJECT_ROOT, cfg["paths"]["checkpoint_dir"])
    results_dir = os.path.join(PROJECT_ROOT, cfg["paths"]["results_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(PROJECT_ROOT, cfg["paths"]["best_model_checkpoint"])

    # 10. Training Loop
    max_epochs = cfg["training"]["epochs"]
    early_stop_patience = cfg["training"]["early_stopping"]["patience"]
    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    early_stopping_triggered = False

    history: Dict[str, List[Any]] = {
        "epoch": [],
        "lr": [],
        "train_loss": [],
        "train_accuracy": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_precision": [],
        "val_macro_recall": [],
        "val_macro_f1": [],
        "epoch_duration_seconds": [],
        "gpu_memory_peak_mb": [],
        "checkpoint_saved": [],
        "patience_counter": [],
    }

    print(f"\n[Training] Starting standardized training ({max_epochs} max epochs, patience={early_stop_patience})...")
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
        )

        val_metrics = evaluate_epoch(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
        )

        scheduler.step()
        epoch_dur = round(time.time() - epoch_start, 2)

        peak_vram = 0.0
        if torch.cuda.is_available():
            peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

        val_macro_f1 = round(val_metrics["macro_f1"], 4)
        val_acc = round(val_metrics["accuracy"], 4)
        val_loss = round(val_metrics["loss"], 4)
        train_macro_f1 = round(train_metrics["macro_f1"], 4)
        train_acc = round(train_metrics["accuracy"], 4)
        train_loss = round(train_metrics["loss"], 4)

        # Model selection: VALIDATION MACRO F1 ONLY
        saved_checkpoint = False
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_val_accuracy = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            saved_checkpoint = True

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_macro_f1": val_macro_f1,
                "val_metrics": val_metrics,
                "config": cfg,
            }
            torch.save(checkpoint, best_checkpoint_path)
            save_msg = f" --> Saved new BEST checkpoint (Val Macro F1: {val_macro_f1:.4f})"
        else:
            patience_counter += 1
            save_msg = f" --> No improvement. Early stopping patience: {patience_counter}/{early_stop_patience}"

        # Record history
        history["epoch"].append(epoch)
        history["lr"].append(current_lr)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["train_macro_f1"].append(train_macro_f1)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_precision"].append(round(val_metrics["macro_precision"], 4))
        history["val_macro_recall"].append(round(val_metrics["macro_recall"], 4))
        history["val_macro_f1"].append(val_macro_f1)
        history["epoch_duration_seconds"].append(epoch_dur)
        history["gpu_memory_peak_mb"].append(peak_vram)
        history["checkpoint_saved"].append(saved_checkpoint)
        history["patience_counter"].append(patience_counter)

        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} [{epoch_dur}s] - "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Train Macro F1: {train_macro_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_macro_f1:.4f} | "
            f"LR: {current_lr:.2e} | VRAM: {peak_vram} MB{save_msg}"
        )

        if patience_counter >= early_stop_patience:
            print(f"\n[Early Stopping] Triggered at epoch {epoch} (patience {early_stop_patience} reached).")
            early_stopping_triggered = True
            break

    total_time = round(time.time() - start_time, 2)
    total_minutes = round(total_time / 60, 2)
    peak_gpu_overall = max(history["gpu_memory_peak_mb"]) if history["gpu_memory_peak_mb"] else 0.0

    print(f"\n[Training Finished] Total elapsed time: {total_minutes} minutes ({total_time}s).")
    print(f"[Best Epoch] Epoch {best_epoch} with Validation Macro F1 = {best_val_macro_f1:.4f} (Accuracy = {best_val_accuracy*100:.2f}%)")

    # 11. Save artifacts
    history_path = os.path.join(results_dir, "training_history.json")
    summary_path = os.path.join(results_dir, "training_summary.json")
    config_out_path = os.path.join(results_dir, "training_config.json")
    curves_path = os.path.join(results_dir, "training_curves.png")

    summary_data = {
        "model": "convnext_tiny",
        "torchvision_constructor": "torchvision.models.convnext_tiny",
        "pretrained_weights": "ConvNeXt_Tiny_Weights.DEFAULT",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "total_epochs_trained": len(history["epoch"]),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "early_stopping_triggered": early_stopping_triggered,
        "early_stopping_patience": early_stop_patience,
        "total_duration_minutes": total_minutes,
        "total_duration_seconds": total_time,
        "peak_gpu_memory_mb": peak_gpu_overall,
        "best_model_path": best_checkpoint_path,
        "training_history_path": history_path,
        "held_out_test_evaluated": False,
    }

    save_json(history, history_path)
    save_json(summary_data, summary_path)
    save_json(cfg, config_out_path)
    plot_training_curves(history, curves_path)

    print(f"[Artifacts Saved]")
    print(f"  - History: {history_path}")
    print(f"  - Summary: {summary_path}")
    print(f"  - Config:  {config_out_path}")
    print(f"  - Curves:  {curves_path}")
    print(f"  - Checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedVision ConvNeXt-Tiny Training")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "convnext_tiny.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--execute-training",
        action="store_true",
        help="Safety authorization flag to execute multi-epoch training (Stage 3)",
    )
    args = parser.parse_args()

    run_training(config_path=args.config, execute_training=args.execute_training)
