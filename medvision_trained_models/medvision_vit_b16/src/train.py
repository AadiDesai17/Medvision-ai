import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    set_seed,
    load_config,
    calculate_class_weights,
    calculate_metrics,
    plot_training_curves,
    get_gpu_memory
)
from src.model import create_vit_b16, count_parameters
from src.dataset import get_dataloaders

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, accum_steps=2, clip_grad=1.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    optimizer.zero_grad()
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, targets)
            # Normalize loss for gradient accumulation
            loss = loss / accum_steps
            
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        running_loss += loss.item() * accum_steps * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(all_targets, all_preds)
    
    return epoch_loss, metrics

@torch.no_grad()
def evaluate_validation(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, targets)
            
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(all_targets, all_preds)
    
    return epoch_loss, metrics

def train(config_path="configs/vit_b16.yaml"):
    config = load_config(config_path)
    set_seed(config["training"].get("seed", 42))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print("MedVision: Vision Transformer ViT-B/16 Training")
    print("==================================================")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Setup paths
    checkpoint_dir = PROJECT_ROOT / config["paths"]["checkpoint_dir"]
    checkpoint_path = PROJECT_ROOT / config["paths"]["best_model_path"]
    config_save_path = PROJECT_ROOT / config["paths"]["config_path"]
    results_dir = PROJECT_ROOT / config["paths"]["results_dir"]
    curves_path = PROJECT_ROOT / config["paths"]["training_curves_path"]
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load DataLoaders
    print("\nLoading dataset splits...")
    train_loader, val_loader, test_loader, class_names, class_to_idx = get_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Val samples:   {len(val_loader.dataset):,}")
    print(f"Test samples:  {len(test_loader.dataset):,} (held out)")
    print(f"Classes ({len(class_names)}): {class_names}")
    
    # Class weights strictly from train set
    class_weights = calculate_class_weights(
        train_loader.dataset.targets,
        num_classes=len(class_names),
        method=config["training"].get("class_weighting", "sqrt_inverse")
    ).to(device)
    
    print(f"\nCalculated Class Weights (training set only):")
    for idx, c_name in enumerate(class_names):
        print(f"  {c_name}: {class_weights[idx].item():.4f}")
        
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Build ViT-B/16 model
    print("\nInstantiating ViT-B/16 with ImageNet-1K pretrained weights...")
    model = create_vit_b16(
        num_classes=config["dataset"]["num_classes"],
        pretrained=True,
        dropout=config["model"].get("dropout", 0.0)
    ).to(device)
    
    param_counts = count_parameters(model)
    print(f"Total Parameters:     {param_counts['total']:,}")
    print(f"Trainable Parameters: {param_counts['trainable']:,}")
    
    # Optimizer & Scheduler
    train_cfg = config["training"]
    opt_cfg = train_cfg["optimizer"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=float(opt_cfg.get("eps", 1e-8))
    )
    
    max_epochs = train_cfg["max_epochs"]
    sched_cfg = train_cfg["scheduler"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=float(sched_cfg.get("eta_min", 1e-6))
    )
    
    use_amp = train_cfg["amp"].get("enabled", True) and (device.type == "cuda")
    init_scale = float(train_cfg.get("amp", {}).get("init_scale", 1024.0))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp, init_scale=init_scale)
    accum_steps = train_cfg.get("gradient_accumulation_steps", 2)
    clip_grad = float(train_cfg.get("gradient_clipping", 1.0))
    patience = train_cfg.get("early_stopping_patience", 5)
    
    print(f"Training Strategy: AMP={use_amp}, Accumulation Steps={accum_steps} (Effective Batch Size={train_cfg['effective_batch_size']})")
    print(f"Optimizer: AdamW (lr={opt_cfg['learning_rate']}, wd={opt_cfg['weight_decay']})")
    print(f"Scheduler: CosineAnnealingLR (T_max={max_epochs})")
    print(f"Early Stopping Patience: {patience} epochs monitoring Val Macro F1\n")
    
    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": []
    }
    
    start_time = time.time()
    
    for epoch in range(1, max_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_start = time.time()
        
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            accum_steps=accum_steps, clip_grad=clip_grad
        )
        
        val_loss, val_metrics = evaluate_validation(model, val_loader, criterion, device)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        train_acc = train_metrics["accuracy"] * 100.0
        train_f1 = train_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"] * 100.0
        val_f1 = val_metrics["macro_f1"]
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_macro_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["lr"].append(current_lr)
        
        # Epoch Summary Output formatted to exact specification
        print("--------------------------------------------------", flush=True)
        print(f"Epoch {epoch}/{max_epochs}", flush=True)
        print(f"Train Loss: {train_loss:.4f}", flush=True)
        print(f"Train Accuracy: {train_acc:.2f}%", flush=True)
        print(f"Train Macro F1: {train_f1:.4f}\n", flush=True)
        print(f"Val Loss: {val_loss:.4f}", flush=True)
        print(f"Val Accuracy: {val_acc:.2f}%", flush=True)
        print(f"Val Macro F1: {val_f1:.4f}\n", flush=True)
        print(f"Learning Rate: {current_lr:.6e}", flush=True)
        print("--------------------------------------------------", flush=True)
        
        # Checkpoint selection on Validation Macro F1
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            
            # Save model checkpoint
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "architecture": "vit_b_16",
                "num_classes": config["dataset"]["num_classes"],
                "class_names": class_names,
                "class_to_idx": class_to_idx,
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc / 100.0,
                "val_loss": val_loss,
                "input_size": config["dataset"]["image_size"],
                "mean": config["dataset"]["mean"],
                "std": config["dataset"]["std"]
            }
            torch.save(checkpoint_payload, checkpoint_path)
            
            # Save metadata config.json
            metadata_payload = {
                "architecture": "vit_b_16",
                "num_classes": config["dataset"]["num_classes"],
                "class_names": class_names,
                "class_mapping": class_to_idx,
                "input_size": config["dataset"]["image_size"],
                "preprocessing": {
                    "image_size": config["dataset"]["image_size"],
                    "mean": config["dataset"]["mean"],
                    "std": config["dataset"]["std"],
                    "color_mode": "RGB (converted from 512x512 Grayscale PNG)"
                },
                "pretrained_weights": "ViT_B_16_Weights.IMAGENET1K_V1",
                "optimizer": "AdamW",
                "learning_rate": float(opt_cfg["learning_rate"]),
                "weight_decay": float(opt_cfg["weight_decay"]),
                "scheduler": "CosineAnnealingLR",
                "physical_batch_size": train_cfg["batch_size"],
                "gradient_accumulation_steps": accum_steps,
                "effective_batch_size": train_cfg["effective_batch_size"],
                "amp": {
                    "enabled": use_amp,
                    "init_scale": init_scale
                },
                "seed": train_cfg["seed"],
                "best_epoch": best_epoch,
                "best_val_macro_f1": float(best_val_macro_f1),
                "best_val_accuracy": float(val_acc / 100.0),
                "best_val_loss": float(val_loss),
                "class_weights": {c_name: float(class_weights[idx].item()) for idx, c_name in enumerate(class_names)}
            }
            with open(config_save_path, "w", encoding="utf-8") as f:
                json.dump(metadata_payload, f, indent=2)
                
            print(f"*** NEW BEST CHECKPOINT ***", flush=True)
            print(f"Validation Macro F1: {val_f1:.4f}\n", flush=True)
        else:
            patience_counter += 1
            print(f"(No improvement in Val Macro F1 for {patience_counter}/{patience} epochs)\n", flush=True)
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs. Best epoch was {best_epoch} with Val Macro F1: {best_val_macro_f1:.4f}", flush=True)
                break
                
    total_time = time.time() - start_time
    print("==================================================")
    print("TRAINING COMPLETE")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best Epoch: {best_epoch} with Val Macro F1: {best_val_macro_f1:.4f}")
    print(f"Checkpoint saved: {checkpoint_path}")
    print("==================================================")
    
    # Save training curves
    plot_training_curves(history, str(curves_path))
    print(f"Training curves saved to: {curves_path}")

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/vit_b16.yaml"
    train(config_file)
