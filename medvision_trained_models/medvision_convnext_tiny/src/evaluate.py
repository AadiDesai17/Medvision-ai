"""
MedVision ConvNeXt-Tiny Evaluation & Error Analysis
Chest X-Ray 3-Class Disease Prediction (NORMAL, PNEUMONIA, TUBERCULOSIS)

Evaluates the trained ConvNeXt-Tiny model on the held-out test split (1,951 samples)
using deterministic test preprocessing. Generates comprehensive classification metrics,
confusion matrix visual, and detailed error analysis CSV.
"""

import os
import sys
import argparse
import csv
import yaml
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import build_convnext_tiny
from src.dataset import ChestXRayDataset, get_transforms, CLASS_NAMES, IDX_TO_CLASS
from src.utils import compute_metrics, plot_confusion_matrix, save_json


def run_evaluation(
    checkpoint_path: str,
    config_path: str,
    split: str = "test",
    output_dir: str = "results/chest_xray_convnext_tiny",
) -> Dict[str, Any]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at: {checkpoint_path}. "
            f"Cannot evaluate before completing training in Stage 3."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Evaluating on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Build model and load checkpoint
    model = build_convnext_tiny(num_classes=cfg["model"]["num_classes"], pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    best_epoch = checkpoint.get("epoch", "N/A")
    val_macro_f1_ckpt = checkpoint.get("val_macro_f1", "N/A")

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    print(f"[Evaluate] Successfully loaded checkpoint from {checkpoint_path}")
    print(f"[Evaluate] Checkpoint trained epoch: {best_epoch} | Validation Macro F1: {val_macro_f1_ckpt}")

    # Dataset & DataLoader (deterministic eval transform)
    transforms_dict = get_transforms()
    eval_transform = transforms_dict[split]
    dataset = ChestXRayDataset(
        dataset_root=cfg["dataset"]["root"],
        split=split,
        transform=eval_transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["training"]["physical_batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
    )
    print(f"[Evaluate] Loaded {len(dataset)} samples from '{split}' split across {len(loader)} batches.")

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[List[float]] = []
    all_paths: List[str] = []

    with torch.no_grad():
        for images, targets, paths in loader:
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
            all_paths.extend(paths)

    avg_loss = total_loss / len(dataset)
    metrics = compute_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_probs=np.array(all_probs),
        class_names=CLASS_NAMES,
    )

    total_samples = len(dataset)
    total_correct = sum(1 for t, p in zip(all_targets, all_preds) if t == p)
    total_incorrect = total_samples - total_correct
    error_percentage = round((total_incorrect / total_samples) * 100, 4)

    metrics["loss"] = round(avg_loss, 4)
    metrics["split"] = split
    metrics["total_samples"] = total_samples
    metrics["total_correct"] = total_correct
    metrics["total_incorrect"] = total_incorrect
    metrics["error_count"] = total_incorrect
    metrics["error_percentage"] = error_percentage
    metrics["checkpoint_path"] = checkpoint_path
    metrics["best_training_epoch"] = best_epoch

    # Output paths
    full_out_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(full_out_dir, exist_ok=True)

    report_path = os.path.join(full_out_dir, f"{split}_evaluation_report.json")
    cm_path = os.path.join(full_out_dir, f"{split}_confusion_matrix.png" if split != "test" else "confusion_matrix.png")
    errors_path = os.path.join(full_out_dir, f"{split}_errors.csv" if split != "test" else "test_errors.csv")

    save_json(metrics, report_path)
    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, cm_path, title=f"ConvNeXt-Tiny {split.title()} Confusion Matrix")

    # Error analysis CSV
    error_records = []
    for path, target, pred, prob in zip(all_paths, all_targets, all_preds, all_probs):
        if target != pred:
            pred_confidence = prob[pred]
            error_records.append({
                "filename": os.path.basename(path),
                "true_class": IDX_TO_CLASS[target],
                "predicted_class": IDX_TO_CLASS[pred],
                "predicted_confidence": f"{pred_confidence * 100:.2f}%",
                "p_normal": f"{prob[0]:.4f}",
                "p_pneumonia": f"{prob[1]:.4f}",
                "p_tuberculosis": f"{prob[2]:.4f}",
                "file_path": path,
            })

    with open(errors_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "true_class", "predicted_class", "predicted_confidence", "p_normal", "p_pneumonia", "p_tuberculosis", "file_path"],
        )
        writer.writeheader()
        writer.writerows(error_records)

    print("\n" + "=" * 65)
    print(f"EVALUATION SUMMARY ({split.upper()} SET - {total_samples} SAMPLES)")
    print("=" * 65)
    print(f"Accuracy:           {metrics['accuracy'] * 100:.2f}% ({total_correct}/{total_samples})")
    print(f"Macro F1:           {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:        {metrics['weighted_f1']:.4f}")
    print(f"Macro Precision:    {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:       {metrics['macro_recall']:.4f}")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall:    {metrics['weighted_recall']:.4f}")
    print(f"Total Errors:       {total_incorrect} / {total_samples} ({error_percentage:.2f}%)")
    print("-" * 65)
    print("Per-Class Results:")
    for cls_name, cls_m in metrics["per_class"].items():
        print(f"  {cls_name:<14} Prec: {cls_m['precision']:.4f} | Rec: {cls_m['recall']:.4f} | F1: {cls_m['f1']:.4f} (Support: {cls_m['support']})")
    print("-" * 65)
    print("Confusion Matrix:")
    print(f"  Classes: {CLASS_NAMES}")
    for i, row in enumerate(metrics["confusion_matrix"]):
        print(f"  {CLASS_NAMES[i]:<14}: {row}")
    print("=" * 65)
    print(f"Report saved: {report_path}")
    print(f"Matrix saved: {cm_path}")
    print(f"Errors saved: {errors_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConvNeXt-Tiny Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "chest_xray_convnext_tiny", "best_model.pth"),
        help="Path to trained checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "convnext_tiny.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split to evaluate",
    )
    parser.add_argument(
        "--execute-test-eval",
        action="store_true",
        help="Safety authorization flag to evaluate test split (Stage 4)",
    )
    args = parser.parse_args()

    if not args.execute_test_eval:
        print("=" * 70)
        print("SAFETY GUARD ACTIVE: STAGE 1 AUDIT & PREPARATION ONLY")
        print("Test set evaluation is strictly reserved for Stage 4 after training approval.")
        print("To evaluate in Stage 4, run:")
        print("  python src/evaluate.py --execute-test-eval")
        print("=" * 70)
    else:
        run_evaluation(checkpoint_path=args.checkpoint, config_path=args.config, split=args.split)
