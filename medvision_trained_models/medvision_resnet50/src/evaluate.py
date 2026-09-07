"""
Evaluation script for trained ResNet50 model on held-out test split (1,951 images).
Computes Accuracy, Macro/Weighted Precision, Recall, F1, per-class metrics,
confusion matrix, diagnostic error analysis, and saves misclassified error records.
Does NOT train or retrain the model.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from dataset import CLASSES, IDX_TO_CLASS, ChestXRayDataset, get_dataloaders, get_transforms
from model import create_resnet50, count_parameters
from utils import compute_metrics, load_checkpoint, plot_confusion_matrix


@torch.no_grad()
def evaluate_test_set(
    model: torch.nn.Module,
    test_ds: ChestXRayDataset,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Runs single-pass evaluation across the held-out test DataLoader."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[List[float]] = []
    misclassified_records: List[Dict[str, Any]] = []

    sample_idx = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)

        preds = torch.argmax(logits, dim=1).cpu().tolist()
        targets_list = targets.tolist()
        probs_list = probs.cpu().tolist()

        for b_idx in range(len(preds)):
            pred = preds[b_idx]
            target = targets_list[b_idx]
            prob = probs_list[b_idx]
            conf = prob[pred]
            file_path, _ = test_ds.samples[sample_idx]

            if pred != target:
                misclassified_records.append({
                    "filename": file_path.name,
                    "true_class": IDX_TO_CLASS[target],
                    "predicted_class": IDX_TO_CLASS[pred],
                    "predicted_confidence": float(round(conf, 6)),
                    "prob_normal": float(round(prob[0], 6)),
                    "prob_pneumonia": float(round(prob[1], 6)),
                    "prob_tuberculosis": float(round(prob[2], 6)),
                })
            sample_idx += 1

        all_preds.extend(preds)
        all_targets.extend(targets_list)
        all_probs.extend(probs_list)

    metrics = compute_metrics(all_targets, all_preds, class_names=CLASSES)
    cm = metrics["confusion_matrix"]

    total_images = len(all_targets)
    correct_count = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
    incorrect_count = total_images - correct_count
    error_pct = round((incorrect_count / total_images) * 100, 2)

    # Diagnostic error analysis
    # Index 0: NORMAL, 1: PNEUMONIA, 2: TUBERCULOSIS
    error_analysis = {
        "pneumonia_misclassified_as_normal": int(cm[1][0]),
        "tuberculosis_misclassified_as_normal": int(cm[2][0]),
        "normal_misclassified_as_pneumonia": int(cm[0][1]),
        "normal_misclassified_as_tuberculosis": int(cm[0][2]),
        "pneumonia_misclassified_as_tuberculosis": int(cm[1][2]),
        "tuberculosis_misclassified_as_pneumonia": int(cm[2][1]),
    }

    metrics["total_test_images"] = total_images
    metrics["number_correct"] = correct_count
    metrics["number_incorrect"] = incorrect_count
    metrics["test_error_count"] = incorrect_count
    metrics["test_error_percentage"] = error_pct
    metrics["error_analysis"] = error_analysis
    metrics["misclassified_records"] = misclassified_records

    return metrics


def run_evaluation(
    checkpoint_path: str,
    config_path: str,
    output_dir: str
):
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_file}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("MEDVISION RESNET50: HELD-OUT TEST EVALUATION (STAGE 4)")
    print(f"Device:            {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Active Checkpoint: {ckpt_file} ({ckpt_file.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 75)

    # 1. Model
    model = create_resnet50(
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
        dropout=cfg["model"].get("head_dropout", 0.2),
    ).to(device)

    load_checkpoint(ckpt_file, model, map_location=str(device))
    model.eval()
    print("[1/5] Loaded weights into ResNet50 model successfully.")

    # 2. Test DataLoader
    dataset_root = Path(cfg["dataset"]["dataset_root"])
    batch_size = cfg["training"].get("batch_size", 16)
    img_size = cfg["dataset"].get("img_size", 224)
    test_dir = dataset_root / "test"

    test_ds = ChestXRayDataset(test_dir, transform=get_transforms("test", img_size=img_size))
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    print(f"[2/5] Loaded held-out test split: {len(test_loader.dataset)} images.")
    for cls in CLASSES:
        print(f"       {cls:<14}: {test_ds.class_counts[cls]} images")

    # 3. Single-pass evaluation
    print("[3/5] Running deterministic evaluation on test set (torch.no_grad)...")
    metrics = evaluate_test_set(model, test_ds, test_loader, device)

    print("\n" + "=" * 55)
    print("--- STANDARDIZED RESNET50 TEST EVALUATION SUMMARY ---")
    print("=" * 55)
    print(f"Total Test Images:     {metrics['total_test_images']}")
    print(f"Number Correct:        {metrics['number_correct']}")
    print(f"Number Incorrect:      {metrics['number_incorrect']}")
    print(f"Test Error Count:      {metrics['test_error_count']}")
    print(f"Test Error Percentage: {metrics['test_error_percentage']:.2f}%")
    print("-" * 55)
    print(f"Overall Accuracy:      {metrics['accuracy']*100:.2f}% (exact: {metrics['accuracy']:.6f})")
    print(f"Macro Precision:       {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:          {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:        {metrics['macro_f1']:.4f}")
    print(f"Weighted Precision:    {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall:       {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1-Score:     {metrics['weighted_f1']:.4f}")
    print("\nPer-Class Breakdown:")
    for cls_name, res in metrics["per_class"].items():
        print(f"  {cls_name:<14} | Precision: {res['precision']:.4f} | Recall: {res['recall']:.4f} | F1: {res['f1_score']:.4f} | Support: {res['support']}")

    print("\nConfusion Matrix (NORMAL, PNEUMONIA, TUBERCULOSIS):")
    cm = metrics["confusion_matrix"]
    for row_name, row in zip(CLASSES, cm):
        print(f"  {row_name:<14}: {row}")

    print("\nDiagnostic Error Breakdown:")
    for err_name, err_cnt in metrics["error_analysis"].items():
        print(f"  {err_name:<42}: {err_cnt}")

    # 4. Save results & artifacts
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    misclassified = metrics.pop("misclassified_records")

    report_path = out_dir / "test_evaluation_report.json"
    report_dict = {
        "model": "resnet50",
        "architecture": "torchvision.models.resnet50",
        "checkpoint_used": str(ckpt_file),
        "checkpoint_selection_criterion": "Highest Validation Macro F1 ONLY (Epoch 13)",
        "test_dataset": str(test_dir),
        "test_count": metrics["total_test_images"],
        "number_correct": metrics["number_correct"],
        "number_incorrect": metrics["number_incorrect"],
        "test_error_count": metrics["test_error_count"],
        "test_error_percentage": metrics["test_error_percentage"],
        "accuracy": metrics["accuracy"],
        "accuracy_pct": round(metrics["accuracy"] * 100, 2),
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "error_analysis": metrics["error_analysis"],
        "device": str(device) + f" ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})",
    }

    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[4/5] Saved test evaluation report to: {report_path}")

    # Save test errors CSV
    errors_csv_path = out_dir / "test_errors.csv"
    if misclassified:
        with open(errors_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(misclassified[0].keys()))
            writer.writeheader()
            writer.writerows(misclassified)
        print(f"Saved {len(misclassified)} misclassified error records to: {errors_csv_path}")
    else:
        print("Zero misclassifications detected; test_errors.csv not created.")

    # Save confusion matrix plot
    cm_path = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, CLASSES, cm_path, title="Confusion Matrix (Standardized ResNet50)")
    print(f"[5/5] Saved confusion matrix plot to: {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Standardized ResNet50 on Held-Out Test Set")
    parser.add_argument("--checkpoint", type=str, default="models/chest_xray_resnet50/best_model.pth")
    parser.add_argument("--config", type=str, default="configs/resnet50.yaml")
    parser.add_argument("--output-dir", type=str, default="results/chest_xray_resnet50")
    args = parser.parse_args()

    run_evaluation(args.checkpoint, args.config, args.output_dir)
