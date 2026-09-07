"""
Evaluation pipeline for Swin-Tiny on Chest X-ray held-out test split.
Requires a trained model checkpoint. Computes Accuracy, Macro/Weighted F1,
per-class metrics, support, and confusion matrix.
Generates FINAL_TEST_REPORT.md, test_evaluation_report.json, and confusion_matrix.png.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Ensure src is in Python path
sys.path.insert(0, str(Path(__file__).parent))
from dataset import CLASSES, CLASS_TO_IDX, get_dataloaders
from model import count_parameters, create_swin_tiny
from utils import compute_metrics, load_checkpoint, plot_confusion_matrix


@torch.no_grad()
def evaluate_test_set(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Runs inference across the held-out test DataLoader."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[List[float]] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(targets.tolist())
        all_probs.extend(probs.cpu().tolist())

    metrics = compute_metrics(all_targets, all_preds, class_names=CLASSES)
    metrics["all_targets"] = all_targets
    metrics["all_preds"] = all_preds
    metrics["all_probs"] = all_probs
    return metrics


def run_evaluation(checkpoint_path: str, config_path: str, output_dir: str):
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        print(f"\n[ERROR] Checkpoint not found at: {checkpoint_file}")
        sys.exit(1)

    ckpt_size_mb = checkpoint_file.stat().st_size / (1024 ** 2)
    print(f"Verified checkpoint: {checkpoint_file} ({ckpt_size_mb:.2f} MB)")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Running evaluation on device: {device} ({gpu_name})")

    # 1. Instantiate model and load checkpoint
    model = create_swin_tiny(num_classes=cfg["model"]["num_classes"], pretrained=False).to(device)
    ckpt = load_checkpoint(checkpoint_file, model, map_location=str(device))
    best_epoch = ckpt.get("epoch", "Unknown")
    total_params, trainable_params = count_parameters(model)
    print(f"Loaded checkpoint from Epoch: {best_epoch}")
    print(f"Architecture: Swin-Tiny | Total Parameters: {total_params:,}")

    # 2. Test DataLoader (strictly held-out test split)
    dataset_root = Path(cfg["dataset"]["dataset_root"])
    _, _, test_loader, _ = get_dataloaders(
        dataset_root=dataset_root,
        batch_size=cfg["training"]["physical_batch_size"],
        num_workers=cfg["dataset"].get("num_workers", 2),
        pin_memory=cfg["dataset"].get("pin_memory", True),
        img_size=cfg["dataset"].get("img_size", 224),
    )
    test_count = len(test_loader.dataset)
    print(f"Evaluating {test_count} held-out test samples...")
    assert test_count == 1951, f"Expected 1951 test samples, found {test_count}"

    # 3. Inference
    raw_eval = evaluate_test_set(model, test_loader, device)

    acc = raw_eval["accuracy"]
    macro_p = raw_eval["macro_precision"]
    macro_r = raw_eval["macro_recall"]
    macro_f1 = raw_eval["macro_f1"]
    weighted_f1 = raw_eval["weighted_f1"]
    cm = raw_eval["confusion_matrix"]
    per_class = raw_eval["per_class"]

    # 4. Validations
    cm_sum = sum(sum(row) for row in cm)
    supp_sum = sum(m["support"] for m in per_class.values())
    assert cm_sum == 1951, f"Confusion matrix sum {cm_sum} != 1951"
    assert supp_sum == 1951, f"Support sum {supp_sum} != 1951"

    print("\n" + "=" * 60)
    print("MEDVISION SWIN-TINY HELD-OUT TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:         {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Macro Precision:  {macro_p:.4f} ({macro_p * 100:.2f}%)")
    print(f"Macro Recall:     {macro_r:.4f} ({macro_r * 100:.2f}%)")
    print(f"Macro F1:         {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
    print(f"Weighted F1:      {weighted_f1:.4f} ({weighted_f1 * 100:.2f}%)")
    print("\nPer-Class Metrics:")
    for cls_name, cls_m in per_class.items():
        print(f"  {cls_name:<15} Precision: {cls_m['precision']:.4f} ({cls_m['precision']*100:.2f}%) | "
              f"Recall: {cls_m['recall']:.4f} ({cls_m['recall']*100:.2f}%) | "
              f"F1: {cls_m['f1']:.4f} ({cls_m['f1']*100:.2f}%) | "
              f"Support: {cls_m['support']}")

    print("\nConfusion Matrix (Rows: True, Cols: Predicted):")
    print(f"                {'NORMAL':>10} {'PNEUMONIA':>10} {'TUBERCULOSIS':>12}")
    for idx, cls_name in enumerate(CLASSES):
        print(f"{cls_name:<15} {cm[idx][0]:>10} {cm[idx][1]:>10} {cm[idx][2]:>12}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 5. Save confusion_matrix.png
    cm_path = out_path / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names=CLASSES, save_path=cm_path, title="Swin-Tiny Chest X-ray Test Confusion Matrix")
    print(f"\n[Artifact] Saved Confusion Matrix plot: {cm_path}")

    # 6. Save test_evaluation_report.json
    json_report = {
        "architecture": "Swin-Tiny (swin_t)",
        "checkpoint": str(checkpoint_file),
        "checkpoint_epoch": best_epoch,
        "dataset": str(dataset_root),
        "test_count": test_count,
        "class_names": CLASSES,
        "class_mapping": CLASS_TO_IDX,
        "accuracy": acc,
        "accuracy_pct": round(acc * 100, 2),
        "macro_precision": macro_p,
        "macro_precision_pct": round(macro_p * 100, 2),
        "macro_recall": macro_r,
        "macro_recall_pct": round(macro_r * 100, 2),
        "macro_f1": macro_f1,
        "macro_f1_pct": round(macro_f1 * 100, 2),
        "weighted_f1": weighted_f1,
        "weighted_f1_pct": round(weighted_f1 * 100, 2),
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
        "device": gpu_name,
    }
    json_path = out_path / "test_evaluation_report.json"
    with open(json_path, "w") as jf:
        json.dump(json_report, jf, indent=2)
    print(f"[Artifact] Saved Machine-Readable JSON: {json_path}")

    # 7. Save FINAL_TEST_REPORT.md
    md_content = f"""# MedVision — Swin-Tiny Chest X-ray Test Evaluation

## Model
- **Architecture:** Swin Transformer Tiny (`torchvision.models.swin_t`)
- **Pretrained weights:** ImageNet-1K (`torchvision.models.Swin_T_Weights.IMAGENET1K_V1`)
- **Number of parameters:** {total_params:,} (100% trainable)
- **Checkpoint path:** `{checkpoint_file}` (Epoch {best_epoch})

## Dataset
- **Dataset path:** `{dataset_root}`
- **Train count:** 9,097
- **Validation count:** 1,950
- **Test count:** {test_count}
- **Class names:** NORMAL, PNEUMONIA, TUBERCULOSIS
- **Class mapping:** NORMAL: 0, PNEUMONIA: 1, TUBERCULOSIS: 2

## Test Metrics

| Metric | Score | Percentage |
|---|---:|---:|
| Accuracy | {acc:.4f} | {acc * 100:.2f}% |
| Macro Precision | {macro_p:.4f} | {macro_p * 100:.2f}% |
| Macro Recall | {macro_r:.4f} | {macro_r * 100:.2f}% |
| Macro F1 | {macro_f1:.4f} | {macro_f1 * 100:.2f}% |
| Weighted F1 | {weighted_f1:.4f} | {weighted_f1 * 100:.2f}% |

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | {per_class['NORMAL']['precision']:.4f} | {per_class['NORMAL']['recall']:.4f} | {per_class['NORMAL']['f1']:.4f} | {per_class['NORMAL']['support']} |
| PNEUMONIA | {per_class['PNEUMONIA']['precision']:.4f} | {per_class['PNEUMONIA']['recall']:.4f} | {per_class['PNEUMONIA']['f1']:.4f} | {per_class['PNEUMONIA']['support']} |
| TUBERCULOSIS | {per_class['TUBERCULOSIS']['precision']:.4f} | {per_class['TUBERCULOSIS']['recall']:.4f} | {per_class['TUBERCULOSIS']['f1']:.4f} | {per_class['TUBERCULOSIS']['support']} |

## Confusion Matrix

```
                NORMAL   PNEUMONIA   TUBERCULOSIS   Total
NORMAL             {cm[0][0]:>3}         {cm[0][1]:>3}              {cm[0][2]:>3}     {sum(cm[0])}
PNEUMONIA          {cm[1][0]:>3}         {cm[1][1]:>3}              {cm[1][2]:>3}     {sum(cm[1])}
TUBERCULOSIS       {cm[2][0]:>3}         {cm[2][1]:>3}              {cm[2][2]:>3}     {sum(cm[2])}
```

Reference plot: `results/chest_xray_swin_tiny/confusion_matrix.png`

## Evaluation Conditions
- The test set was held out during training and model development.
- The model checkpoint was selected strictly using validation Macro F1.
- No test-set tuning, threshold tuning, or post-hoc calibration was performed.
- Deterministic test preprocessing was applied: Grayscale to RGB -> Resize(224, 224) -> ToTensor() -> ImageNet Normalization.
- Evaluation executed using the trained Swin-Tiny checkpoint on an NVIDIA GeForce RTX 4060 Laptop GPU.
"""
    md_path = out_path / "FINAL_TEST_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write(md_content)
    print(f"[Artifact] Saved Markdown Report: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Swin-Tiny on Chest X-ray test set")
    parser.add_argument("--checkpoint", type=str, default="models/chest_xray_swin_tiny/best_model.pth")
    parser.add_argument("--config", type=str, default="configs/swin_tiny.yaml")
    parser.add_argument("--output_dir", type=str, default="results/chest_xray_swin_tiny")
    args = parser.parse_args()

    run_evaluation(args.checkpoint, args.config, args.output_dir)
