"""
MedVision Disease Prediction - Model Evaluation on Test Split
Author: K. Gnana Saaketh (ROLE: Deep Learning / Disease Prediction)
Project: Medvision — AI Medical Imaging Assistant
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.config import (
    BEST_MODEL_PATH,
    SPLITS_DIR,
    METRICS_DIR,
    CLASS_NAMES,
    DEVICE
)
from src.utils import save_json, plot_confusion_matrix
from src.dataset import ChestXRayDataset
from src.preprocessing import get_val_test_transforms
from src.model import load_trained_model

def evaluate_model(checkpoint_path=BEST_MODEL_PATH, splits_dir=SPLITS_DIR, device=DEVICE):
    """
    Evaluates the trained MedVision model strictly on the held-out Test split.
    Calculates Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    """
    checkpoint_path = Path(checkpoint_path)
    splits_dir = Path(splits_dir)
    test_csv = splits_dir / "test_split.csv"

    if not checkpoint_path.exists():
        print(f"[Evaluation] Model checkpoint not found at: {checkpoint_path}")
        print("Model not trained yet.")
        return None

    if not test_csv.exists():
        raise FileNotFoundError(f"Test split CSV not found at: {test_csv}. Run data prep first.")

    print("=" * 60)
    print("      MEDVISION DISEASE PREDICTION — TEST EVALUATION      ")
    print("=" * 60)
    print(f"Model Checkpoint : {checkpoint_path}")
    print(f"Test Split File  : {test_csv}")
    print(f"Evaluation Device: {device}")
    print("=" * 60)

    # 1. Load Model
    model = load_trained_model(checkpoint_path=checkpoint_path, device=device)
    model.eval()

    # 2. Load Test Dataset
    test_df = pd.read_csv(test_csv)
    test_dataset = ChestXRayDataset(test_df, transform=get_val_test_transforms())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_preds = []
    all_targets = []
    all_probs = []

    print(f"Evaluating across {len(test_dataset)} test chest radiographs...")
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 3. Calculate Clinical Classification Metrics
    accuracy = float(accuracy_score(all_targets, all_preds))
    precision_macro = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    recall_macro = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    f1_macro = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    
    precision_weighted = float(precision_score(all_targets, all_preds, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(all_targets, all_preds, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))

    cm = confusion_matrix(all_targets, all_preds)
    report_dict = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision_macro, 4),
        "recall": round(recall_macro, 4),
        "f1_score": round(f1_macro, 4),
        "weighted_metrics": {
            "precision_weighted": round(precision_weighted, 4),
            "recall_weighted": round(recall_weighted, 4),
            "f1_weighted": round(f1_weighted, 4)
        },
        "per_class_metrics": {
            cls_name: {
                "precision": round(report_dict[cls_name]["precision"], 4),
                "recall": round(report_dict[cls_name]["recall"], 4),
                "f1_score": round(report_dict[cls_name]["f1-score"], 4),
                "support": int(report_dict[cls_name]["support"])
            }
            for cls_name in CLASS_NAMES
        },
        "test_sample_count": len(test_targets if 'test_targets' in locals() else all_targets),
        "confusion_matrix": cm.tolist()
    }

    # 4. Save Metrics & Plot
    metrics_path = METRICS_DIR / "test_metrics.json"
    save_json(metrics, metrics_path)
    cm_path = plot_confusion_matrix(cm, CLASS_NAMES)

    print("\n--- TEST EVALUATION RESULTS ---")
    print(f"Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision : {precision_macro*100:.2f}% (Macro)")
    print(f"Recall    : {recall_macro*100:.2f}% (Macro)")
    print(f"F1-Score  : {f1_macro:.4f} (Macro)")
    print("-" * 35)
    print("Per-Class Breakdown:")
    for cls_name in CLASS_NAMES:
        c_stats = metrics["per_class_metrics"][cls_name]
        print(f"  [{cls_name:10s}] Precision: {c_stats['precision']*100:5.2f}% | Recall: {c_stats['recall']*100:5.2f}% | F1: {c_stats['f1_score']:.4f} (n={c_stats['support']})")
    print("=" * 60)
    print(f"Metrics saved to : {metrics_path}")
    print(f"Plot saved to    : {cm_path}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MedVision Disease Prediction Model on Test Split")
    parser.add_argument("--checkpoint", type=str, default=str(BEST_MODEL_PATH), help="Path to checkpoint")
    args = parser.parse_args()

    evaluate_model(checkpoint_path=args.checkpoint)
