import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, calculate_metrics, plot_confusion_matrix
from src.model import create_vit_b16
from src.dataset import get_dataloaders

def evaluate_test_set(config_path="configs/vit_b16.yaml"):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = PROJECT_ROOT / config["paths"]["best_model_path"]
    report_path = PROJECT_ROOT / config["paths"]["evaluation_report_path"]
    cm_path = PROJECT_ROOT / config["paths"]["confusion_matrix_path"]
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found at: {checkpoint_path}\n"
            "Please train the model before running evaluation."
        )
        
    print("==================================================")
    print("MedVision ViT-B/16: Test Set Final Evaluation")
    print("==================================================")
    print(f"Loading best checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    class_names = checkpoint.get("class_names", config["dataset"]["classes"])
    num_classes = len(class_names)
    
    # Instantiate model without downloading weights since we load state_dict
    model = create_vit_b16(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print("Loading test dataset (held out)...")
    _, _, test_loader, _, _ = get_dataloaders(config)
    print(f"Total test images: {len(test_loader.dataset):,}")
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    print("\nRunning inference over test set...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)
            
    metrics = calculate_metrics(all_targets, all_preds, class_names=class_names)
    
    # Print formatted evaluation summary
    print("\n==================================================")
    print("MEDVISION ViT-B/16 TEST EVALUATION REPORT")
    print("==================================================")
    print(f"Accuracy:         {metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision:  {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:     {metrics['macro_recall']:.4f}")
    print(f"Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:      {metrics['weighted_f1']:.4f}")
    print("--------------------------------------------------")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("--------------------------------------------------")
    for name in class_names:
        c_stats = metrics["per_class"][name]
        print(f"{name:<15} {c_stats['precision']:<12.4f} {c_stats['recall']:<12.4f} {c_stats['f1']:<12.4f} {c_stats['support']:<8}")
    print("==================================================")
    
    # Save JSON report
    report_data = {
        "model": "ViT-B/16",
        "checkpoint_epoch": checkpoint.get("epoch", "N/A"),
        "best_val_macro_f1": checkpoint.get("val_macro_f1", "N/A"),
        "best_val_accuracy": checkpoint.get("val_accuracy", "N/A"),
        "best_val_loss": checkpoint.get("val_loss", "N/A"),
        "test_metrics": metrics
    }
    os.makedirs(report_path.parent, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Full metrics report saved to: {report_path}")
    
    # Save confusion matrix
    plot_confusion_matrix(all_targets, all_preds, class_names, str(cm_path))
    print(f"Confusion matrix plot saved to: {cm_path}")
    
    # Save FINAL_REPORT.md distinguishing VALIDATION from TEST results
    final_report_md = report_path.parent / "FINAL_REPORT.md"
    with open(final_report_md, "w", encoding="utf-8") as f:
        f.write("# MedVision ViT-B/16: Chest X-ray Classification Final Report\n\n")
        f.write("## 1. Executive Summary\n")
        f.write("This report presents the final evaluation of the standalone **Vision Transformer ViT-B/16** model on the 3-class Chest X-ray dataset (`NORMAL`, `PNEUMONIA`, `TUBERCULOSIS`).\n\n")
        f.write("The model was selected based strictly on the highest **Validation Macro F1** during training, completely isolated from the held-out test split.\n\n")
        f.write("---\n\n")
        f.write("## 2. Validation Results (Model Selection Checkpoint)\n")
        f.write(f"- **Best Checkpoint Epoch**: {checkpoint.get('epoch', 'N/A')}\n")
        val_f1_val = checkpoint.get('val_macro_f1', 0.0)
        val_acc_val = checkpoint.get('val_accuracy', 0.0)
        val_loss_val = checkpoint.get('val_loss', 0.0)
        f.write(f"- **Validation Macro F1**: {val_f1_val:.4f}\n")
        f.write(f"- **Validation Accuracy**: {val_acc_val*100:.2f}%\n")
        f.write(f"- **Validation Loss**: {val_loss_val:.4f}\n\n")
        f.write("---\n\n")
        f.write("## 3. Held-Out Test Results (Final Unseen Evaluation)\n")
        f.write(f"- **Test Set Size**: {len(test_loader.dataset):,} images (untouched during training and model selection)\n")
        f.write(f"- **Test Accuracy**: {metrics['accuracy']*100:.2f}%\n")
        f.write(f"- **Test Macro Precision**: {metrics['macro_precision']:.4f}\n")
        f.write(f"- **Test Macro Recall**: {metrics['macro_recall']:.4f}\n")
        f.write(f"- **Test Macro F1**: {metrics['macro_f1']:.4f}\n")
        f.write(f"- **Test Weighted F1**: {metrics['weighted_f1']:.4f}\n\n")
        f.write("### Per-Class Test Performance\n\n")
        f.write("| Class | Precision | Recall | F1-Score | Support |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for name in class_names:
            c = metrics["per_class"][name]
            f.write(f"| **{name}** | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} | {c['support']} |\n")
        f.write("\n---\n\n")
        f.write("## 4. Artifacts & Deliverables\n")
        f.write(f"- **Best Checkpoint**: `{checkpoint_path.as_posix()}`\n")
        f.write(f"- **Evaluation Metrics JSON**: `{report_path.as_posix()}`\n")
        f.write(f"- **Confusion Matrix**: `{cm_path.as_posix()}`\n")
        f.write(f"- **Training Curves**: `{(report_path.parent / 'training_curves.png').as_posix()}`\n")
    print(f"Final comprehensive report saved to: {final_report_md}")
    
    return metrics

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/vit_b16.yaml"
    evaluate_test_set(config_file)
