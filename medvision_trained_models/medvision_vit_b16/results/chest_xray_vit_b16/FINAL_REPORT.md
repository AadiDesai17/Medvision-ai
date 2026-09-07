# MedVision ViT-B/16: Chest X-ray Classification Final Report

## 1. Executive Summary
This report presents the final evaluation of the standalone **Vision Transformer ViT-B/16** model on the 3-class Chest X-ray dataset (`NORMAL`, `PNEUMONIA`, `TUBERCULOSIS`).

The model was selected based strictly on the highest **Validation Macro F1** during training, completely isolated from the held-out test split.

---

## 2. Validation Results (Model Selection Checkpoint)
- **Best Checkpoint Epoch**: 5
- **Validation Macro F1**: 0.9902
- **Validation Accuracy**: 98.92%
- **Validation Loss**: 0.0502

---

## 3. Held-Out Test Results (Final Unseen Evaluation)
- **Test Set Size**: 1,951 images (untouched during training and model selection)
- **Test Accuracy**: 98.97%
- **Test Macro Precision**: 0.9906
- **Test Macro Recall**: 0.9905
- **Test Macro F1**: 0.9905
- **Test Weighted F1**: 0.9897

### Per-Class Test Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **NORMAL** | 0.9869 | 0.9893 | 0.9881 | 839 |
| **PNEUMONIA** | 0.9890 | 0.9843 | 0.9866 | 637 |
| **TUBERCULOSIS** | 0.9958 | 0.9979 | 0.9968 | 475 |

---

## 4. Artifacts & Deliverables
- **Best Checkpoint**: `C:/Users/AADI/.gemini/antigravity/scratch/medvision_vit_b16/models/chest_xray_vit_b16/best_model.pth`
- **Evaluation Metrics JSON**: `C:/Users/AADI/.gemini/antigravity/scratch/medvision_vit_b16/results/chest_xray_vit_b16/test_evaluation_report.json`
- **Confusion Matrix**: `C:/Users/AADI/.gemini/antigravity/scratch/medvision_vit_b16/results/chest_xray_vit_b16/confusion_matrix.png`
- **Training Curves**: `C:/Users/AADI/.gemini/antigravity/scratch/medvision_vit_b16/results/chest_xray_vit_b16/training_curves.png`
