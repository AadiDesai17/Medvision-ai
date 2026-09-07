# MedVision — Swin-Tiny Chest X-ray Classification Experiment

## Project Status: Stage 3 Completed (Full Model Training)
The standalone training run for **Swin Transformer Tiny (Swin-Tiny)** on the 3-class Chest X-ray dataset completed with 100% success across all 15 epochs.

- **Best Epoch:** Epoch 15
- **Best Validation Macro F1:** **0.9930**
- **Best Validation Accuracy:** **99.23%** (1,935 / 1,950 images correct)
- **Best Checkpoint:** `models/chest_xray_swin_tiny/best_model.pth` (315.39 MB)
- **Held-out Test Split:** Completely untouched during training (reserved for Stage 4)

---

## 1. Project Overview
This project is an isolated experiment training and evaluating the **Swin Transformer Tiny (Swin-Tiny)** architecture on the Chest X-ray dataset for 3-class disease classification:
1. `NORMAL` (Index: `0`)
2. `PNEUMONIA` (Index: `1`)
3. `TUBERCULOSIS` (Index: `2`)

This experiment forms part of a 4-architecture benchmark alongside ResNet50, ViT-B/16, and ConvNeXt-Tiny under strict fair-comparison constraints.

---

## 2. Directory Structure
```
medvision_swin_tiny/
├── src/
│   ├── model.py                  # Swin-Tiny model factory with dynamic classifier head replacement
│   ├── dataset.py                # Dataset loader, RGB conversion, transforms, and class weight calculator
│   ├── train.py                  # Full training pipeline (AMP FP16, early stopping, CosineAnnealingLR)
│   ├── evaluate.py               # Test split evaluation script (Accuracy, Macro/Weighted F1, confusion matrix)
│   ├── utils.py                  # Reproducibility seed, metrics, plotting, checkpoint save/load
│   └── audit_dataset.py          # Stage 1 dataset integrity audit script
├── configs/
│   ├── swin_tiny.yaml            # YAML configuration for training
│   └── baseline_config.json      # JSON baseline parameters
├── models/
│   └── chest_xray_swin_tiny/
│       └── best_model.pth        # Saved best model checkpoint (Epoch 15, 315.39 MB)
├── results/
│   └── chest_xray_swin_tiny/
│       ├── training_curves.png   # Loss and Validation Macro F1 curves
│       ├── training_history.json # Epoch-by-epoch training metrics
│       └── training_summary.json # Final training summary statistics
├── predict_swin_tiny.py          # Single-image inference CLI
├── smoke_test.py                 # Real-data GPU forward/backward verification script
├── requirements.txt              # Project package requirements
└── README.md                     # Project documentation & audit report
```

---

## 3. Dataset Audit & Class Weights
- **Dataset Path:** `C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia` (Referenced in-place, strictly read-only)
- **Split Distribution:**
  - **Train:** NORMAL: 3,911 | PNEUMONIA: 2,971 | TUBERCULOSIS: 2,215 (Total: 9,097)
  - **Validation:** NORMAL: 838 | PNEUMONIA: 637 | TUBERCULOSIS: 475 (Total: 1,950)
  - **Test (Held-out):** NORMAL: 839 | PNEUMONIA: 637 | TUBERCULOSIS: 475 (Total: 1,951)
- **Programmatic Class Weights (Moderated Square-Root Inverse-Frequency):**
  - `NORMAL`: `0.8630`
  - `PNEUMONIA`: `0.9902`
  - `TUBERCULOSIS`: `1.1468`

---

## 4. Training Configuration & Hyperparameters
- **Architecture:** Swin Transformer Tiny (`torchvision.models.swin_t`)
- **Pretrained Weights:** ImageNet-1K (`torchvision.models.Swin_T_Weights.IMAGENET1K_V1`)
- **Classifier Input Dimension:** 768 (`model.head.in_features`)
- **Total Parameters:** 27,521,661 (100% trainable)
- **Input Resolution:** 224 x 224 (grayscale converted to RGB before ImageNet normalization)
- **Optimizer:** AdamW (lr=3e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
- **Scheduler:** CosineAnnealingLR (T_max=15, eta_min=1e-6)
- **Batching:** Physical Batch Size 16, Gradient Accumulation 2 (Effective Batch Size 32)
- **Mixed Precision:** CUDA AMP FP16 with safe GradScaler (`init_scale=1024.0`)
- **Gradient Clipping:** Max norm 1.0
- **Random Seed:** 42

---

## 5. Epoch-by-Epoch Training Results
| Epoch | Train Loss | Val Loss | Val Accuracy | Val Macro F1 | Learning Rate | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 0.1668 | 0.0726 | 0.9790 | 0.9808 | 3.00e-05 | New Best |
| 02 | 0.0803 | 0.0651 | 0.9805 | 0.9816 | 2.97e-05 | New Best |
| 03 | 0.0612 | 0.0740 | 0.9846 | 0.9861 | 2.87e-05 | New Best |
| 04 | 0.0433 | 0.0534 | 0.9877 | 0.9888 | 2.72e-05 | New Best |
| 05 | 0.0446 | 0.0455 | 0.9892 | 0.9903 | 2.52e-05 | New Best |
| 06 | 0.0370 | 0.0654 | 0.9862 | 0.9874 | 2.28e-05 | — |
| 07 | 0.0267 | 0.0761 | 0.9877 | 0.9887 | 2.00e-05 | — |
| 08 | 0.0239 | 0.0579 | 0.9913 | 0.9921 | 1.70e-05 | New Best |
| 09 | 0.0180 | 0.0640 | 0.9913 | 0.9920 | 1.40e-05 | — |
| 10 | 0.0167 | 0.0716 | 0.9913 | 0.9920 | 1.10e-05 | — |
| 11 | 0.0126 | 0.0888 | 0.9887 | 0.9898 | 8.25e-06 | — |
| 12 | 0.0098 | 0.0670 | 0.9908 | 0.9916 | 5.80e-06 | — |
| 13 | 0.0062 | 0.0646 | 0.9923 | 0.9929 | 3.77e-06 | New Best |
| 14 | 0.0084 | 0.0646 | 0.9918 | 0.9925 | 2.25e-06 | — |
| 15 | 0.0069 | 0.0634 | 0.9923 | 0.9930 | 1.32e-06 | **Final Best Checkpoint** |

---

## 6. Best Validation Metrics Summary (Epoch 15)
- **Validation Accuracy:** **99.23%** (1,935 / 1,950 correct)
- **Validation Macro Precision:** **99.34%**
- **Validation Macro Recall:** **99.26%**
- **Validation Macro F1:** **0.9930**
- **Validation Weighted F1:** **0.9923**
- **Validation Loss:** `0.0634`
- **Per-Class Validation Performance:**
  - `NORMAL`: Precision: 98.81% | Recall: 99.40% | F1: 99.11% (833/838 correct)
  - `PNEUMONIA`: Precision: 99.21% | Recall: 98.59% | F1: 98.90% (628/637 correct)
  - `TUBERCULOSIS`: Precision: 100.00% | Recall: 99.79% | F1: 99.89% (474/475 correct)
- **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (Peak VRAM: 1.55 GB)
- **Total Training Duration:** 20.32 minutes
