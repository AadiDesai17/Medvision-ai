# MedVision ConvNeXt-Tiny: Chest X-Ray 3-Class Disease Prediction

Standardized deep learning pipeline for triaging Chest X-rays into **NORMAL**, **PNEUMONIA**, and **TUBERCULOSIS** using the modern convolutional architecture **ConvNeXt-Tiny** (`torchvision.models.convnext_tiny`).

---

## Project Overview

This project is an isolated, reproducible implementation following the standardized MedVision experimental benchmarking protocol. It provides strict isolation from prior experiments (ResNet50, ViT-B/16, Swin-Tiny) and enforces rigorous image-level separation and audit compliance.

- **Workspace:** `C:\Users\AADI\.gemini\antigravity\scratch\medvision_convnext_tiny`
- **Dataset (Read-Only):** `C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia`
- **Classes:** `NORMAL` (0), `PNEUMONIA` (1), `TUBERCULOSIS` (2)
- **Cohort Size:** 12,998 total radiographic images (512x512 PNG, grayscale)

---

## Standardized Protocol Specifications

| Hyperparameter / Component | Specification |
|:---|:---|
| **Architecture** | `torchvision.models.convnext_tiny` |
| **Pretrained Weights** | `ConvNeXt_Tiny_Weights.DEFAULT` (ImageNet-1K v1) |
| **Classifier Head** | `LayerNorm2d(768) -> Flatten() -> Linear(768, 3)` |
| **Total Parameters** | `27,822,435` (27.82M) |
| **Trainable Parameters** | `27,822,435` (100% trainable) |
| **Input Resolution** | `224 x 224` |
| **Input Channels** | `3` (Grayscale replicated to RGB `(I, I, I)` in-memory via `SafeRGB`) |
| **Normalization** | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| **Train Augmentations** | `RandomRotation(7 degrees)`, `ColorJitter(brightness=0.05, contrast=0.05)` |
| **Prohibited Augmentations** | `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomResizedCrop` |
| **Eval Augmentations** | Deterministic: `SafeRGB -> Resize(224, 224) -> ToTensor -> Normalize` |
| **Class Weights** | Moderated square-root inverse frequency: `[0.8630, 0.9902, 1.1468]` |
| **Loss Function** | Weighted `nn.CrossEntropyLoss`, `label_smoothing=0.0` |
| **Optimizer** | `AdamW` (lr=`3e-5`, weight_decay=`0.01`, betas=`(0.9, 0.999)`, eps=`1e-8`) |
| **Scheduler** | `CosineAnnealingLR` ($T_{\max}=15$, $\eta_{\min}=10^{-6}$) |
| **Batch Size** | Physical: 16, Gradient Accumulation: 2 -> Effective: 32 |
| **Mixed Precision** | CUDA FP16 AMP (`torch.amp.autocast('cuda')`), `GradScaler(init_scale=1024)` |
| **Gradient Clipping** | `max_norm = 1.0` |
| **Training Budget** | Maximum 15 epochs, Early stopping patience = 5 epochs |
| **Model Selection** | **Validation Macro F1 ONLY** |
| **Deterministic Seed** | `42` |

---

## Directory Structure

```
medvision_convnext_tiny/
├── configs/
│   └── convnext_tiny.yaml             # Complete standardized configuration
├── models/
│   └── chest_xray_convnext_tiny/     # Checkpoint destination (best_model.pth)
├── results/
│   └── chest_xray_convnext_tiny/     # Metrics, confusion matrix, audit reports
├── src/
│   ├── model.py                       # ConvNeXt-Tiny model definition and head
│   ├── dataset.py                     # SafeRGB transform, PyTorch dataset & loaders
│   ├── utils.py                       # Metrics, seed, plots, serialization
│   ├── train.py                       # Standardized training pipeline (with Stage 1 safety guard)
│   └── evaluate.py                    # Test set evaluation & diagnostic error analysis
├── predict_convnext_tiny.py           # Standalone single-image CLI inference tool
├── smoke_test.py                      # Hardware & non-training verification script
├── requirements.txt                   # Environment dependencies
└── README.md                          # Project documentation
```

---

## Stage 1 Verification & Status

Stage 1 Audit & Preparation is complete and verified:
- Dataset audited: 12,998 images, 100% PNG, 100% 512x512 grayscale, zero corrupt/zero-byte files, zero cross-split duplicate hashes.
- Hardware verified: NVIDIA GeForce RTX 4060 Laptop GPU (7.996 GB VRAM, Ada Lovelace).
- Peak VRAM in smoke test: **1,139.89 MB** (well below 8 GB ceiling).
- Full audit report located at: `results/chest_xray_convnext_tiny/STAGE1_AUDIT_REPORT.md`.

---

## Usage Instructions

### 1. Run Smoke Test & Hardware Verification
```bash
python smoke_test.py
```

### 2. Single Image Inference CLI
```bash
python predict_convnext_tiny.py --image "path/to/xray.png"
# Or run on a built-in test sample:
python predict_convnext_tiny.py --demo
```

### 3. Full Training (Stage 3 — requires approval)
```bash
python src/train.py --execute-training
```

### 4. Held-Out Test Split Evaluation (Stage 4 — requires approval)
```bash
python src/evaluate.py --execute-test-eval
```
