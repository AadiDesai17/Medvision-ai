# MedVision ViT-B/16: Chest X-ray Classification

## 1. Standalone Project Overview & Purpose
This repository contains a standalone, fully isolated machine learning experiment fine-tuning **Vision Transformer (ViT-B/16)** for multi-class Chest X-ray disease classification:
1. `NORMAL`
2. `PNEUMONIA`
3. `TUBERCULOSIS`

> **Isolation Notice**: This experiment is completely independent of the previous MedVision ResNet-50 project located in `medvision_disease_prediction`. It uses its own configuration, modular source code, checkpoint directory, and evaluation outputs to prevent cross-contamination while strictly preserving the identical predefined dataset splits for future fair benchmarking against ResNet-50, Swin-Tiny, and ConvNeXt-Tiny.

---

## 2. Dataset Configuration & Audit Results
- **Dataset Path**: `C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia`
- **Data Integrity**: Referenced in-place via configuration; never duplicated or modified.
- **Image Modality**: 100% single-channel Grayscale (`L`), 512x512 resolution, PNG format (`.png`).
- **Audit Findings**: 0 corrupt files, 0 zero-byte files, 0 unreadable files across all 12,998 images.
- **Split Distribution (Audited & Verified)**:
  - **TRAIN (9,097 images)**:
    - NORMAL: 3,911 (42.99%)
    - PNEUMONIA: 2,971 (32.66%)
    - TUBERCULOSIS: 2,215 (24.35%)
  - **VAL (1,950 images)**:
    - NORMAL: 838 (42.97%)
    - PNEUMONIA: 637 (32.67%)
    - TUBERCULOSIS: 475 (24.36%)
  - **TEST (1,951 images)**:
    - NORMAL: 839 (43.00%)
    - PNEUMONIA: 637 (32.65%)
    - TUBERCULOSIS: 475 (24.35%)
- **Patient Separation Disclosure**: Filenames across the open-source dataset lack consistent standardized patient identifiers. Patient-level separation is therefore **unverified and not claimed**. Predefined splits are strictly respected.

---

## 3. Model Architecture & Parameters
- **Backbone**: Vision Transformer Base with 16x16 patch projection (`vit_b_16`) from `torchvision.models`.
- **Pretrained Weights**: `ViT_B_16_Weights.DEFAULT` (`ViT_B_16_Weights.IMAGENET1K_V1`).
- **Classification Head**: Custom linear projection replacing `model.heads.head`:
  ```python
  model.heads.head = nn.Linear(in_features=768, out_features=3)
  ```
- **Total Parameters**: 85,800,963
- **Trainable Parameters**: 85,800,963 (Full fine-tuning)

---

## 4. Input Preprocessing & Augmentation Pipeline
Because ViT-B/16 was pretrained on 3-channel ImageNet RGB images:
- **Grayscale Handling**: Converted safely from Grayscale (`'L'`) to 3-channel (`'RGB'`) via `PIL.Image.convert('RGB')` upon loading.
- **Training Pipeline**:
  1. `Resize((224, 224), interpolation=InterpolationMode.BILINEAR)`
  2. `RandomRotation(degrees=7)` (medically conservative subtle tilt)
  3. `ColorJitter(brightness=0.05, contrast=0.05)` (subtle exposure variation)
  4. `ToTensor()`
  5. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
  *(Vertical and horizontal flips are strictly omitted to maintain cardiac silhouette and anatomical laterality).*
- **Validation, Test & Inference Pipeline (Strictly Deterministic)**:
  1. `Resize((224, 224), interpolation=InterpolationMode.BILINEAR)`
  2. `ToTensor()`
  3. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

---

## 5. Hardware, CUDA & Environment Specifications
- **Operating System**: Windows 11
- **Python**: 3.14.4 (64-bit)
- **PyTorch**: 2.13.0+cu126
- **torchvision**: 0.28.0+cu126
- **CUDA Device**: NVIDIA GeForce RTX 4060 Laptop GPU (8.00 GB VRAM, Compute Capability 8.9)
- **Automatic Mixed Precision (AMP)**: `torch.amp.autocast('cuda')` with `torch.amp.GradScaler('cuda', init_scale=1024.0)`.
  *(Calibrated `init_scale=1024.0` prevents FP16 gradient overflow caused by high-magnitude ViT pre-logits).*
- **Smoke-Test VRAM Footprint**: Peak memory allocated was 1.62 GB (20.3% of 8 GB VRAM).

---

## 6. Training Strategy & Hyperparameters
- **Physical Batch Size**: 16
- **Gradient Accumulation Steps**: 2 (Effective Batch Size: 32)
- **Optimizer**: `AdamW` (learning rate: `3e-5`, weight decay: `0.01`, betas: `(0.9, 0.999)`, eps: `1e-8`)
- **LR Scheduler**: `CosineAnnealingLR` ($T_{\max}=15, \eta_{\min}=10^{-6}$)
- **Class Imbalance Strategy**: Moderated square-root inverse-frequency weights derived strictly from the training split:
  - `NORMAL`: 0.8631
  - `PNEUMONIA`: 0.9902
  - `TUBERCULOSIS`: 1.1468
- **Loss Function**: `nn.CrossEntropyLoss(weight=class_weights)`
- **Model Selection**: Checkpointed on highest **Validation Macro F1** (evaluated after every epoch).
- **Early Stopping**: Patience = 5 epochs monitoring Validation Macro F1.
- **Maximum Epochs**: 15 epochs.
- **Reproducibility**: Seed fixed to `42` across Python, NumPy, and PyTorch CUDA deterministic backend.

---

## 7. Project Directory Layout
```
medvision_vit_b16/
├── configs/
│   └── vit_b16.yaml             # Centralized YAML configuration
├── src/
│   ├── dataset.py               # Dataset and transforms implementation
│   ├── model.py                 # ViT-B/16 architecture builder
│   ├── train.py                 # Mixed-precision training loop with Macro F1 checkpointing
│   ├── evaluate.py              # Final test set evaluation and report generator
│   └── utils.py                 # Metrics, plotting, and helper functions
├── models/
│   └── chest_xray_vit_b16/
│       ├── best_model.pth       # Best model checkpoint (saved during training)
│       └── config.json          # Checkpoint configuration and training metadata
├── results/
│   └── chest_xray_vit_b16/
│       ├── test_evaluation_report.json # Final evaluation metrics
│       ├── confusion_matrix.png        # Confusion matrix heatmap
│       └── training_curves.png         # Loss and Macro F1 curves
├── predict_vit_b16.py           # Interactive demo inference script
├── smoke_test.py                # Pre-training GPU validation script
├── requirements.txt             # Minimal dependencies
└── README.md                    # Documentation
```

---

## 8. Usage Commands

### 1. Environment Verification / Smoke Test
```powershell
python smoke_test.py
```
*(Validates CUDA, loads weights, executes forward/backward pass with AMP, checks gradients, and measures VRAM).*

### 2. Full Training Run
```powershell
python src/train.py
```
*(Executes 15-epoch training loop with live metrics logging and saves best checkpoint based on Validation Macro F1).*

### 3. Evaluation on Held-out Test Set
```powershell
python src/evaluate.py
```
*(Evaluates best checkpoint on untouched 1,951 test images; produces `test_evaluation_report.json` and `confusion_matrix.png`).*

### 4. Interactive Demonstration / Prediction
```powershell
python predict_vit_b16.py
```
Prompts:
`Enter path to Chest X-ray image:`
Outputs real classification probabilities and prediction:
```
========================================
MedVision - ViT-B/16
Chest X-ray Classifier
========================================

Image: jtiptj_test_person100_bacteria_477.png

Prediction: PNEUMONIA
Confidence: 98.45%

Class probabilities:
NORMAL:          0.82%
PNEUMONIA:      98.45%
TUBERCULOSIS:    0.73%

========================================
```
