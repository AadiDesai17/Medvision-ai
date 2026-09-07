# Stage 3 Full Training Report: Standardized ResNet50

**Project:** MedVision Chest X-ray 3-Class Disease Prediction  
**Architecture:** ResNet50 (`torchvision.models.resnet50`)  
**Stage:** Stage 3 — Full Standardized ResNet50 Retraining  
**Timestamp:** 2026-09-07  
**Status:** COMPLETE & VERIFIED  

---

## 1. Hardware Environment
- **Device:** `NVIDIA GeForce RTX 4060 Laptop GPU`
- **Total Dedicated VRAM:** `7.996 GB` (~8,188 MB)
- **Compute Capability:** `(8, 9)` (Ada Lovelace architecture)
- **Host Architecture:** x86_64 (AMD64) Windows

---

## 2. Software Versions
- **Python:** `3.14.4` (tags/v3.14.4:23116f9)
- **PyTorch:** `2.13.0+cu126`
- **torchvision:** `0.28.0+cu126`
- **CUDA Driver / Runtime:** CUDA 12.6

---

## 3. Dataset Counts
- **Dataset Path:** `C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia` (Accessed strictly read-only)
- **Diagnostic Categories:**
  - `0`: `NORMAL` (Healthy / clear lung fields)
  - `1`: `PNEUMONIA` (Acute alveolar / interstitial consolidation)
  - `2`: `TUBERCULOSIS` (Active / cavitary mycobacterial infiltrates)
- **Training Cohort:** **9,097 images**
  - `NORMAL`: 3,911 (42.99%)
  - `PNEUMONIA`: 2,971 (32.66%)
  - `TUBERCULOSIS`: 2,215 (24.35%)
- **Validation Cohort:** **1,950 images**
  - `NORMAL`: 838 (42.97%)
  - `PNEUMONIA`: 637 (32.67%)
  - `TUBERCULOSIS`: 475 (24.36%)
- **Held-Out Test Cohort:** **1,951 images** (STRICTLY UNTOUCHED — NOT ACCESSED OR EVALUATED)

---

## 4. Model Architecture
- **Backbone Network:** Deep 50-layer Residual Network (`torchvision.models.resnet50`)
- **Pretrained Weights:** Official ImageNet-1K V2 weights via `torchvision.models.ResNet50_Weights.DEFAULT`
- **Feature Compression:** Global Adaptive Average Pooling compressing activations to a 2048-dimensional embedding
- **Classification Head (`model.fc`):**
  ```python
  nn.Sequential(
      nn.Dropout(p=0.2),
      nn.Linear(in_features=2048, out_features=3)
  )
  ```
- **Explainability Target Layer:** `model.layer4[-1]` (final residual Bottleneck block), hooked for Grad-CAM

---

## 5. Parameter Count
- **Total Parameters:** `23,514,179` (23.51M)
- **Trainable Parameters:** `23,514,179` (100.0% trainable end-to-end)
- **Frozen Parameters:** `0`
- **Classification Head Parameters:** `6,147` ($2048 \times 3 + 3 = 6,147$)

---

## 6. Preprocessing & Data Augmentation
- **Grayscale Conversion:** In-memory `SafeRGB` channel replication ($(I, I, I)$). Source dataset files on disk remain strictly unmodified.
- **Training Preprocessing:**
  1. `SafeRGB()`
  2. `transforms.Resize((224, 224))`
  3. `transforms.RandomRotation(degrees=7)`
  4. `transforms.ColorJitter(brightness=0.05, contrast=0.05)`
  5. `transforms.ToTensor()`
  6. `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- **Strictly Prohibited Augmentations:**
  - `RandomHorizontalFlip`: Disabled
  - `RandomVerticalFlip`: Disabled
  - `RandomResizedCrop`: Disabled
- **Validation Preprocessing:**
  1. `SafeRGB()`
  2. `transforms.Resize((224, 224))`
  3. `transforms.ToTensor()`
  4. `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

---

## 7. Class Weights
Moderated square-root inverse-frequency weights computed strictly from training partition counts:
- **`NORMAL` (Class 0):** `0.8630`
- **`PNEUMONIA` (Class 1):** `0.9902`
- **`TUBERCULOSIS` (Class 2):** `1.1468`

---

## 8. Loss Function
- **Criterion:** Weighted Cross-Entropy Loss (`nn.CrossEntropyLoss(weight=class_weights)`)
- **Label Smoothing:** **STRICTLY 0.0 (NO label smoothing)**

---

## 9. Optimizer
- **Optimizer:** Decoupled Weight Decay AdamW (`torch.optim.AdamW`)
- **Base Learning Rate:** `3e-5` (`0.00003`)
- **Weight Decay:** `0.01`
- **Betas:** `(0.9, 0.999)`
- **Epsilon:** `1e-8`

---

## 10. Learning Rate Scheduler
- **Scheduler:** `CosineAnnealingLR`
- **Cycle Period ($T_{\max}$):** `15` epochs
- **Minimum Learning Rate ($\eta_{\min}$):** `1e-6` (`0.000001`)

---

## 11. Batch Size
- **Physical Batch Size:** `16` samples per GPU forward/backward step

---

## 12. Gradient Accumulation
- **Accumulation Steps:** `2`
- **Effective Batch Size:** `32` samples ($16 \times 2 = 32$)

---

## 13. AMP Configuration & Gradient Clipping
- **Mixed Precision:** CUDA FP16 Automatic Mixed Precision (`torch.amp.autocast('cuda', dtype=torch.float16)`)
- **GradScaler:** `torch.amp.GradScaler('cuda', init_scale=1024)`
- **Gradient Clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` applied after gradient unscaling

---

## 14. Deterministic Seed
- **Seed Value:** `42`
- **Controls Enforced:** `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, `np.random.seed(42)`, `random.seed(42)`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`

---

## 15. Epoch-by-Epoch Training Trajectory

| Epoch | Learning Rate | Train Loss | Train Acc (%) | Train Macro F1 | Val Loss | Val Acc (%) | Val Macro Prec | Val Macro Rec | Val Macro F1 | Duration (s) | Peak VRAM (MB) | Best Checkpoint |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **01** | $3.00 \times 10^{-5}$ | 0.3536 | 89.47% | 0.8959 | 0.0773 | 97.69% | 0.9797 | 0.9774 | **0.9785** | 99.9s | 1120.7 | ⭐ Saved (New Best) |
| **02** | $2.97 \times 10^{-5}$ | 0.1019 | 96.61% | 0.9677 | 0.0589 | 98.26% | 0.9853 | 0.9821 | **0.9836** | 161.8s | 1120.7 | ⭐ Saved (New Best) |
| **03** | $2.87 \times 10^{-5}$ | 0.0659 | 97.71% | 0.9784 | 0.0536 | 98.36% | 0.9862 | 0.9835 | **0.9848** | 116.2s | 1120.7 | ⭐ Saved (New Best) |
| **04** | $2.72 \times 10^{-5}$ | 0.0511 | 98.22% | 0.9832 | 0.0537 | 98.56% | 0.9873 | 0.9861 | **0.9867** | 112.6s | 1120.7 | ⭐ Saved (New Best) |
| **05** | $2.52 \times 10^{-5}$ | 0.0414 | 98.74% | 0.9882 | 0.0509 | 98.67% | 0.9871 | 0.9877 | **0.9874** | 125.4s | 1120.7 | ⭐ Saved (New Best) |
| **06** | $2.28 \times 10^{-5}$ | 0.0299 | 99.02% | 0.9910 | 0.0637 | 98.51% | 0.9872 | 0.9854 | 0.9863 | 118.8s | 1120.7 | Patience: 1/5 |
| **07** | $2.00 \times 10^{-5}$ | 0.0205 | 99.38% | 0.9942 | 0.0548 | 98.87% | 0.9898 | 0.9892 | **0.9895** | 124.0s | 1120.7 | ⭐ Saved (New Best) |
| **08** | $1.70 \times 10^{-5}$ | 0.0130 | 99.57% | 0.9960 | 0.0605 | 98.62% | 0.9876 | 0.9868 | 0.9872 | 123.2s | 1120.7 | Patience: 1/5 |
| **09** | $1.40 \times 10^{-5}$ | 0.0115 | 99.66% | 0.9968 | 0.0665 | 98.82% | 0.9891 | 0.9891 | 0.9891 | 121.4s | 1120.7 | Patience: 2/5 |
| **10** | $1.10 \times 10^{-5}$ | 0.0092 | 99.68% | 0.9971 | 0.0654 | 98.77% | 0.9889 | 0.9883 | 0.9886 | 122.5s | 1120.7 | Patience: 3/5 |
| **11** | $8.25 \times 10^{-6}$ | 0.0075 | 99.77% | 0.9979 | 0.0649 | 98.51% | 0.9876 | 0.9850 | 0.9863 | 126.1s | 1120.7 | Patience: 4/5 |
| **12** | $5.80 \times 10^{-6}$ | 0.0023 | 99.92% | 0.9993 | 0.0626 | 98.92% | 0.9899 | 0.9901 | **0.9900** | 120.6s | 1120.7 | ⭐ Saved (New Best) |
| **13** | $3.77 \times 10^{-6}$ | 0.0066 | 99.79% | 0.9981 | 0.0570 | 98.97% | 0.9907 | 0.9901 | **0.9904** | 96.1s | 1120.7 | 🏆 **BEST CHECKPOINT** |
| **14** | $2.25 \times 10^{-6}$ | 0.0032 | 99.92% | 0.9993 | 0.0596 | 98.92% | 0.9903 | 0.9896 | 0.9900 | 63.6s | 1120.7 | Patience: 1/5 |
| **15** | $1.32 \times 10^{-6}$ | 0.0040 | 99.89% | 0.9990 | 0.0578 | 98.92% | 0.9903 | 0.9896 | 0.9900 | 64.0s | 1120.7 | Patience: 2/5 |

---

## 16. Best Epoch
- **Optimal Epoch Selected:** **Epoch 13**
- **Trigger Criterion:** Strict peak in **Validation Macro F1** ($0.9904$)
- **Validation Loss at Best Epoch:** `0.0570`
- **Validation Accuracy at Best Epoch:** `98.97%` (1,930 / 1,950 images correct)

---

## 17. Best Validation Macro F1
- **Peak Validation Macro F1:** **`0.9904`** (99.04%)
- **Validation Macro Precision:** `0.9907` (99.07%)
- **Validation Macro Recall:** `0.9901` (99.01%)
- **Validation Weighted F1:** `0.9897` (98.97%)

---

## 18. Early Stopping Status
- **Patience Configured:** `5` consecutive epochs without Macro F1 improvement
- **Trigger Status:** **Not triggered** (full 15 epochs completed cleanly)
- **Convergence Behavior:** Model reached a preliminary peak at Epoch 07 ($0.9895$), endured a 4-epoch plateau without tripping early stopping, broke through to new peaks at Epoch 12 ($0.9900$) and Epoch 13 ($0.9904$), and finished at Epoch 15 with patience counter at 2/5.

---

## 19. Total Training Duration
- **Total Clock Time:** **28.30 minutes** (1,698 seconds)
- **Average Epoch Duration:** ~113.2 seconds / epoch

---

## 20. Peak GPU Memory
- **Peak VRAM Allocated:** **`1120.73 MB`** (~1.12 GB)
- **Headroom Remaining:** Over `6.87 GB` available out of 8.00 GB total capacity on the RTX 4060 GPU
- **Execution Stability:** Zero OOM events, zero gradient overflow skips

---

## 21. Checkpoint Path & Generated Artifacts
- **Optimal Checkpoint Saved:** `models/chest_xray_resnet50/best_model.pth`
  - Checkpoint File Size: `94,375,761 bytes` (90.00 MB)
  - Modification Timestamp: `Mon Sep 7 14:35:11 2026`
- **Preserved Historical Checkpoint:** `models/chest_xray_resnet50/historical_best_model.pth` (Intact & untouched)
- **Training History Log:** `results/chest_xray_resnet50/training_history.json`
- **Training Summary Log:** `results/chest_xray_resnet50/training_summary.json`
- **Training Configuration Snapshot:** `results/chest_xray_resnet50/training_config.json`
- **Convergence Curves Plot:** `results/chest_xray_resnet50/training_curves.png`

---

## 22. Strict Scientific Disclaimer

> [!IMPORTANT]
> **Held-Out Test Set Untouched:**
> As mandated by the Stage 3 protocol, the 1,951 images in the held-out test split were **STRICTLY NOT EVALUATED**.
> Zero inferences or metric calculations were performed on the test partition during training or model selection. The test set remains completely unexposed and unbiased for Stage 4 evaluation.

---

STAGE 3 COMPLETE — FULL TRAINING FINISHED — HELD-OUT TEST NOT EVALUATED.
