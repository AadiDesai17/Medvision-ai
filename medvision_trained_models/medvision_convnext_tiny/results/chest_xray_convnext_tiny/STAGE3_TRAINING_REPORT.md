# Stage 3 Full Training Report: Standardized ConvNeXt-Tiny

**Project:** MedVision Chest X-ray 3-Class Disease Prediction  
**Architecture:** ConvNeXt-Tiny (`torchvision.models.convnext_tiny`)  
**Stage:** Stage 3 — Full Standardized ConvNeXt-Tiny Retraining  
**Timestamp:** 2026-09-07  
**Status:** COMPLETE & VERIFIED  

---

## 1. Hardware Environment
- **Device:** `NVIDIA GeForce RTX 4060 Laptop GPU`
- **Total Dedicated VRAM:** `7.996 GB` (~8,187.5 MB)
- **Compute Capability:** `(8, 9)` (Ada Lovelace architecture)
- **Host Architecture:** x86_64 (AMD64) Windows

---

## 2. Software Versions
- **Python:** `3.14.4` (tags/v3.14.4:23116f9)
- **PyTorch:** `2.13.0+cu126`
- **torchvision:** `0.28.0+cu126`
- **CUDA Runtime:** CUDA 12.6

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
- **Backbone Network:** Modern hierarchical convolutional architecture (`torchvision.models.convnext_tiny`)
- **Pretrained Initialization:** Official ImageNet-1K V1 weights via `torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT`
- **Feature Structure:** 4 hierarchical stages with depths `[3, 3, 9, 3]` and feature channels `[96, 192, 384, 768]` utilizing 7x7 depthwise convolutions, inverted bottlenecks, LayerNorm, and GELU activations
- **Global Feature Pooling:** Adaptive average pooling (`nn.AdaptiveAvgPool2d((1, 1))`)
- **Classification Head (`model.classifier`):**
  ```python
  Sequential(
    (0): LayerNorm2d((768,), eps=1e-06, elementwise_affine=True, bias=True)
    (1): Flatten(start_dim=1, end_dim=-1)
    (2): Linear(in_features=768, out_features=3, bias=True)
  )
  ```
- **Explainability Target Layer:** `model.features[-1][-1]` (Final CNBlock in stage 7), hooked for Grad-CAM

---

## 5. Parameter Count
- **Total Parameters:** `27,822,435` (27.82M)
- **Trainable Parameters:** `27,822,435` (100.0% trainable end-to-end)
- **Frozen Parameters:** `0`
- **Classification Head Parameters:** $768 \times 3 + 3 = 2,307$ (Linear) + $768 \times 2 = 1,536$ (LayerNorm2d) = `3,843` parameters

---

## 6. Pretrained Weights Source
- **Weights Enum:** `torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT`
- **Resolved Weight Identifier:** `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`
- **Source URL:** `https://download.pytorch.org/models/convnext_tiny-983f1562.pth`
- **Local Cache:** `C:\Users\AADI/.cache\torch\hub\checkpoints\convnext_tiny-983f1562.pth` (109 MB)

---

## 7. Preprocessing & Augmentation Protocol
- **Grayscale Conversion:** In-memory `SafeRGB` channel replication (`(I, I, I)`). Source dataset files on disk remain strictly unmodified.
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
- **Validation & Test Preprocessing:**
  1. `SafeRGB()`
  2. `transforms.Resize((224, 224))`
  3. `transforms.ToTensor()`
  4. `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

---

## 8. Class Weights
Moderated square-root inverse-frequency weights computed strictly from training partition counts:
- **`NORMAL` (Class 0):** `0.8630`
- **`PNEUMONIA` (Class 1):** `0.9902`
- **`TUBERCULOSIS` (Class 2):** `1.1468`

---

## 9. Loss Function
- **Criterion:** Weighted Cross-Entropy Loss (`nn.CrossEntropyLoss(weight=class_weights)`)
- **Label Smoothing:** **STRICTLY 0.0 (NO label smoothing)**

---

## 10. Optimizer
- **Optimizer:** Decoupled Weight Decay AdamW (`torch.optim.AdamW`)
- **Base Learning Rate:** `3e-5` (`0.00003`)
- **Weight Decay:** `0.01`
- **Betas:** `(0.9, 0.999)`
- **Epsilon:** `1e-8`

---

## 11. Learning Rate Scheduler
- **Scheduler:** `CosineAnnealingLR` (`torch.optim.lr_scheduler.CosineAnnealingLR`)
- **Cycle Period ($T_{\max}$):** `15` epochs
- **Minimum Learning Rate ($\eta_{\min}$):** `1e-6` (`0.000001`)

---

## 12. Batch Size
- **Physical Batch Size:** `16` samples per GPU forward/backward step

---

## 13. Gradient Accumulation
- **Accumulation Steps:** `2`
- **Effective Batch Size:** `32` samples ($16 \times 2 = 32$)

---

## 14. AMP Configuration
- **Mixed Precision:** CUDA FP16 Automatic Mixed Precision (`torch.amp.autocast('cuda', dtype=torch.float16)`)
- **GradScaler:** `torch.amp.GradScaler('cuda', init_scale=1024)`

---

## 15. Gradient Clipping
- **Gradient Clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` applied after gradient unscaling

---

## 16. Deterministic Seed
- **Seed Value:** `42`
- **Controls Enforced:** `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, `np.random.seed(42)`, `random.seed(42)`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`, `os.environ["PYTHONHASHSEED"] = "42"`

---

## 17. Epoch-by-Epoch Training Trajectory

| Epoch | Learning Rate | Train Loss | Train Acc (%) | Train Macro F1 | Val Loss | Val Acc (%) | Val Macro Prec | Val Macro Rec | Val Macro F1 | Duration (s) | Peak VRAM (MB) | Best Checkpoint |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **01** | $3.00 \times 10^{-5}$ | 0.1398 | 95.49% | 0.9566 | 0.0558 | 98.21% | 0.9851 | 0.9821 | **0.9835** | 96.0s | 1455.5 | Saved (New Best) |
| **02** | $2.97 \times 10^{-5}$ | 0.0532 | 98.23% | 0.9837 | 0.0660 | 98.26% | 0.9826 | 0.9859 | **0.9840** | 67.6s | 1455.5 | Saved (New Best) |
| **03** | $2.87 \times 10^{-5}$ | 0.0374 | 98.78% | 0.9889 | 0.0514 | 98.87% | 0.9898 | 0.9895 | **0.9896** | 71.0s | 1455.5 | Saved (New Best) |
| **04** | $2.72 \times 10^{-5}$ | 0.0325 | 99.00% | 0.9909 | 0.0511 | 98.77% | 0.9889 | 0.9884 | 0.9886 | 66.8s | 1455.5 | Patience: 1/5 |
| **05** | $2.52 \times 10^{-5}$ | 0.0218 | 99.38% | 0.9944 | 0.0525 | 98.77% | 0.9886 | 0.9886 | 0.9886 | 65.1s | 1455.5 | Patience: 2/5 |
| **06** | $2.28 \times 10^{-5}$ | 0.0109 | 99.67% | 0.9970 | 0.0557 | 98.92% | 0.9899 | 0.9900 | **0.9900** | 67.9s | 1455.5 | Saved (New Best) |
| **07** | $2.00 \times 10^{-5}$ | 0.0090 | 99.70% | 0.9973 | 0.0727 | 98.92% | 0.9909 | 0.9892 | **0.9901** | 68.4s | 1455.5 | Saved (New Best) |
| **08** | $1.70 \times 10^{-5}$ | 0.0069 | 99.82% | 0.9984 | 0.0678 | 99.03% | 0.9916 | 0.9906 | **0.9911** | 63.8s | 1455.5 | Saved (New Best) |
| **09** | $1.40 \times 10^{-5}$ | 0.0071 | 99.79% | 0.9980 | 0.0953 | 98.62% | 0.9868 | 0.9883 | 0.9874 | 63.7s | 1455.5 | Patience: 1/5 |
| **10** | $1.10 \times 10^{-5}$ | 0.0038 | 99.90% | 0.9991 | 0.0761 | 98.97% | 0.9903 | 0.9906 | 0.9904 | 63.8s | 1455.5 | Patience: 2/5 |
| **11** | $8.25 \times 10^{-6}$ | 0.0042 | 99.85% | 0.9986 | 0.0703 | 99.08% | 0.9917 | 0.9914 | **0.9915** | 74.2s | 1455.5 | **BEST CHECKPOINT** |
| **12** | $5.80 \times 10^{-6}$ | 0.0012 | 99.93% | 0.9994 | 0.0764 | 99.03% | 0.9913 | 0.9908 | 0.9911 | 106.1s | 1455.5 | Patience: 1/5 |
| **13** | $3.77 \times 10^{-6}$ | 0.0005 | 99.97% | 0.9997 | 0.0767 | 99.08% | 0.9921 | 0.9910 | 0.9915 | 97.2s | 1455.5 | Patience: 2/5 |
| **14** | $2.25 \times 10^{-6}$ | 0.0007 | 99.99% | 0.9999 | 0.0753 | 99.03% | 0.9913 | 0.9908 | 0.9911 | 92.1s | 1455.5 | Patience: 3/5 |
| **15** | $1.32 \times 10^{-6}$ | 0.0007 | 99.98% | 0.9998 | 0.0760 | 99.08% | 0.9918 | 0.9912 | 0.9915 | 74.1s | 1455.5 | Patience: 4/5 |

---

## 18. Best Epoch
- **Optimal Epoch Selected:** **Epoch 11**
- **Trigger Criterion:** Strict peak in **Validation Macro F1** ($0.9915$)
- **Validation Loss at Best Epoch:** `0.0703`
- **Validation Accuracy at Best Epoch:** `99.08%` (1,932 / 1,950 images correct)

---

## 19. Best Validation Macro F1
- **Peak Validation Macro F1:** **`0.9915`** (99.15%)
- **Validation Macro Precision:** `0.9917` (99.17%)
- **Validation Macro Recall:** `0.9914` (99.14%)
- **Validation Weighted F1:** `0.9908` (99.08%)
- **Validation Confusion Matrix at Best Epoch:**
  - `NORMAL` (true 838): 829 predicted NORMAL, 9 predicted PNEUMONIA, 0 predicted TB
  - `PNEUMONIA` (true 637): 7 predicted NORMAL, 630 predicted PNEUMONIA, 0 predicted TB
  - `TUBERCULOSIS` (true 475): 2 predicted NORMAL, 0 predicted PNEUMONIA, 473 predicted TB

---

## 20. Best Validation Accuracy
- **Best Validation Accuracy:** **`99.08%`** (1,932 / 1,950 images correctly classified)

---

## 21. Early Stopping Status
- **Patience Configured:** `5` consecutive epochs without Macro F1 improvement
- **Trigger Status:** **Not triggered** (full 15 epochs completed cleanly)
- **Convergence Behavior:** The model achieved progressive improvements up through Epoch 11 ($0.9915$), followed by an optimal plateau with ties at Epochs 13 and 15, ending cleanly at Epoch 15 with patience counter at 4/5.

---

## 22. Total Training Duration
- **Total Clock Time:** **19.02 minutes** (1,141.03 seconds)
- **Average Epoch Duration:** ~76.1 seconds / epoch

---

## 23. Peak GPU Memory
- **Peak GPU Memory Allocated:** **1,455.51 MB** (~1.42 GB out of 7.996 GB capacity)
- **Available Headroom:** ~6.54 GB of dedicated VRAM remained free throughout training.

---

## 24. Checkpoint Path
- **Best Model Checkpoint:** [`models/chest_xray_convnext_tiny/best_model.pth`](file:///C:/Users/AADI/.gemini/antigravity/scratch/medvision_convnext_tiny/models/chest_xray_convnext_tiny/best_model.pth)
- **File Size:** `334,095,977 bytes` (~318.6 MB)
- **Checkpoint Contents:** `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `val_macro_f1`, `val_metrics`, `config`

---

## 25. Confirmation of Test Set Protection
- **Held-Out Test Set (1,951 images):** **STRICTLY UNTOUCHED & PROTECTED**
- **Test Metrics Calculated:** `NONE` (Zero test accuracy or Macro F1 computed)
- **Test Confusion Matrix:** `NONE` (Not generated)
- **Evaluation Status:** Held-out test evaluation is strictly reserved for Stage 4 upon user authorization.

---

STAGE 3 COMPLETE — FULL TRAINING FINISHED — HELD-OUT TEST NOT EVALUATED.
