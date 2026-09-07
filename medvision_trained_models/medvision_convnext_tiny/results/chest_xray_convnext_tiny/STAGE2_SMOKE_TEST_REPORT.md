# Stage 2 Smoke Test Report: Standardized ConvNeXt-Tiny Implementation

**Project:** MedVision Chest X-ray 3-Class Disease Prediction  
**Architecture:** ConvNeXt-Tiny (`torchvision.models.convnext_tiny`)  
**Stage:** Stage 2 — Finalize Implementation + Real-Data GPU Smoke Test  
**Timestamp:** 2026-09-07  
**Status:** COMPLETE & 100% VERIFIED  

---

## 1. Model Architecture
- **Backbone Network:** Modern hierarchical convolutional network (`torchvision.models.convnext_tiny`)
- **Pretrained Initialization:** Official ImageNet-1K V1 weights via `torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT`
- **Feature Structure:** 4 hierarchical stages with depths `[3, 3, 9, 3]` and feature channels `[96, 192, 384, 768]` utilizing 7x7 depthwise convolutions, inverted bottlenecks, LayerNorm, and GELU activations
- **Global Feature Pooling:** Adaptive average pooling (`nn.AdaptiveAvgPool2d((1, 1))`)
- **Classifier Head:**
  ```
  Sequential(
    (0): LayerNorm2d((768,), eps=1e-06, elementwise_affine=True, bias=True)
    (1): Flatten(start_dim=1, end_dim=-1)
    (2): Linear(in_features=768, out_features=3, bias=True)
  )
  ```
- **Grad-CAM Target Layer:** `model.features[-1][-1]` (Final CNBlock in stage 7)

---

## 2. Parameter Count
- **Total Parameters:** `27,822,435` (27.82M)
- **Trainable Parameters:** `27,822,435` (100.0% trainable end-to-end)
- **Frozen Parameters:** `0`
- **Head Parameters:** $768 \times 3 + 3 = 2,307$ (Linear) + $768 \times 2 = 1,536$ (LayerNorm2d) = `3,843` parameters

---

## 3. Dataset Verification
- **Dataset Root Path (Read-Only):** `C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia`
- **Samples Loaded:**
  - `train`: 9,097 images (NORMAL: 3,911, PNEUMONIA: 2,971, TUBERCULOSIS: 2,215)
  - `val`: 1,950 images (NORMAL: 838, PNEUMONIA: 637, TUBERCULOSIS: 475)
  - `test`: 1,951 images (NORMAL: 839, PNEUMONIA: 637, TUBERCULOSIS: 475)
  - **Cohort Total:** 12,998 real chest radiographs across 3 diagnostic categories
- **Grayscale Handling:** In-memory `SafeRGB` channel replication (1-channel $\rightarrow$ 3-channel RGB `(I, I, I)`)
- **Standardized Train Augmentations:**
  - `Resize((224, 224))`
  - `RandomRotation(degrees=7)`
  - `ColorJitter(brightness=0.05, contrast=0.05)`
  - `ToTensor()`
  - `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
  - Strict prohibition of horizontal/vertical flips and random resized crops
- **Standardized Class Weights Applied:**
  - `NORMAL` (Class 0): `0.8630`
  - `PNEUMONIA` (Class 1): `0.9902`
  - `TUBERCULOSIS` (Class 2): `1.1468`
- **Real Batch Sample Count:** 16 images
- **Labels Checked:** Target values within valid index bounds $[0, 2]$

---

## 4. Tensor Shapes
- **Input Batch Tensor Shape:** `torch.Size([16, 3, 224, 224])`
- **Batch Targets Tensor Shape:** `torch.Size([16])`
- **Forward Logits Tensor Shape:** `torch.Size([16, 3])`
- **Softmax Probability Tensor Shape:** `torch.Size([16, 3])` (all row sums equal to $1.0000 \pm 10^{-4}$)

---

## 5. Loss Value
- **Loss Criterion:** Weighted Cross-Entropy Loss (`torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)`)
- **Label Smoothing:** Strictly disabled (`0.0`)
- **Evaluated Loss Value on Real Batch:** `1.1733` (finite, non-zero, well-conditioned)

---

## 6. Gradient Verification
- **Backward Pass:** Executed with scaled gradient accumulation step (`loss / 2.0`)
- **Gradient Tensors Checked:** 182 parameter tensors
- **Finite Check:** 100% of gradients are finite (zero `NaN` or `Inf` values detected)
- **Gradient Norm Before Clipping:** `17.3021`
- **Gradient Clipping Applied:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- **Effective Norm After Clipping:** Clipped safely to $\le 1.0$ (finite: `True`)

---

## 7. AMP Verification
- **API Used:** Modern PyTorch AMP API (`torch.amp.autocast('cuda', dtype=torch.float16)`)
- **GradScaler Initialization:** `torch.amp.GradScaler('cuda', init_scale=1024)`
- **Scale Status:** Initialized at `1024.0`; scaled backward, unscaled for gradient clipping, and updated cleanly without scale skips (`1024.0 -> 1024.0`)

---

## 8. Optimizer Verification
- **Optimizer:** Decoupled Weight Decay AdamW (`torch.optim.AdamW`)
- **Parameters Optimized:** All 27,822,435 network parameters
- **Learning Rate:** `3e-5` (`0.00003`)
- **Weight Decay:** `0.01`
- **Betas:** `(0.9, 0.999)`
- **Epsilon:** `1e-8`
- **Optimizer Step:** Completed cleanly via `scaler.step(optimizer)`, zeroed gradients cleanly via `optimizer.zero_grad(set_to_none=True)`

---

## 9. Scheduler Verification
- **Scheduler:** `CosineAnnealingLR`
- **Configuration:** $T_{\max} = 15$ epochs, $\eta_{\min} = 1\text{e-}6$
- **Initial Learning Rate:** `3.000000e-05`
- **Learning Rate After Step 1:** `2.968314e-05` (smooth cosine deceleration confirmed)

---

## 10. Checkpoint Load and Save Verification
- **Safe Disposable Checkpoint Path:** `models/chest_xray_convnext_tiny/disposable_smoke_test_ckpt.pth`
- **File Size Generated:** `106.20 MB` (27,822,435 float32 weights + bias terms)
- **Load Verification:** Re-instantiated clean ConvNeXt-Tiny model without pretrained weights, loaded weights via `torch.load`
- **Weights Integrity:** 100% parameter equivalence verified (`torch.equal(p1, p2)` across all 182 layers)
- **Disposable Artifact Cleanup:** Disposable file was cleanly unlinked and deleted immediately after verification (zero residual test files)

---

## 11. GPU Name & Hardware Environment
- **Device Name:** `NVIDIA GeForce RTX 4060 Laptop GPU`
- **Compute Capability:** `(8, 9)` (Ada Lovelace)
- **Total Dedicated VRAM:** `7.996 GB` (~8,187.5 MB)
- **Python Version:** `3.14.4`
- **PyTorch Version:** `2.13.0+cu126`
- **torchvision Version:** `0.28.0+cu126`

---

## 12. Peak GPU Memory & Headroom
- **Baseline Memory Allocated:** `455.64 MB`
- **Peak Memory Allocated:** `1,139.89 MB`
- **Peak Memory Reserved:** `1,258.00 MB`
- **Available VRAM Headroom:** Over `7.04 GB` remaining (`7,047.61 MB` free under 8,187.50 MB ceiling)
- **OOM Risk Assessment:** Negligible; batch size 16 with FP16 AMP operates with immense headroom (consuming < 14% of VRAM)

---

## 13. Pass/Fail Summary Table

| Test # | Verification Item | Target Standard | Result | Status |
|:---:|:---|:---|:---:|:---:|
| 1 | **Deterministic Seed** | Seed = 42 across Python, NumPy, PyTorch, CUDA | Seed set | **PASS** |
| 2 | **GPU Detection** | NVIDIA GeForce RTX 4060 Laptop GPU | Detected (8.00 GB) | **PASS** |
| 3 | **Dataset Loader** | 9,097 train CXR images loaded | 9,097 images | **PASS** |
| 4 | **Class Weights** | [0.8630, 0.9902, 1.1468] | Exact match | **PASS** |
| 5 | **Real Batch Loading** | Batch size = 16, shape [16, 3, 224, 224] | Shape verified | **PASS** |
| 6 | **Label Range** | Labels $\in \{0, 1, 2\}$ | Range verified | **PASS** |
| 7 | **Model Architecture** | ConvNeXt-Tiny + LayerNorm2d + Flatten + Linear(768, 3) | Verified | **PASS** |
| 8 | **Parameter Count** | 27,822,435 total and trainable parameters | Exact match | **PASS** |
| 9 | **Forward Pass & AMP** | Logits shape [16, 3], finite, CUDA FP16 | Verified | **PASS** |
| 10 | **Weighted Loss** | Weighted CE without label smoothing | Loss = 1.1733 | **PASS** |
| 11 | **Backward Pass** | Finite gradients across all 182 tensors | Verified finite | **PASS** |
| 12 | **Gradient Clipping** | Clip norm to max_norm = 1.0 | Finite clipped norm | **PASS** |
| 13 | **Optimizer Step** | AdamW step + GradScaler update | Completed | **PASS** |
| 14 | **Scheduler Step** | CosineAnnealingLR step (LR reduced) | Verified | **PASS** |
| 15 | **Checkpoint Save/Load** | Disposable save, load, exact weights match, cleanup | 100% match | **PASS** |
| 16 | **Inference Verification** | Softmax output sums to 1.0 on real CXR batch | Sum = 1.0000 | **PASS** |
| 17 | **Memory Safety** | Peak VRAM < 50% capacity | 1,139.89 MB (< 14%) | **PASS** |
| 18 | **Project Isolation** | Dataset read-only, old projects untouched | Fully preserved | **PASS** |

---

## 14. Confirmation of Non-Interference and No Full Training

- **Multi-Epoch Training Performed:** `FALSE` (strictly 1 batch forward/backward step for validation)
- **Source Dataset Modified:** `FALSE` (`C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia` accessed strictly read-only)
- **Old Project Modified:** `FALSE` (`C:\Users\AADI\.gemini\antigravity\scratch\medvision_disease_prediction` untouched)
- **ResNet50 Project Modified:** `FALSE` (`C:\Users\AADI\.gemini\antigravity\scratch\medvision_resnet50` untouched)
- **ViT-B/16 Project Modified:** `FALSE` (`C:\Users\AADI\.gemini\antigravity\scratch\medvision_vit_b16` untouched)
- **Swin-Tiny Project Modified:** `FALSE` (`C:\Users\AADI\.gemini\antigravity\scratch\medvision_swin_tiny` untouched)
- **Held-Out Test Set Evaluated:** `FALSE` (test set remains strictly untouched for Stage 4)
- **Safety Guard in `src/train.py`:** Active and verified blocking unauthorized execution

---

STAGE 2 COMPLETE — SMOKE TEST PASSED — FULL TRAINING NOT PERFORMED.
