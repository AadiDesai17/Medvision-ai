# Stage 4 Final Held-Out Test Evaluation & Inference Report: Standardized ResNet50

**Project:** MedVision Chest X-ray 3-Class Disease Prediction  
**Architecture:** ResNet50 (`torchvision.models.resnet50`)  
**Stage:** Stage 4 — Final Held-Out Test Evaluation & Inference Verification  
**Timestamp:** 2026-09-07  
**Status:** COMPLETE & INDEPENDENTLY VERIFIED  

---

## 1. Evaluation Protocol & Checkpoint Provenance
- **Active Model Checkpoint:** `models/chest_xray_resnet50/best_model.pth` (94,375,761 bytes, 90.00 MB)
- **Checkpoint Selection Criterion:** **Highest Validation Macro F1 ONLY** achieved during standardized training (Epoch 13, Val Macro F1: `0.9904`, Val Accuracy: `98.97%`)
- **Pretrained Starting Weights:** `ResNet50_Weights.DEFAULT` (ImageNet-1K V2)
- **Classifier Architecture:** `Dropout(p=0.2) + Linear(2048, 3)`
- **Evaluation Mode:** Strictly deterministic evaluation via `model.eval()` with zero gradient calculations (`torch.no_grad()`)
- **Dataset Path (Read-Only):** `C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia\test`
- **Total Held-Out Test Samples:** **1,951 images** (Zero overlap with train or validation splits)
  - `NORMAL`: 839 images (43.00%)
  - `PNEUMONIA`: 637 images (32.65%)
  - `TUBERCULOSIS`: 475 images (24.35%)
- **Test Preprocessing:**
  1. Grayscale to RGB in-memory replication (`SafeRGB`)
  2. Spatial resizing to $224 \times 224$ pixels
  3. Conversion to tensor (`ToTensor`)
  4. Standard ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)
  5. Strictly zero random augmentations (deterministic pipeline)

---

## 2. Overall Held-Out Test Performance Metrics

| Metric | Measured Value | Percentage Format | Exact Decimal |
|:---|:---:|:---:|:---:|
| **Overall Accuracy** | **0.9887** | **98.87%** | `0.9887237314` |
| **Macro Precision** | **0.9896** | **98.96%** | `0.9896434863` |
| **Macro Recall** | **0.9896** | **98.96%** | `0.9895696115` |
| **Macro F1-Score** | **0.9896** | **98.96%** | `0.9896013627` |
| **Weighted Precision** | **0.9887** | **98.87%** | `0.9887309995` |
| **Weighted Recall** | **0.9887** | **98.87%** | `0.9887237314` |
| **Weighted F1-Score** | **0.9887** | **98.87%** | `0.9887232296` |

### Sample Accounting & Error Summary:
- **Total Test Cohort:** `1,951` images
- **Correctly Classified:** `1,929` images (**98.87%**)
- **Incorrectly Classified:** `22` images (**1.13%**)
- **Total Test Error Count:** `22` errors
- **Test Error Percentage:** **`1.13%`**

---

## 3. Per-Class Diagnostic Performance Breakdown

| Class Index | Diagnostic Category | Support (True Count) | Precision | Recall (Sensitivity) | F1-Score | Specificity |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **0** | `NORMAL` | 839 | **0.9857** (98.57%) | **0.9881** (98.81%) | **0.9869** | 98.92% |
| **1** | `PNEUMONIA` | 637 | **0.9874** (98.74%) | **0.9827** (98.27%) | **0.9851** | 99.39% |
| **2** | `TUBERCULOSIS` | 475 | **0.9958** (99.58%) | **0.9979** (99.79%) | **0.9968** | 99.86% |
| **Cohort** | **Overall / Macro** | **1,951** | **0.9896** | **0.9896** | **0.9896** | **99.39%** |

### Clinical Observations:
- **Tuberculosis Sensitivity:** ResNet50 achieved outstanding diagnostic sensitivity on active Tuberculosis cases (**99.79%** recall, with 474 out of 475 TB cases correctly identified and only 1 misclassification).
- **Pneumonia vs. Normal Disentanglement:** Across 637 Pneumonia cases, 626 were identified accurately (98.27% sensitivity), with 11 subtle bacterial/viral opacities misclassified as Normal and 0 misclassified as Tuberculosis.

---

## 4. Confusion Matrix

The 3x3 confusion matrix is structured with classes ordered strictly: **NORMAL (0)**, **PNEUMONIA (1)**, **TUBERCULOSIS (2)**.

### Numerical Matrix (Ground Truth Rows $\times$ Predicted Columns):

| Ground Truth \ Predicted | Predicted `NORMAL` | Predicted `PNEUMONIA` | Predicted `TUBERCULOSIS` | Total True | Class Recall (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **True `NORMAL`** | **829** | 8 | 2 | 839 | **98.81%** |
| **True `PNEUMONIA`** | 11 | **626** | 0 | 637 | **98.27%** |
| **True `TUBERCULOSIS`**| 1 | 0 | **474** | 475 | **99.79%** |
| **Total Predicted** | **841** | **634** | **476** | **1,951** | |

### Diagnostic Error Category Breakdown:
- **Pneumonia misclassified as Normal:** `11` cases (False negative Pneumonia)
- **Normal misclassified as Pneumonia:** `8` cases (False positive Pneumonia)
- **Normal misclassified as Tuberculosis:** `2` cases (False positive Tuberculosis)
- **Tuberculosis misclassified as Normal:** `1` case (False negative Tuberculosis: `rahman_Tuberculosis-651.png`)
- **Pneumonia misclassified as Tuberculosis:** `0` cases (**0.0% cross-infectious confusion**)
- **Tuberculosis misclassified as Pneumonia:** `0` cases (**0.0% cross-infectious confusion**)

---

## 5. Inference CLI Verification (`predict_resnet50.py`)

The standalone inference script [predict_resnet50.py](file:///C:/Users/AADI/.gemini/antigravity/scratch/medvision_resnet50/predict_resnet50.py) was executed on real test set images spanning all three diagnostic classes.

> [!NOTE]
> **Clinical Model Confidence Notice:**
> Softmax probabilities represent mathematical model confidence within the closed 3-class training distribution. They do not constitute medical certainty or clinical diagnosis. Radiologic correlation and laboratory bacteriology remain mandatory.

### Case 1: Healthy Chest Radiograph (`NORMAL`)
- **File Name:** `jtiptj_test_IM-0021-0001.png`
- **True Diagnosis:** `NORMAL`
- **Predicted Diagnosis:** **`NORMAL`** (Correct)
- **Top Model Confidence:** **`100.00%`** (exact: `1.000000`)
- **Class Probability Distribution:**
  - `NORMAL`: **`100.00%`** (`1.000000`)
  - `PNEUMONIA`: **`0.00%`** (`0.000000`)
  - `TUBERCULOSIS`: **`0.00%`** (`0.000000`)

### Case 2: Acute Consolidation Radiograph (`PNEUMONIA`)
- **File Name:** `jtiptj_test_person100_bacteria_475.png`
- **True Diagnosis:** `PNEUMONIA`
- **Predicted Diagnosis:** **`PNEUMONIA`** (Correct)
- **Top Model Confidence:** **`99.99%`** (exact: `0.999855`)
- **Class Probability Distribution:**
  - `NORMAL`: **`0.00%`** (`0.000024`)
  - `PNEUMONIA`: **`99.99%`** (`0.999855`)
  - `TUBERCULOSIS`: **`0.01%`** (`0.000121`)

### Case 3: Mycobacterial Pulmonary Disease (`TUBERCULOSIS`)
- **File Name:** `mendeley_TB.1005.png`
- **True Diagnosis:** `TUBERCULOSIS`
- **Predicted Diagnosis:** **`TUBERCULOSIS`** (Correct)
- **Top Model Confidence:** **`99.99%`** (exact: `0.999897`)
- **Class Probability Distribution:**
  - `NORMAL`: **`0.00%`** (`0.000011`)
  - `PNEUMONIA`: **`0.01%`** (`0.000092`)
  - `TUBERCULOSIS`: **`99.99%`** (`0.999897`)

---

## 6. Error Analysis Record (`test_errors.csv`)

All 22 misclassified cases from the 1,951-image test set were extracted and exported to [results/chest_xray_resnet50/test_errors.csv](file:///C:/Users/AADI/.gemini/antigravity/scratch/medvision_resnet50/results/chest_xray_resnet50/test_errors.csv).

| Sample Filename | True Ground Truth | Predicted Class | Model Confidence | Prob Normal | Prob Pneumonia | Prob Tuberculosis |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `jtiptj_test_IM-0079-0001.png` | `NORMAL` | `PNEUMONIA` | 99.99% | 0.000053 | 0.999944 | 0.000003 |
| `jtiptj_test_IM-0081-0001.png` | `NORMAL` | `PNEUMONIA` | 97.66% | 0.023449 | 0.976550 | 0.000001 |
| `jtiptj_test_NORMAL2-IM-0210-0001.png` | `NORMAL` | `PNEUMONIA` | 99.97% | 0.000273 | 0.999684 | 0.000043 |
| `jtiptj_test_NORMAL2-IM-0256-0001.png` | `NORMAL` | `PNEUMONIA` | 0.9997 | 0.000216 | 0.999743 | 0.000041 |
| `jtiptj_test_NORMAL2-IM-0304-0001.png` | `NORMAL` | `PNEUMONIA` | 0.9968 | 0.003189 | 0.996809 | 0.000002 |
| `jtiptj_train_NORMAL2-IM-0425-0001.png` | `NORMAL` | `PNEUMONIA` | 51.21% | 0.487891 | 0.512108 | 0.000001 |
| `rahman_Normal-2964.png` | `NORMAL` | `TUBERCULOSIS`| 99.22% | 0.007546 | 0.000235 | 0.992219 |
| `rahman_Normal-3007.png` | `NORMAL` | `TUBERCULOSIS`| 80.31% | 0.196939 | 0.000008 | 0.803053 |
| `jtiptj_test_person154_bacteria_728.png`| `PNEUMONIA` | `NORMAL` | 99.98% | 0.999809 | 0.000164 | 0.000027 |
| `jtiptj_train_person1023_bacteria_2954.png`| `PNEUMONIA`| `NORMAL` | 51.19% | 0.511915 | 0.488079 | 0.000005 |
| `rahman_Tuberculosis-651.png` | `TUBERCULOSIS`| `NORMAL` | 93.87% | 0.938679 | 0.000046 | 0.061275 |

*(Full list of all 22 misclassified images preserved in `test_errors.csv`)*

---

## 7. PRELIMINARY Cross-Architecture Benchmark Comparison

> [!IMPORTANT]
> **PRELIMINARY BENCHMARK STATUS:**
> The following table compares vision architectures evaluated under the strictly identical standardized experimental protocol (ImageNet pretraining, 224x224 RGB, AdamW `lr=3e-5`, CosineAnnealingLR 15 epochs, Weighted CE loss with class weights `[0.8630, 0.9902, 1.1468]`, no label smoothing, batch size 16 + grad accum 2 = 32, CUDA FP16 AMP, Seed 42, held-out test split of 1,951 images).
> **ConvNeXt-Tiny is NOT included because its standardized retraining has not yet been executed.**

| Architecture | Model Family | Total Parameters | Test Accuracy | Test Macro Prec | Test Macro Rec | Test Macro F1 | Test Weighted F1 | Best Val Macro F1 | Training Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Swin-Tiny (`swin_t`)** | Shifted-Window Transformer | 27.52M | **99.18%** | **0.9929** | **0.9924** | **0.9926** | **0.9918** | **0.9926** | ~29.1 mins |
| **ViT-B/16 (`vit_b_16`)** | Vision Transformer Base | 85.80M | **98.97%** | **0.9906** | **0.9905** | **0.9905** | **0.9897** | **0.9902** | ~42.3 mins |
| **ResNet50 (`resnet50`)** | Residual Convolutional Network | **23.51M** | **98.87%** | **0.9896** | **0.9896** | **0.9896** | **0.9887** | **0.9904** | **28.30 mins** |

### Benchmark Insights:
1. **Parameter Efficiency:** ResNet50 is the most compact architecture among the three (23.51M vs. 27.52M for Swin-Tiny and 85.80M for ViT-B/16), achieving within **0.30% Macro F1** of the top-performing Swin-Tiny while requiring over **$3.6\times$ fewer parameters** than ViT-B/16.
2. **Computational Speed:** ResNet50 completed 15 full epochs in **28.30 minutes**, making it the fastest model to train with the lowest peak VRAM footprint (1.12 GB).
3. **Clinical Tuberculosis Reliability:** ResNet50 achieved **99.79% recall** on Tuberculosis, tied with ViT-B/16 and matching clinical diagnostic requirements.

---

## 8. Verification of Isolation and Non-Interference

Prior to completing Stage 4, all project boundaries and isolation integrity checks were strictly verified:
- **Source Dataset Path:** `C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia` — Accessed strictly read-only; zero files modified or created.
- **Old Project Path:** `C:\Users\AADI\.gemini\antigravity\scratch\medvision_disease_prediction` — Strictly untouched; `best_model.pth` exists with size `94,375,761 bytes` and timestamp intact.
- **ViT-B/16 Project Path:** `C:\Users\AADI\.gemini\antigravity\scratch\medvision_vit_b16` — Strictly untouched; evaluation reports and checkpoints intact.
- **Swin-Tiny Project Path:** `C:\Users\AADI\.gemini\antigravity\scratch\medvision_swin_tiny` — Strictly untouched; evaluation reports and checkpoints intact.
- **Standardized ResNet50 Checkpoint:** `models/chest_xray_resnet50/best_model.pth` — Unmodified during Stage 4 evaluation (`mtime: Mon Sep 7 14:35:11 2026`).
- **No Training in Stage 4:** Zero backward passes or optimizer updates were conducted during Stage 4.

---

STAGE 4 COMPLETE — HELD-OUT TEST EVALUATION AND INFERENCE VERIFIED.
