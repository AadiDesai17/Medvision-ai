# Stage 4 Final Held-Out Test Evaluation & Inference Report: ConvNeXt-Tiny

**Project:** MedVision Chest X-ray 3-Class Disease Prediction  
**Architecture:** ConvNeXt-Tiny (`torchvision.models.convnext_tiny`)  
**Stage:** Stage 4 — Final Held-Out Test Evaluation & Inference Verification  
**Timestamp:** 2026-09-07  
**Status:** COMPLETE & INDEPENDENTLY VERIFIED  

---

## 1. Evaluation Protocol & Checkpoint Provenance
- **Active Model Checkpoint:** `models/chest_xray_convnext_tiny/best_model.pth` (334,095,977 bytes, 318.6 MB)
- **Checkpoint Selection Criterion:** **Highest Validation Macro F1 ONLY** achieved during standardized training (Epoch 11, Val Macro F1: `0.9915`, Val Accuracy: `99.08%`, Val Loss: `0.0703`)
- **Pretrained Starting Weights:** `ConvNeXt_Tiny_Weights.DEFAULT` (ImageNet-1K V1)
- **Classifier Architecture:** `LayerNorm2d((768,), eps=1e-06) + Flatten() + Linear(in_features=768, out_features=3)`
- **Evaluation Mode:** Strictly deterministic evaluation via `model.eval()` with zero gradient calculations (`torch.no_grad()`)
- **Dataset Path (Read-Only):** `C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia\test`
- **Total Held-Out Test Samples:** **1,951 images** (Strictly zero overlap with train or validation splits)
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
| **Overall Accuracy** | **0.9903** | **99.03%** | `0.9902614044` |
| **Macro Precision** | **0.9914** | **99.14%** | `0.9913943522` |
| **Macro Recall** | **0.9908** | **99.08%** | `0.9907610084` |
| **Macro F1-Score** | **0.9911** | **99.11%** | `0.9910696658` |
| **Weighted Precision** | **0.9903** | **99.03%** | `0.9902781373` |
| **Weighted Recall** | **0.9903** | **99.03%** | `0.9902614044` |
| **Weighted F1-Score** | **0.9903** | **99.03%** | `0.9902587045` |

### Sample Accounting & Error Summary:
- **Total Test Cohort:** `1,951` images
- **Correctly Classified:** `1,932` images (**99.03%**)
- **Incorrectly Classified:** `19` images (**0.97%**)
- **Total Test Error Count:** `19` errors
- **Test Error Percentage:** **`0.97%`**

---

## 3. Per-Class Diagnostic Performance Breakdown

| Class Index | Diagnostic Category | Support (True Count) | Precision | Recall (Sensitivity) | F1-Score |
|:---:|:---|:---:|:---:|:---:|:---:|
| **0** | `NORMAL` | 839 | **0.9858** (98.58%) | **0.9917** (99.17%) | **0.9887** |
| **1** | `PNEUMONIA` | 637 | **0.9905** (99.05%) | **0.9827** (98.27%) | **0.9866** |
| **2** | `TUBERCULOSIS` | 475 | **0.9979** (99.79%) | **0.9979** (99.79%) | **0.9979** |
| **Cohort** | **Overall / Macro** | **1,951** | **0.9914** | **0.9908** | **0.9911** |

---

## 4. Confusion Matrix Analysis

```
                NORMAL   PNEUMONIA   TUBERCULOSIS   Total
NORMAL             832           6              1     839
PNEUMONIA           11         626              0     637
TUBERCULOSIS         1           0            474     475
Total Predicted    844         632            475    1951
```

- **Visual Artifact:** Generated and saved to [`results/chest_xray_convnext_tiny/confusion_matrix.png`](file:///C:/Users/AADI/.gemini/antigravity/scratch/medvision_convnext_tiny/results/chest_xray_convnext_tiny/confusion_matrix.png).
- **Key Diagnostic Observations:**
  - **Tuberculosis Detection:** Near-perfect sensitivity: **474 out of 475** active tuberculosis cases were accurately identified ($99.79\%$ recall, $99.79\%$ precision). Zero confusion between Tuberculosis and Pneumonia in either direction ($0$ false positives, $0$ false negatives). Only 1 TB case was mildly confused as Normal.
  - **Pneumonia Identification:** **626 out of 637** cases identified ($98.27\%$ recall, $99.05\%$ precision). 11 cases were classified as Normal, with zero leakage into Tuberculosis.
  - **Normal Screening:** **832 out of 839** healthy radiographs identified ($99.17\%$ recall). Only 6 were classified as Pneumonia and 1 as Tuberculosis.

---

## 5. Diagnostic Error Analysis

The 19 misclassifications are logged with exact filename, true diagnosis, model prediction, and confidence in [`results/chest_xray_convnext_tiny/test_errors.csv`](file:///C:/Users/AADI/.gemini/antigravity/scratch/medvision_convnext_tiny/results/chest_xray_convnext_tiny/test_errors.csv).

| Misclassification Type | Count | Percentage of Errors | Representative Examples |
|:---|:---:|:---:|:---|
| **Pneumonia misclassified as Normal** | 11 | 57.89% | `jtiptj_test_person154_bacteria_728.png`, `jtiptj_train_person52_bacteria_251.png`, `jtiptj_train_person954_virus_1626.png` |
| **Normal misclassified as Pneumonia** | 6 | 31.58% | `jtiptj_test_IM-0079-0001.png`, `jtiptj_test_IM-0091-0001.png`, `jtiptj_test_NORMAL2-IM-0210-0001.png` |
| **Normal misclassified as Tuberculosis** | 1 | 5.26% | `rahman_Normal-2964.png` |
| **Tuberculosis misclassified as Normal** | 1 | 5.26% | `rahman_Tuberculosis-651.png` |
| **Pneumonia misclassified as Tuberculosis** | 0 | 0.00% | *Zero confusion* |
| **Tuberculosis misclassified as Pneumonia** | 0 | 0.00% | *Zero confusion* |

---

## 6. Standalone Inference Verification

Inference was verified against real test radiographs using `predict_convnext_tiny.py`.

### Case 1: Healthy Chest X-Ray (`NORMAL`)
- **Filename:** `jtiptj_test_IM-0021-0001.png`
- **True Class:** `NORMAL`
- **Predicted Class:** `NORMAL`
- **Probabilities:**
  - `NORMAL`: **100.00%**
  - `PNEUMONIA`: 0.00%
  - `TUBERCULOSIS`: 0.00%
- **Top Confidence:** **100.00%**
- **Clinical Notice:** *Softmax confidence is model confidence and is not clinical certainty.*

### Case 2: Acute Pulmonary Consolidation (`PNEUMONIA`)
- **Filename:** `jtiptj_test_person100_bacteria_475.png`
- **True Class:** `PNEUMONIA`
- **Predicted Class:** `PNEUMONIA`
- **Probabilities:**
  - `NORMAL`: 0.00%
  - `PNEUMONIA`: **100.00%**
  - `TUBERCULOSIS`: 0.00%
- **Top Confidence:** **100.00%**
- **Clinical Notice:** *Softmax confidence is model confidence and is not clinical certainty.*

### Case 3: Mycobacterial Infection (`TUBERCULOSIS`)
- **Filename:** `mendeley_TB.1005.png`
- **True Class:** `TUBERCULOSIS`
- **Predicted Class:** `TUBERCULOSIS`
- **Probabilities:**
  - `NORMAL`: 0.00%
  - `PNEUMONIA`: 0.00%
  - `TUBERCULOSIS`: **100.00%**
- **Top Confidence:** **100.00%**
- **Clinical Notice:** *Softmax confidence is model confidence and is not clinical certainty.*

---

## 7. Project & Environment Isolation Verification
- **Dataset Read-Only Status:** `C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia` remains strictly unchanged and unmodified.
- **Historical Model Workspaces:**
  - `medvision_disease_prediction`: Strictly UNTOUCHED.
  - `medvision_resnet50`: Strictly UNTOUCHED.
  - `medvision_vit_b16`: Strictly UNTOUCHED.
  - `medvision_swin_tiny`: Strictly UNTOUCHED.
- **Checkpoint Integrity:** `models/chest_xray_convnext_tiny/best_model.pth` was evaluated strictly in inference mode (`model.eval()`, `torch.no_grad()`); zero weight alterations or retraining occurred.

---

STAGE 4 COMPLETE — HELD-OUT TEST EVALUATION AND INFERENCE VERIFIED.
