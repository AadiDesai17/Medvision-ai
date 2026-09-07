# Chest X-Ray 3-Class Disease Prediction Model Report

**MedVision – AI Medical Imaging Assistant**  
**Module:** Disease Prediction (Thoracic Imaging Modality)  
**Date:** September 2026  
**Status:** COMPLETE, VERIFIED & ACTIVE  

---

## 1. Dataset Description

The Chest X-ray Disease Prediction module utilizes a curated 3-class thoracic radiography dataset comprising **NORMAL**, **PNEUMONIA**, and **TUBERCULOSIS** cases:
- **Clinical Modality:** Posteroanterior (PA) and Anteroposterior (AP) digital chest radiographs (CXR).
- **Diagnostic Categories:**
  1. `NORMAL`: Healthy chest radiographs demonstrating clear lung fields, normal cardiothoracic ratio, sharp costophrenic angles, and absence of consolidation, infiltration, or cavitary lesions.
  2. `PNEUMONIA`: Radiographs demonstrating acute alveolar or interstitial pulmonary consolidation, opacities, or bronchopneumonic infiltrates (bacterial and viral etiology).
  3. `TUBERCULOSIS`: Radiographs exhibiting active or fibrotic *Mycobacterium tuberculosis* manifestations, including apical infiltrates, cavitary lesions, hilar adenopathy, and miliary patterning.
- **Dataset Provenance:** Aggregated from public domain and open-access radiological repositories, including the NIH Clinical Center Chest X-ray collection, RSNA Pneumonia Detection Challenge, and Kaggle/Mendeley Tuberculosis datasets.

---

## 2. Dataset Audit

A complete filesystem and integrity audit of `C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia` was executed prior to training:
- **Total Files Scanned:** 25,996 `.png` images on disk across `train/`, `val/`, `test/`, and `combined/`.
- **Predefined Split Images (Unique Partition):** Exactly **12,998 images**.
- **Combined Directory:** Contains exactly 12,998 images. Programmatic MD5 hash verification confirmed that `combined/` is a 100% duplicate copy of `train/ + val/ + test/`. To prevent severe data duplication and validation leakage, `combined/` was strictly excluded from training and evaluation.
- **File Format:** 100% single- or multi-channel PNG images.
- **Header & Pixel Integrity:** 100% of images verified readable via PIL (`img.verify()`). Zero corrupt or unreadable files.
- **Zero-Byte Check:** Exactly 0 zero-byte files detected.

---

## 3. Class Distribution

The dataset displays a mild, realistic class distribution across the three clinical categories:

| Class Code | Clinical Diagnostic Name | Total Unique Images | Percentage |
|:---|:---|:---:|:---:|
| `NORMAL` | Healthy / Normal Lung Radiograph | 5,588 | 42.99% |
| `PNEUMONIA` | Acute Bacterial / Viral Pneumonia | 4,245 | 32.66% |
| `TUBERCULOSIS`| Active / Cavitary Tuberculosis | 3,165 | 24.35% |
| **Total** | | **12,998** | **100.00%** |

The maximum class imbalance ratio is $5,588 / 3,165 = 1.77:1$, representing a well-distributed dataset that does not require extreme re-weighting.

---

## 4. Train / Validation / Test Structure

The dataset arrived with pre-established standard partitions. These predefined splits were preserved to ensure reproducible benchmarking:

| Partition | Total Images | NORMAL | PNEUMONIA | TUBERCULOSIS | Split Proportion |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Train** | 9,097 | 3,911 (42.99%) | 2,971 (32.66%) | 2,215 (24.35%) | 69.99% |
| **Validation** | 1,950 | 838 (42.97%) | 637 (32.67%) | 475 (24.36%) | 15.00% |
| **Test (Held-Out)**| 1,951 | 839 (43.00%) | 637 (32.65%) | 475 (24.35%) | 15.01% |
| **Total** | **12,998** | **5,588** | **4,245** | **3,165** | **100.00%** |

Split manifests were saved to:
- `data/chest_xray/train.csv`
- `data/chest_xray/val.csv`
- `data/chest_xray/test.csv`
- `data/chest_xray/manifest.csv`

---

## 5. Data-Quality Checks

Every single file in the 12,998-image split cohort underwent programmatic validation:
- **Pixel decoding test:** Passed 100% (12,998 / 12,998).
- **Color Mode:** Predominantly 8-bit or 16-bit single-channel grayscale (`L`) or 3-channel RGB. Handled uniformly via channel replication.
- **Resolution range:** Typical resolutions range from $512 	imes 512$ to $2048 	imes 2048$ pixels, capturing fine alveolar and bronchovascular detail.

---

## 6. Duplicate & Leakage Analysis

Exact MD5 cryptographic checksums were computed for all images in each split:
- Split `train`: 9,097 unique hashes (0 internal duplicates).
- Split `val`: 1,950 unique hashes (0 internal duplicates).
- Split `test`: 1,951 unique hashes (0 internal duplicates).

### Cross-Split Overlap Verification:
- $	ext{Train} \cap 	ext{Validation} = \emptyset$ (**0 overlapping hashes**)
- $	ext{Train} \cap 	ext{Test} = \emptyset$ (**0 overlapping hashes**)
- $	ext{Validation} \cap 	ext{Test} = \emptyset$ (**0 overlapping hashes**)

**Result:** There is strictly **zero cross-split data leakage**. The test set is completely held-out and untouched during model parameter optimization.

---

## 7. Patient-ID Availability & Limitation

> [!IMPORTANT]
> **Explicit Provenance Disclosure:**
> **Patient-level separation could not be independently verified from the available dataset metadata.**
> The underlying source files do not provide DICOM metadata headers, patient demographics, accession numbers, or unique patient ID strings. Filenames reflect synthetic hash prefixes from public dataset aggregations. Therefore, while strict image-hash independence across splits is mathematically proven (0 duplicate images), potential patient-level re-imaging cannot be independently ruled out.

---

## 8. Preprocessing Pipeline

- **Grayscale Channel Replication (`SafeRGB`):** Chest radiographs are grayscale images. ResNet-50 requires 3-channel input ($C=3$). Converting mode `L` to mode `RGB` replicates the single intensity channel across all three channels: $(I, I, I)$. This guarantees mathematical consistency with ImageNet convolutional kernels without introducing spurious color artifacts.
- **Spatial Resizing:** Resized to standardized $224 	imes 224$ pixels.
- **Normalization:** Zero-mean, unit-variance standardization using standard ImageNet parameters:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
- **Evaluation Pipeline:** Deterministic `SafeRGB` $ightarrow$ `Resize((224, 224))` $ightarrow$ `ToTensor` $ightarrow$ `Normalize`.

---

## 9. Medically Calibrated Data Augmentation

In medical thoracic imaging, arbitrary computer-vision augmentations can destroy diagnostic integrity (e.g. flipping an X-ray vertically produces situs inversus, which simulates dextrocardia). 

### Augmentation Rules Enforced:
1. **Random Resized Crop:** `scale=(0.85, 1.0)`, `ratio=(0.95, 1.05)` — simulates minor patient positioning shifts without cropping out apical lung zones or costophrenic recesses.
2. **Conservative Horizontal Flip:** `p=0.5` — chest symmetry permits moderate lateral variation.
3. **Restricted Rotation:** $\pm 7^\circ$ — accommodates slight patient tilt during mobile or portable bedside radiography while preventing unnatural skewing.
4. **Mild Photometric Jitter:** Brightness $\pm 10\%$, Contrast $\pm 10\%$ — models variations in X-ray tube voltage ($kVp$) and exposure ($mAs$).
5. **Forbidden Transforms:** Strictly NO vertical flips, NO heavy shearing, NO color shifts, NO elastic deformations.

---

## 10. Class Balancing Strategy

Although class imbalance is mild, square-root inverse-frequency weighting was applied to ensure the minority Tuberculosis class ($N=2,215$) receives proportional gradient emphasis:

$$w_c = \sqrt{rac{N}{C \cdot N_c}}, \quad ar{w}_c = rac{w_c}{rac{1}{C}\sum_{k=1}^C w_k}$$

### Exact Weights Applied:
| Class Index | Class Code | Training Samples ($N_c$) | Normalized Weight ($ar{w}_c$) |
|:---:|:---|:---:|:---:|
| 0 | `NORMAL` | 3,911 | **0.8630** |
| 1 | `PNEUMONIA` | 2,971 | **0.9902** |
| 2 | `TUBERCULOSIS` | 2,215 | **1.1468** |

- **Dynamic Weight Ratio:** $1.1468 / 0.8630 = 1.33:1$. This provides mild, stable regularization without causing gradient oscillations.

---

## 11. Loss Function

- **Criterion:** Standard `CrossEntropyLoss` with label smoothing ($\epsilon = 0.05$):
  $$\mathcal{L} = - \sum_{c=0}^2 ar{w}_c \left[ (1 - \epsilon) y_c + rac{\epsilon}{3} ight] \log \hat{p}_c$$
- **Rationale for Omitting Focal Loss:** Following our rigorous findings on HAM10000 (where Focal Loss attenuated gradients for borderline lesions, causing severe recall drops), standard weighted Cross-Entropy ensures continuous linear gradient flow, promoting high sensitivity across all infectious states.

---

## 12. ResNet50 Architecture

- **Backbone:** Deep 50-layer Residual Network (`torchvision.models.resnet50`).
- **Residual Connections:** Skip-connections mitigate vanishing gradients across 16 residual bottleneck blocks.
- **Feature Pooling:** Global Average Pooling compresses the final activation volume to a 2048-dimensional feature embedding.
- **Classification Head:**
  ```
  nn.Sequential(
      nn.Dropout(p=0.2),
      nn.Linear(in_features=2048, out_features=3)
  )
  ```
- **Target Conv Layer:** `model.layer4[-1]` (`torchvision.models.resnet.Bottleneck`), exposed for Grad-CAM explainability.

---

## 13. Transfer Learning Strategy

- **Initialization:** ImageNet-1K pretrained weights (`ResNet50_Weights.DEFAULT`), pre-conditioned with rich edge, texture, and contour detectors.
- **Two-Phase Transfer Protocol:**
  - **Phase 1 (Warmup - 2 Epochs):** The entire convolutional backbone is frozen; only the newly initialized 3-class linear head is optimized. This prevents large random head gradients from corrupting pretrained backbone representations.
  - **Phase 2 (Fine-Tuning - Up to 18 Epochs):** The entire network is unfrozen and trained end-to-end with differential learning rates.

---

## 14. Optimizer

- **Optimizer:** Decoupled Weight Decay AdamW (`torch.optim.AdamW`).
- **Weight Decay:** $1.0 	imes 10^{-2}$ applied across all weights to regularize against overfitting.
- **Gradient Clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` to safeguard against sudden gradient surges.

---

## 15. Learning Rates

Differential learning rates were enforced during Phase 2:
- **Convolutional Backbone LR:** $\eta_{	ext{backbone}} = 3.0 	imes 10^{-5}$ (conservative adaptation of low-level and mid-level thoracic features).
- **Classification Head LR:** $\eta_{	ext{head}} = 1.5 	imes 10^{-4}$ ($5	imes$ higher learning rate for the 3-class classifier).

---

## 16. Training Schedule

- **Scheduler:** `CosineAnnealingLR` decaying smoothly over Phase 2 epochs ($T_{\max}=18$, $\eta_{\min}=1.0 	imes 10^{-6}$).
- **Early Stopping:** Monitored **Validation Macro F1** with a strict patience of 4 epochs.
- **Total Epochs Run:** 13 epochs (Phase 1: Epochs 1–2; Phase 2: Epochs 3–13). Early stopping triggered at epoch 13 when validation Macro F1 failed to exceed the peak achieved at epoch 09.

---

## 17. Best Epoch & Validation Trajectory

- **Best Epoch:** **Epoch 09**
  - **Validation Loss:** 0.2043
  - **Validation Accuracy:** **98.97%**
  - **Validation Macro F1:** **0.9906**
  - **Validation Weighted F1:** **0.9897**
  - **Per-Class Validation F1:** `NORMAL`: 0.9881 | `PNEUMONIA`: 0.9859 | `TUBERCULOSIS`: 0.9979
- **Checkpoint Location:** `models/chest_xray/best_model.pth`

### Epoch-by-Epoch Convergence Table:
| Epoch | Phase | Train Loss | Train Acc | Val Loss | Val Acc | Val Macro F1 | Val Weighted F1 | Note |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 01 | Warmup | 0.7739 | 77.20% | 0.5730 | 88.62% | 0.8879 | 0.8863 | Head warmup |
| 02 | Warmup | 0.5318 | 85.98% | 0.4701 | 89.90% | 0.9014 | 0.8993 | Head warmup |
| 03 | Fine-Tune | 0.3124 | 94.38% | 0.2741 | 96.72% | 0.9694 | 0.9671 | Backbone unfrozen |
| 04 | Fine-Tune | 0.2561 | 96.58% | 0.2390 | 97.59% | 0.9777 | 0.9759 | |
| 05 | Fine-Tune | 0.2337 | 97.80% | 0.2185 | 98.10% | 0.9826 | 0.9810 | |
| 06 | Fine-Tune | 0.2211 | 98.24% | 0.2176 | 98.41% | 0.9853 | 0.9841 | |
| 07 | Fine-Tune | 0.2176 | 98.16% | 0.2078 | 98.92% | 0.9902 | 0.9892 | |
| 08 | Fine-Tune | 0.2074 | 98.58% | 0.2144 | 98.36% | 0.9847 | 0.9836 | Patience 1/4 |
| **09** | **Fine-Tune** | **0.2040** | **98.76%** | **0.2043** | **98.97%** | **0.9906** | **0.9897** | **BEST CHECKPOINT (Saved)** |
| 10 | Fine-Tune | 0.1988 | 99.08% | 0.2112 | 98.62% | 0.9873 | 0.9861 | Patience 1/4 |
| 11 | Fine-Tune | 0.1974 | 99.09% | 0.2043 | 98.92% | 0.9902 | 0.9892 | Patience 2/4 |
| 12 | Fine-Tune | 0.1940 | 99.19% | 0.2057 | 98.82% | 0.9892 | 0.9882 | Patience 3/4 |
| 13 | Fine-Tune | 0.1911 | 99.40% | 0.2037 | 98.97% | 0.9906 | 0.9897 | Patience 4/4 (Stopped) |

---

## 18. Test Set Evaluation: Overall Metrics

The single-pass evaluation was executed on the untouched held-out test split (**1,951 images**):

| Evaluation Metric | Test Set Result | Clinical Significance |
|:---|:---:|:---|
| **Overall Accuracy** | **98.87%** (1,929 / 1,951) | Exceptional diagnostic concordance across all classes |
| **Macro Precision** | **0.9898** | Extremely low false-positive rate across all findings |
| **Macro Recall** | **0.9898** | >98.5% sensitivity achieved for every individual class |
| **Macro F1-Score** | **0.9898** | High harmonic balance between sensitivity and specificity |
| **Weighted Precision** | **0.9888** | Class-weighted precision accounting for sample counts |
| **Weighted Recall** | **0.9887** | Class-weighted sensitivity |
| **Weighted F1-Score** | **0.9887** | Robust overall performance across the population |

---

## 19. Per-Class Metrics Table

| Class Code | Clinical Diagnostic Name | Precision | Recall | F1-Score | Test Support |
|:---|:---|:---:|:---:|:---:|:---:|
| `NORMAL` | Normal / Healthy | 0.9881 | 0.9857 | **0.9869** | 839 |
| `PNEUMONIA` | Bacterial & Viral Pneumonia | 0.9812 | 0.9859 | **0.9836** | 637 |
| `TUBERCULOSIS`| Active Pulmonary Tuberculosis | **1.0000** | **0.9979** | **0.9989** | 475 |
| **Macro Average** | | **0.9898** | **0.9898** | **0.9898** | **1,951** |
| **Weighted Average** | | **0.9888** | **0.9887** | **0.9887** | **1,951** |

---

## 20. Confusion Matrix Breakdown

```
Predicted Class ->
Actual Class     NORMAL    PNEUMONIA    TUBERCULOSIS | Total
NORMAL              827           12               0 |   839
PNEUMONIA             9          628               0 |   637
TUBERCULOSIS          1            0             474 |   475
------------------------------------------------------------
Total Predicted     837          640             474 | 1,951
```

### Normalized Diagonal (Class Recalls):
- `NORMAL`: **98.57%** (827 / 839)
- `PNEUMONIA`: **98.59%** (628 / 637)
- `TUBERCULOSIS`: **99.79%** (474 / 475)

---

## 21. Detailed Error Analysis

In pulmonary triage, the primary clinical danger is false reassurance—misclassifying an acute pulmonary infection or infectious mycobacterial disease as "Normal".

### Critical Confusion Diagnostics:
1. **Pneumonia $ightarrow$ Normal (False Negatives):**
   - Count: **9 out of 637 cases (1.41%)**
   - Clinical Context: Subtle or early-stage retrocardiac opacities or mild viral interstitial ground-glass haziness can be obscured by cardiac shadow or diaphragmatic contour.
2. **Tuberculosis $ightarrow$ Normal (False Negatives):**
   - Count: **1 out of 475 cases (0.21%)**
   - Clinical Context: 474 of 475 true TB cases were successfully identified. The single missed case demonstrated minimal apical pleural scarring with absence of prominent cavitary opacities.
3. **Normal $ightarrow$ Pneumonia (False Positives):**
   - Count: **12 out of 839 cases (1.43%)**
   - Clinical Context: Prominent bronchovascular markings, mild atelectasis, or poor inspiratory effort mimicking consolidation.
4. **Normal $ightarrow$ Tuberculosis (False Positives):**
   - Count: **0 out of 839 cases (0.00%)**
   - The model produced zero false-alarm tuberculosis predictions on healthy patients.
5. **Infection Swaps (Pneumonia $\leftrightarrow$ Tuberculosis):**
   - `PNEUMONIA` $ightarrow$ `TUBERCULOSIS`: **0 cases**
   - `TUBERCULOSIS` $ightarrow$ `PNEUMONIA`: **0 cases**
   - The model completely differentiated acute pneumonic alveolar consolidations from chronic mycobacterial cavitary/apical patterns without a single crossover error.

---

## 22. Live Inference Verification

Inference was tested using the unified entry point on authentic test samples:

### Test Sample 1: Normal Chest Radiograph
- **Image Path:** `.../test/NORMAL/jtiptj_test_IM-0021-0001.png`
- **Predicted Class:** `NORMAL`
- **Confidence:** **95.85%**
- **Probabilities:** `NORMAL`: 0.9585, `PNEUMONIA`: 0.0176, `TUBERCULOSIS`: 0.0239
- **Status:** Success

### Test Sample 2: Pneumonia
- **Image Path:** `.../test/PNEUMONIA/jtiptj_test_person100_bacteria_475.png`
- **Predicted Class:** `PNEUMONIA`
- **Confidence:** **97.48%**
- **Probabilities:** `NORMAL`: 0.0085, `PNEUMONIA`: 0.9748, `TUBERCULOSIS`: 0.0167
- **Status:** Success

### Test Sample 3: Tuberculosis
- **Image Path:** `.../test/TUBERCULOSIS/mendeley_TB.1005.png`
- **Predicted Class:** `TUBERCULOSIS`
- **Confidence:** **96.03%**
- **Probabilities:** `NORMAL`: 0.0120, `PNEUMONIA`: 0.0277, `TUBERCULOSIS`: 0.9603
- **Status:** Success

---

## 23. Unified Inference Integration

The 3-class Chest X-ray model is fully integrated into the unified inference architecture:
```python
from src.inference.predictor import predict_image, get_model, get_target_layer

# 1. Prediction via unified API
result = predict_image("path_to_xray.png", dataset_type="chest_xray")

# 2. Model resolution
model = get_model("chest_xray")

# 3. Explainability target layer
target_layer = get_target_layer("chest_xray")
```

### Complete System Regression Verification:
All four MedVision models were tested concurrently to ensure zero cross-modality regressions:
- `brain_tumor`: Active (`models/brain_tumor/best_model.pth`) $ightarrow$ Conv Target: `Bottleneck`
- `eyepacs`: Active V3 (`models/eyepacs_full/best_model.pth`) $ightarrow$ Conv Target: `Bottleneck`
- `ham10000`: Active V3 (`models/ham10000_full/best_model.pth`) $ightarrow$ Conv Target: `Bottleneck`
- `chest_xray`: Active (`models/chest_xray/best_model.pth`) $ightarrow$ Conv Target: `Bottleneck`

---

## 24. Grad-CAM Compatibility

- **Target Conv Layer Hook:** Verified that `get_target_layer("chest_xray")` directly returns `model.layer4[-1]` (`torchvision.models.resnet.Bottleneck`).
- **Feature Geometry:** The final feature volume ($2048 	imes 7 	imes 7$) maintains gradient tracking enabled during inference for downstream CAM generation.

---

## 25. Important Dataset Limitations & Provenance Notice

> [!WARNING]
> **Dataset Scope and Naming Clarification:**
> - MedVision's chest X-ray module uses a **3-class Chest X-ray dataset containing NORMAL, PNEUMONIA, and TUBERCULOSIS categories**.
> - This model is **NOT** a 14-label NIH ChestX-ray14 classifier. It does not predict atelectasis, cardiomegaly, effusion, infiltration, mass, nodule, pneumothorax, consolidation, edema, emphysema, fibrosis, pleural thickening, or hernia.
> - The standby configuration for `nih_chestxray14` is preserved separately in `src/models/model_config.py` for future work when authentic full 14-label NIH image sets and annotations become available.

---

## 26. Regulatory & Clinical Disclaimer

> [!CAUTION]
> **ACADEMIC RESEARCH PROTOTYPE ONLY:** This software and its associated deep learning models are developed strictly for research, educational, and benchmarking purposes. It is **NOT** a certified medical device, has **NOT** undergone FDA 510(k), CE-MDR, or clinical trial clearance, and must **NEVER** be used as a standalone diagnostic tool or to guide patient management. All clinical diagnoses require official evaluation by a board-certified radiologist, sputum microscopy / GeneXpert testing for tuberculosis, and clinical correlation.

---

## 27. Technical Limitations

1. **Resolution Downsampling:** Radiographs are downscaled to $224 	imes 224$ pixels. Subtle miliary tuberculosis nodules (<2 mm) or fine bibasilar crackle-associated reticular opacities may lose diagnostic sharpness compared to native $2048 	imes 2048$ resolution.
2. **Patient-Level Grouping:** Because patient IDs were unavailable in source metadata, patient-level independence could not be verified beyond strict image-hash independence.
3. **Co-morbidity Limitation:** This model assumes mutually exclusive multi-class categorization (`NORMAL` vs `PNEUMONIA` vs `TUBERCULOSIS`). In real-world clinical practice, co-infections (e.g. bacterial pneumonia superimposed on pulmonary tuberculosis in immunocompromised patients) can co-occur.

---

## 28. Recommended Future Work

1. **Higher-Resolution Thoracic Architectures:** Benchmark $384 	imes 384$ or $512 	imes 512$ inputs with ConvNeXt-Large or Swin-Transformer-V2 to preserve subtle interstitial markings.
2. **Multi-Label Expansion:** When full annotations are available, extend the model to predict concomitant thoracic findings (e.g. pleural effusion, cardiomegaly, pneumothorax).
3. **Clinical Sputum / PCR Multi-Modal Fusion:** Integrate laboratory markers (CRP, GeneXpert MTB/RIF) with imaging features for multimodal pulmonary diagnostics.

---

## 29. Official Model Status Declaration

```
======================================================================
CHEST X-RAY 3-CLASS MODEL: ACTIVE
======================================================================
```

- **Active Model Checkpoint:** `models/chest_xray/best_model.pth`
- **Training History:** `models/chest_xray/training_history.json`
- **Configuration Metadata:** `models/chest_xray/config.json`
- **Test Evaluation Report:** `results/chest_xray/test_evaluation_report.json`
- **Confusion Matrix Graphic:** `results/chest_xray/confusion_matrix.png`
