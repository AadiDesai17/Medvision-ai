# Stage 1 Audit & Standardized Retraining Preparation Report: ConvNeXt-Tiny

**Project:** MedVision Chest X-ray 3-Class Disease Prediction  
**Model Architecture:** ConvNeXt-Tiny (`torchvision.models.convnext_tiny`)  
**Stage:** Stage 1 — Audit + Standardized Retraining Preparation  
**Timestamp:** 2026-09-07  
**Status:** COMPLETE & VERIFIED  

---

## 1. Project Isolation Status

- **Standalone Workspace Path:** `C:\Users\AADI\.gemini\antigravity\scratch\medvision_convnext_tiny`
- **Old Project Path (Historical Reference):** `C:\Users\AADI\.gemini\antigravity\scratch\medvision_disease_prediction`
- **ResNet50 Project Path:** `C:\Users\AADI\.gemini\antigravity\scratch\medvision_resnet50`
- **ViT-B/16 Project Path:** `C:\Users\AADI\.gemini\antigravity\scratch\medvision_vit_b16`
- **Swin-Tiny Project Path:** `C:\Users\AADI\.gemini\antigravity\scratch\medvision_swin_tiny`
- **Isolation Verification:**
  - The standalone project is completely self-contained with its own executable scripts (`src/model.py`, `src/dataset.py`, `src/utils.py`, `src/train.py`, `src/evaluate.py`, `predict_convnext_tiny.py`, `smoke_test.py`), configuration (`configs/convnext_tiny.yaml`), and dependencies (`requirements.txt`).
  - Zero imports from ResNet50, ViT-B/16, Swin-Tiny, or the old project.
  - All existing project directories remain strictly **UNTOUCHED**.

---

## 2. Dataset Path

- **Dataset Root Path:** `C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia`
- **Access Mode:** STRICTLY READ-ONLY. Zero files modified, deleted, or created within the dataset root.
- **Top-Level Directories:** `.cache`, `.gitattributes`, `README.md`, `test/`, `train/`, `val/`
- **Combined Folder Check:** Confirmed absent in root (`combined/` folder does not exist; no duplicate directory included).

---

## 3. Dataset Counts by Split and Class

Every single directory was programmatically audited and verified against the expected cohort splits:

| Split | Class | Expected Count | Actual Count | Match Status |
|:---|:---|:---:|:---:|:---:|
| **train** | `NORMAL` | 3,911 | 3,911 | Verified Match |
| **train** | `PNEUMONIA` | 2,971 | 2,971 | Verified Match |
| **train** | `TUBERCULOSIS` | 2,215 | 2,215 | Verified Match |
| **train** | **TOTAL** | **9,097** | **9,097** | **Verified Match** |
| **val** | `NORMAL` | 838 | 838 | Verified Match |
| **val** | `PNEUMONIA` | 637 | 637 | Verified Match |
| **val** | `TUBERCULOSIS` | 475 | 475 | Verified Match |
| **val** | **TOTAL** | **1,950** | **1,950** | **Verified Match** |
| **test** | `NORMAL` | 839 | 839 | Verified Match |
| **test** | `PNEUMONIA` | 637 | 637 | Verified Match |
| **test** | `TUBERCULOSIS` | 475 | 475 | Verified Match |
| **test** | **TOTAL** | **1,951** | **1,951** | **Verified Match** |
| **GRAND TOTAL** | **ALL CLASSES** | **12,998** | **12,998** | **Verified Match (100%)** |

No new random train/validation/test split was created; the exact predefined split structure is preserved.

---

## 4. Image Properties (Format, Mode, Dimensions)

- **File Format:** 100% `.png` (all 12,998 images). Zero non-PNG files in class folders.
- **Image Color Mode:** 100% single-channel grayscale (`L`). All images require `SafeRGB` channel replication (`(I, I, I)`) to form 3-channel tensors compatible with ImageNet pretrained convolutional filters without modifying source files.
- **Spatial Resolution:** Exactly **512 x 512 pixels** across 100% of the images in the cohort:
  - Minimum dimensions: `(512, 512)`
  - Maximum dimensions: `(512, 512)`

---

## 5. Corrupt / Invalid Image Count

- **Integrity Validation:** 12,998 / 12,998 images loaded, decoded, and verified via PIL (`img.verify()`).
- **Corrupt Images Detected:** Exactly **0** (zero).
- **Unreadable / Truncated Files:** Exactly **0** (zero).
- **Zero-Byte Files:** Exactly **0** (zero).

---

## 6. Cross-Split Duplicate / Hash Findings

Cryptographic SHA-256 digests were computed for all 12,998 images across splits:

- **Train Split Hashes:** 9,097 unique SHA-256 hashes (0 internal duplicates).
- **Validation Split Hashes:** 1,950 unique SHA-256 hashes (0 internal duplicates).
- **Test Split Hashes:** 1,951 unique SHA-256 hashes (0 internal duplicates).
- **Cross-Split Overlap:**
  - $\text{Train} \cap \text{Validation}$: **0 overlapping hashes**
  - $\text{Train} \cap \text{Test}$: **0 overlapping hashes**
  - $\text{Validation} \cap \text{Test}$: **0 overlapping hashes**
- **Conclusion:** There is strictly zero image-level data leakage between the training, validation, and held-out test splits.

---

## 7. Patient-Level Separation Limitation

> [!IMPORTANT]
> **Explicit Scientific Disclosure:**
> **Only IMAGE-LEVEL separation is claimed based on exact hash verification.**
> Patient-level separation cannot be claimed or guaranteed. The underlying dataset comprises public domain and aggregated radiological collections where filenames are synthetic or prefixed with dataset source keys (e.g. `jtiptj_test_IM-0021-0001.png`, `jtiptj_test_person100_bacteria_477.png`, `mendeley_TB.1.png`). No patient identifier tags, DICOM metadata headers, accession numbers, or hospital MRNs are available in the dataset files. Therefore, while zero duplicate images exist across partitions, repeated radiographic exposures from the same individual across splits cannot be independently ruled out.

---

## 8. Python Version

- **Python:** `3.14.4` (tags/v3.14.4:23116f9, Apr 7 2026, 14:10:54) [MSC v.1944 64 bit (AMD64)]

---

## 9. PyTorch Version

- **PyTorch:** `2.13.0+cu126`

---

## 10. torchvision Version

- **torchvision:** `0.28.0+cu126`

---

## 11. CUDA Availability

- **CUDA Available:** `True`
- **Device Count:** 1

---

## 12. GPU Name

- **GPU:** `NVIDIA GeForce RTX 4060 Laptop GPU`
- **Compute Capability:** `(8, 9)` (Ada Lovelace architecture)

---

## 13. VRAM

- **Total VRAM:** `7.996 GB` (~8,187.5 MB)
- **Observed Peak Memory in Stage 1 GPU Smoke Test:**
  - Peak VRAM Allocated: `1,139.89 MB`
  - Peak VRAM Reserved: `1,258.00 MB`
  - VRAM Headroom: `~6.74 GB` free capacity remaining under the 8.00 GB hardware threshold.

---

## 14. ConvNeXt-Tiny Availability

- **torchvision ConvNeXt-Tiny Constructor:** `torchvision.models.convnext_tiny`
- **Status:** Fully available and verified.

---

## 15. ImageNet Pretrained Weight Availability

- **Weights Enum:** `torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT`
- **Resolved Weight Target:** `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`
- **Weight Checkpoint:** Locally cached at `C:\Users\AADI/.cache\torch\hub\checkpoints\convnext_tiny-983f1562.pth` (109 MB).
- **Status:** Verified and cached locally; instantiated cleanly without network errors.

---

## 16. Parameter Count

- **Original ImageNet Model:**
  - Total Parameters: `28,589,128` (28.59M)
- **Modified 3-Class Model:**
  - Total Parameters: `27,822,435` (27.82M)
  - Trainable Parameters: `27,822,435` (100% trainable during standardized end-to-end retraining)
  - Classifier Head Parameters: 768 * 3 + 3 = 2,307 (Linear) + 768 * 2 = 1,536 (LayerNorm2d) = `3,843` parameters.

---

## 17. Model Architecture

- **Backbone:** Standard ConvNeXt-Tiny (`torchvision.models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)`)
  - Features stages: 4 hierarchical convolutional stages (depths [3, 3, 9, 3], channels [96, 192, 384, 768])
  - Stage 0-7 CNBlocks with 7x7 depthwise convolutions, LayerNorm, inverted bottleneck 1x1 convolutions, GELU activations, and stochastic depth.
- **Global Pooling:** `nn.AdaptiveAvgPool2d((1, 1))`
- **Original Classifier:**
  ```
  Sequential(
    (0): LayerNorm2d((768,), eps=1e-06, elementwise_affine=True, bias=True)
    (1): Flatten(start_dim=1, end_dim=-1)
    (2): Linear(in_features=768, out_features=1000, bias=True)
  )
  ```
- **Final 3-Class Classifier:**
  ```
  Sequential(
    (0): LayerNorm2d((768,), eps=1e-06, elementwise_affine=True, bias=True)
    (1): Flatten(start_dim=1, end_dim=-1)
    (2): Linear(in_features=768, out_features=3, bias=True)
  )
  ```
- **Grad-CAM Target Layer:** `model.features[-1][-1]` (Final CNBlock in stage 7)

---

## 18. Preprocessing & Augmentations

Strictly aligned with the standardized cross-architecture benchmarking protocol:

- **Grayscale Handling:** `SafeRGB()` — replicates 1-channel grayscale to 3-channel RGB `(I, I, I)` in memory. Does NOT modify files on disk.
- **Training Preprocessing:**
  1. `SafeRGB()`
  2. `transforms.Resize((224, 224))`
  3. `transforms.RandomRotation(degrees=7)`
  4. `transforms.ColorJitter(brightness=0.05, contrast=0.05)`
  5. `transforms.ToTensor()`
  6. `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- **Strictly Prohibited Augmentations:**
  - `RandomHorizontalFlip`: Strictly DISABLED
  - `RandomVerticalFlip`: Strictly DISABLED
  - `RandomResizedCrop`: Strictly DISABLED
- **Validation, Test, and Inference Preprocessing:**
  1. `SafeRGB()`
  2. `transforms.Resize((224, 224))`
  3. `transforms.ToTensor()`
  4. `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

---

## 19. Class Weights

Moderated square-root inverse-frequency weights computed on the 9,097 training images:
- `NORMAL` (Class 0): **0.8630**
- `PNEUMONIA` (Class 1): **0.9902**
- `TUBERCULOSIS` (Class 2): **1.1468**

---

## 20. Loss Function

- Weighted Cross-Entropy Loss: `nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)`
- **Label Smoothing:** Strictly **0.0 (NO label smoothing)**.

---

## 21. Optimizer

- **Optimizer:** Decoupled Weight Decay AdamW (`torch.optim.AdamW`)
  - Learning Rate: `3e-5` (`0.00003`)
  - Weight Decay: `0.01`
  - Betas: `(0.9, 0.999)`
  - Epsilon: `1e-8`

---

## 22. Scheduler

- **Learning Rate Scheduler:** `CosineAnnealingLR` (`torch.optim.lr_scheduler.CosineAnnealingLR`)
  - $T_{\max}$: `15` epochs
  - $\eta_{\min}$: `1e-6`

---

## 23. Batch Configuration

- **Physical Batch Size:** `16`
- **Gradient Accumulation Steps:** `2`
- **Effective Batch Size:** `32` ($16 \times 2 = 32$)

---

## 24. AMP Configuration

- **Automatic Mixed Precision (AMP):** CUDA FP16 mixed precision using current PyTorch API (`torch.amp.autocast('cuda', dtype=torch.float16)`)
- **GradScaler:** `torch.amp.GradScaler('cuda', init_scale=1024)`
- **Gradient Clipping:** `max_norm = 1.0` applied after unscaling

---

## 25. Seed & Reproducibility Controls

- **Deterministic Seed:** `42`
- **Reproducibility Controls:**
  - `random.seed(42)`
  - `np.random.seed(42)`
  - `torch.manual_seed(42)`
  - `torch.cuda.manual_seed_all(42)`
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - `os.environ["PYTHONHASHSEED"] = "42"`

---

## 26. Early Stopping

- **Early Stopping Patience:** `5` epochs without improvement
- **Maximum Epochs:** `15`

---

## 27. Model Selection Criterion

- **Primary Selection Metric:** **VALIDATION MACRO F1 ONLY**
- **Evaluation Frequency:** Evaluated after every epoch on the 1,950-image validation split.
- **Best Model Checkpoint:** Overwritten only when validation Macro F1 strictly exceeds the previous best.

---

## 28. Required Python Files & Directories

All required files are verified present, standalone, and executable in `C:\Users\AADI\.gemini\antigravity\scratch\medvision_convnext_tiny`:

1. `src/model.py` — ConvNeXt-Tiny factory with ImageNet pretrained weights, 3-class classifier head (`LayerNorm2d(768) + Flatten() + Linear(768, 3)`), parameter counter, and Grad-CAM target layer.
2. `src/dataset.py` — SafeRGB grayscale-to-RGB conversion, standardized transforms, PyTorch Dataset, and standardized class weights.
3. `src/utils.py` — Seed management, multi-class metrics (Accuracy, Macro/Weighted Precision/Recall/F1, per-class metrics, confusion matrix), and plotting.
4. `src/train.py` — Full reproducible standardized training pipeline with safety guard against Stage 1 execution.
5. `src/evaluate.py` — Full evaluation script for held-out test split (1,951 images) with diagnostic error analysis and safety guard.
6. `predict_convnext_tiny.py` — Standalone single-image CLI and interactive inference runner with clinical disclaimer.
7. `smoke_test.py` — Hardware, model, dataset, AMP, and VRAM non-training verification runner.
8. `configs/convnext_tiny.yaml` — Complete standardized configuration parameter file.
9. `requirements.txt` — Minimal runtime environment dependencies.
10. `README.md` — Project documentation, audit records, and standardized protocol summary.
11. `models/chest_xray_convnext_tiny/` — Checkpoint destination directory.
12. `results/chest_xray_convnext_tiny/` — Output directory for audit reports, evaluation JSONs, and plots.

---

## 29. Identified Risks and Mitigations

| Risk | Impact | Mitigation Strategy |
|:---|:---|:---|
| **GPU Out-Of-Memory (OOM)** | Interrupted training | Physical batch size 16 with FP16 AMP consumed only 1,139.89 MB peak VRAM during real batch smoke testing, leaving ~6.74 GB of headroom under the 8.00 GB ceiling. |
| **Cross-Architecture Contamination** | Inconsistent codebase | Created an entirely isolated directory `medvision_convnext_tiny` with zero imports from ResNet50, ViT, or Swin. |
| **Inconsistent Augmentations** | Unfair benchmarking | Prohibited augmentations (`RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomResizedCrop`) are strictly disabled; rotation ($\pm 7^\circ$) and subtle jitter (0.05) match standard protocol. |
| **Label Smoothing Discrepancy** | Protocol divergence | `label_smoothing` is explicitly set to `0.0` (disabled) in both YAML config and loss instantiation. |
| **Premature Training in Stage 1** | Violation of Stage 1 constraints | `src/train.py` contains a mandatory safety guard preventing execution without explicit Stage 2/3 `--execute-training` command-line flag. |
| **Premature Test Set Snooping** | Data leakage / bias | `src/evaluate.py` contains a mandatory safety guard preventing execution without explicit `--execute-test-eval` flag. |

---

## 30. Stage 2 Plan

Stage 2 represents the **Smoke Test & Baseline Verification Review**:
1. Review the Stage 1 audit findings and ensure all 30 requirements are satisfied.
2. Confirm that the standalone non-training smoke test has executed cleanly with zero errors, producing `results/chest_xray_convnext_tiny/stage1_smoke_test_metrics.json`.
3. Verify that the 100% trainable parameter count (27,822,435) matches the exact architectural specification.
4. Prepare execution flags for Stage 3 (Full Standardized Retraining) upon explicit user approval.

---

STAGE 1 COMPLETE — NO FULL TRAINING PERFORMED.
