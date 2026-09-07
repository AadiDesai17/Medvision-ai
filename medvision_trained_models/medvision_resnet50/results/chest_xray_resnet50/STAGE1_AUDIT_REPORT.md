# Stage 1 Audit & Standardized Retraining Preparation Report: ResNet50

**Project:** MedVision Chest X-ray 3-Class Disease Prediction  
**Model Architecture:** ResNet50 (`torchvision.models.resnet50`)  
**Stage:** Stage 1 — Audit + Standardized Retraining Preparation  
**Timestamp:** 2026-09-07  
**Status:** COMPLETE & VERIFIED  

---

## 1. Project Isolation Status

- **Standalone Workspace Path:** `C:\Users\AADI\.gemini\antigravity\scratch\medvision_resnet50`
- **Old Project Path (Historical Reference):** `C:\Users\AADI\.gemini\antigravity\scratch\medvision_disease_prediction`
- **ViT-B/16 Project Path:** `C:\Users\AADI\.gemini\antigravity\scratch\medvision_vit_b16`
- **Swin-Tiny Project Path:** `C:\Users\AADI\.gemini\antigravity\scratch\medvision_swin_tiny`
- **Isolation Verification:**
  - The standalone project is completely self-contained with its own executable scripts (`src/model.py`, `src/dataset.py`, `src/utils.py`, `src/train.py`, `src/evaluate.py`, `predict_resnet50.py`), configuration (`configs/resnet50.yaml`), and dependencies.
  - Zero imports from the old project or other model directories.
  - The old project at `medvision_disease_prediction` remains strictly **UNTOUCHED** (read-only historical reference).
  - The pre-existing checkpoint in `medvision_resnet50/models/chest_xray_resnet50/best_model.pth` (94,375,761 bytes) has been safely preserved as `models/chest_xray_resnet50/historical_best_model.pth`.
  - The historical reference checkpoint in `medvision_disease_prediction\models\chest_xray\best_model.pth` exists and was confirmed unmodified.

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

## 4. Image Format, Mode, and Dimensions

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

- **Total VRAM:** `7.996 GB` (~8,188 MB)
- **Observed Peak Memory in Stage 1 GPU Smoke Test:** `853.75 MB` (plenty of headroom under 8 GB capacity)

---

## 14. ResNet50 Availability

- **torchvision ResNet50 Constructor:** `<function resnet50 at 0x0000023234F19220>` (`torchvision.models.resnet50`)
- **Status:** Fully available and verified.

---

## 15. ImageNet Pretrained Weight Availability

- **Weights Enum:** `torchvision.models.ResNet50_Weights.DEFAULT`
- **Resolved Weight Target:** `ResNet50_Weights.IMAGENET1K_V2`
- **Status:** Verified and cached locally; instantiated cleanly without network errors.

---

## 16. Proposed Model Architecture

- **Backbone:** Standard ResNet50 (`torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)`)
- **Classifier Head:**
  ```python
  model.fc = nn.Sequential(
      nn.Dropout(p=0.2),
      nn.Linear(in_features=2048, out_features=3)
  )
  ```
- **Total Parameters:** `23,514,179` (23.51M)
- **Trainable Parameters:** `23,514,179` (100% trainable during standardized end-to-end retraining)
- **Grad-CAM Hook Layer:** `model.layer4[-1]` (Bottleneck block)

---

## 17. Proposed Preprocessing & Augmentation

Strictly aligned with the standardized ViT-B/16 and Swin-Tiny experimental protocol:

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

## 18. Proposed Optimizer, Loss, and Scheduler

- **Optimizer:** Decoupled Weight Decay AdamW (`torch.optim.AdamW`)
  - Learning Rate: `3e-5` (`0.00003`)
  - Weight Decay: `0.01`
  - Betas: `(0.9, 0.999)`
  - Epsilon: `1e-8`
- **Loss Function:** Moderated square-root inverse-frequency weighted `nn.CrossEntropyLoss`:
  - `NORMAL` (Class 0): `0.8630`
  - `PNEUMONIA` (Class 1): `0.9902`
  - `TUBERCULOSIS` (Class 2): `1.1468`
  - Label Smoothing: **STRICTLY 0.0 (NO label smoothing)**
- **Learning Rate Scheduler:** `CosineAnnealingLR`
  - $T_{\max}$: `15` epochs
  - $\eta_{\min}$: `1e-6`

---

## 19. Proposed Batch, Gradient Accumulation, and AMP Configuration

- **Physical Batch Size:** `16`
- **Gradient Accumulation Steps:** `2`
- **Effective Batch Size:** `32` ($16 \times 2 = 32$)
- **Automatic Mixed Precision (AMP):** CUDA FP16 mixed precision using PyTorch AMP API (`torch.amp.autocast('cuda', dtype=torch.float16)`)
- **GradScaler:** `torch.amp.GradScaler('cuda', init_scale=1024)` with safe initialization
- **Gradient Clipping:** `max_norm = 1.0` applied after unscaling

---

## 20. Proposed Seed

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

## 21. Proposed Early Stopping & Model-Selection Rule

- **Primary Metric:** **Validation Macro F1 ONLY**
- **Evaluation Frequency:** After every epoch on the 1,950-image validation split
- **Checkpoint Saving:** Best model weights saved strictly when validation Macro F1 exceeds the previous best
- **Early Stopping Patience:** `5` epochs without improvement
- **Maximum Epochs:** `15`

---

## 22. Required Standalone Python Files & Directories

All required files are verified present, standalone, and executable in `C:\Users\AADI\.gemini\antigravity\scratch\medvision_resnet50`:

1. `src/model.py` — ResNet50 factory with ImageNet pretrained weights, 3-class classifier head (`Dropout(0.2) + Linear(2048, 3)`), parameter counter, and Grad-CAM target layer.
2. `src/dataset.py` — SafeRGB grayscale-to-RGB conversion, standardized transforms, PyTorch Dataset, and class weight calculator.
3. `src/utils.py` — Seed management, multi-class metrics (Accuracy, Macro/Weighted Precision/Recall/F1, per-class metrics, confusion matrix), and plotting.
4. `src/train.py` — Full reproducible standardized training pipeline with safety guard against Stage 1 execution.
5. `src/evaluate.py` — Full evaluation script for held-out test split (1,951 images) with diagnostic error analysis.
6. `predict_resnet50.py` — Standalone single-image CLI and interactive inference runner with clinical disclaimer.
7. `configs/resnet50.yaml` — Complete standardized configuration parameter file.
8. `requirements.txt` — Minimal runtime environment dependencies.
9. `README.md` — Project documentation, audit records, and standardized protocol summary.
10. `models/chest_xray_resnet50/` — Checkpoint directory containing preserved `historical_best_model.pth`.
11. `results/chest_xray_resnet50/` — Output directory for audit reports, evaluation JSONs, and plots.

---

## 23. Identified Risks and Mitigations

| Risk | Impact | Mitigation Strategy |
|:---|:---|:---|
| **GPU Out-Of-Memory (OOM)** | Interrupted training | Physical batch size 16 with FP16 AMP consumed only 853.75 MB peak VRAM during real batch smoke testing, well below the 8.00 GB ceiling. |
| **Silent Overwrite of Historical Weights** | Loss of historical baseline | Pre-existing checkpoint was backed up as `models/chest_xray_resnet50/historical_best_model.pth`. Old project at `medvision_disease_prediction` remains untouched. |
| **Inconsistent Augmentation with ViT/Swin** | Unfair cross-architecture benchmarking | Random horizontal flip, vertical flip, and random resized crop are strictly disabled; rotation ($\pm 7^\circ$) and subtle jitter (0.05) match ViT/Swin exactly. |
| **Label Smoothing Discrepancy** | Protocol divergence | `label_smoothing` is explicitly set to `0.0` (disabled) in both YAML config and loss instantiation. |
| **Premature Training in Stage 1** | Violation of Stage 1 constraints | `src/train.py` contains a mandatory safety guard preventing execution without explicit Stage 2 `--execute-training` command-line flag. |

---

## 24. Final Stage 1 Statement

**STAGE 1 COMPLETE — NO FULL TRAINING PERFORMED.**
