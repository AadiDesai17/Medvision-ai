# MedVision — Standalone ResNet50 Chest X-ray Retraining Project

## Project Status: Stage 4 Completed (Held-Out Test Evaluation & Inference Verified)
This repository is an isolated, standalone project for the 3-class Chest X-ray disease prediction model based on the **ResNet50** architecture (`torchvision.models.resnet50`). It executes the standardized retraining protocol aligned with the **ViT-B/16** and **Swin-Tiny** benchmark experiments.

> **CRITICAL RULE ENFORCED:** **STAGE 4 COMPLETE — HELD-OUT TEST EVALUATION AND INFERENCE VERIFIED.**  
> The standardized model achieved **98.87% Test Accuracy** and **0.9896 Test Macro F1** on the 1,951 held-out test images. The source dataset is read-only. The old project and other architecture projects remain untouched.

---

## 1. Multi-Architecture Benchmark Context
MedVision evaluates deep learning vision architectures under a strictly standardized medical radiography protocol:
1. **ResNet50** (50-layer Residual Network — this project)
2. **ViT-B/16** (Vision Transformer Base with 16x16 patch resolution — `medvision_vit_b16`)
3. **Swin-Tiny** (Hierarchical Shifted-Window Transformer Tiny — `medvision_swin_tiny`)

### Clinical Classification Targets:
- **Class `0`:** `NORMAL` (Clear lung fields, normal cardiothoracic ratio, sharp costophrenic recesses)
- **Class `1`:** `PNEUMONIA` (Acute bacterial and viral alveolar/interstitial opacities/consolidation)
- **Class `2`:** `TUBERCULOSIS` (Active or fibrotic mycobacterial pulmonary infiltrates, cavitary lesions, adenopathy)

---

## 2. Standalone Project Structure
```
medvision_resnet50/
├── configs/
│   └── resnet50.yaml                    # Full standardized training configuration
├── models/
│   └── chest_xray_resnet50/
│       ├── historical_best_model.pth    # Safely preserved historical checkpoint (90.00 MB)
│       └── best_model.pth               # Newly trained standardized checkpoint (Epoch 13, 90.00 MB)
├── results/
│   └── chest_xray_resnet50/
│       ├── STAGE1_AUDIT_REPORT.md       # Complete 24-point Stage 1 Audit Report
│       ├── STAGE2_SMOKE_TEST_REPORT.md  # Comprehensive Stage 2 GPU Smoke Test Report
│       ├── STAGE3_TRAINING_REPORT.md    # Full Stage 3 Standardized Retraining Report
│       ├── STAGE4_FINAL_TEST_REPORT.md  # Full Stage 4 Test Evaluation & Inference Report
│       ├── test_evaluation_report.json  # Held-out test evaluation metrics
│       ├── test_errors.csv              # Error analysis log for 22 misclassified test images
│       ├── confusion_matrix.png         # Test split confusion matrix plot
│       ├── training_config.json         # Snapshot of active training configuration
│       ├── training_history.json        # Epoch-by-epoch loss, metrics, and timing log
│       ├── training_summary.json        # High-level training summary and optimal metrics
│       ├── training_curves.png          # High-resolution convergence curves plot
│       ├── HISTORICAL_MODEL_REPORT.md   # Historical reference report
│       ├── historical_confusion_matrix.png
│       ├── historical_test_evaluation_report.json
│       ├── historical_training_config.json
│       └── historical_training_history.json
├── src/
│   ├── dataset.py                       # SafeRGB, standardized transforms, dataset loader, class weights
│   ├── evaluate.py                      # Standalone test split evaluator & confusion matrix plotter
│   ├── model.py                         # ResNet50 factory, ImageNet weights, Dropout(0.2)+Linear(2048, 3)
│   ├── train.py                         # Standardized training pipeline (with full logging & curve plotting)
│   └── utils.py                         # Seed reproducibility, multi-class metrics, checkpoint utilities
├── predict_resnet50.py                  # Live single-image inference CLI & interactive prompt
├── smoke_test.py                        # Real-data GPU forward/backward & AMP verification
├── requirements.txt                     # Minimal system requirements
└── README.md                            # Comprehensive project documentation
```

---

## 3. Final Held-Out Test Performance (Stage 4)
Evaluated on **1,951** independent test images (`NORMAL`: 839, `PNEUMONIA`: 637, `TUBERCULOSIS`: 475):

- **Overall Accuracy:** **`98.87%`** (1,929 / 1,951 correct)
- **Macro Precision:** **`0.9896`** (98.96%)
- **Macro Recall:** **`0.9896`** (98.96%)
- **Macro F1-Score:** **`0.9896`** (98.96%)
- **Weighted F1-Score:** **`0.9887`** (98.87%)
- **Test Error Count:** **`22`** errors (**`1.13%`**)

### Per-Class Test Metrics:
- **NORMAL:** Precision: `0.9857` | Recall: `0.9881` | F1: `0.9869` | Support: 839
- **PNEUMONIA:** Precision: `0.9874` | Recall: `0.9827` | F1: `0.9851` | Support: 637
- **TUBERCULOSIS:** Precision: `0.9958` | Recall: `0.9979` | F1: `0.9968` | Support: 475

---

## 4. Preliminary Cross-Architecture Benchmark Comparison

| Architecture | Model Family | Total Parameters | Test Accuracy | Test Macro F1 | Test Weighted F1 | Best Val Macro F1 | Training Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Swin-Tiny (`swin_t`)** | Shifted-Window Transformer | 27.52M | **99.18%** | **0.9926** | **0.9918** | **0.9926** | ~29.1 mins |
| **ViT-B/16 (`vit_b_16`)** | Vision Transformer Base | 85.80M | **98.97%** | **0.9905** | **0.9897** | **0.9902** | ~42.3 mins |
| **ResNet50 (`resnet50`)** | Residual Convolutional Network | **23.51M** | **98.87%** | **0.9896** | **0.9887** | **0.9904** | **28.30 mins** |

*Note: ConvNeXt-Tiny is pending and will be added to the final benchmark.*
