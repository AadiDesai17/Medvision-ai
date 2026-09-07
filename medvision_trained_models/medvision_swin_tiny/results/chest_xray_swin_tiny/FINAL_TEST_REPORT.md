# MedVision — Swin-Tiny Chest X-ray Test Evaluation

## Model
- **Architecture:** Swin Transformer Tiny (`torchvision.models.swin_t`)
- **Pretrained weights:** ImageNet-1K (`torchvision.models.Swin_T_Weights.IMAGENET1K_V1`)
- **Number of parameters:** 27,521,661 (100% trainable)
- **Checkpoint path:** `models\chest_xray_swin_tiny\best_model.pth` (Epoch 15)

## Dataset
- **Dataset path:** `C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia`
- **Train count:** 9,097
- **Validation count:** 1,950
- **Test count:** 1951
- **Class names:** NORMAL, PNEUMONIA, TUBERCULOSIS
- **Class mapping:** NORMAL: 0, PNEUMONIA: 1, TUBERCULOSIS: 2

## Test Metrics

| Metric | Score | Percentage |
|---|---:|---:|
| Accuracy | 0.9918 | 99.18% |
| Macro Precision | 0.9929 | 99.29% |
| Macro Recall | 0.9924 | 99.24% |
| Macro F1 | 0.9926 | 99.26% |
| Weighted F1 | 0.9918 | 99.18% |

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.9881 | 0.9928 | 0.9905 | 839 |
| PNEUMONIA | 0.9905 | 0.9843 | 0.9874 | 637 |
| TUBERCULOSIS | 1.0000 | 1.0000 | 1.0000 | 475 |

## Confusion Matrix

```
                NORMAL   PNEUMONIA   TUBERCULOSIS   Total
NORMAL             833           6                0     839
PNEUMONIA           10         627                0     637
TUBERCULOSIS         0           0              475     475
```

Reference plot: `results/chest_xray_swin_tiny/confusion_matrix.png`

## Evaluation Conditions
- The test set was held out during training and model development.
- The model checkpoint was selected strictly using validation Macro F1.
- No test-set tuning, threshold tuning, or post-hoc calibration was performed.
- Deterministic test preprocessing was applied: Grayscale to RGB -> Resize(224, 224) -> ToTensor() -> ImageNet Normalization.
- Evaluation executed using the trained Swin-Tiny checkpoint on an NVIDIA GeForce RTX 4060 Laptop GPU.
