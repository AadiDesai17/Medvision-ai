# MedVision Trained Models

Four standardized Chest X-ray 3-class disease prediction models trained on the same dataset and evaluated on the same held-out test set.

## Target Disease Classes
- **NORMAL**: Healthy chest radiograph without active pulmonary infection or infiltrates
- **PNEUMONIA**: Bacterial or viral consolidation/infiltrate
- **TUBERCULOSIS**: Active pulmonary tuberculosis presentation

---

## Final Standardized Held-Out Test Performance

All models were evaluated against the exact same held-out test set of 1,951 images under identical test protocols:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet50** | 98.87% | 98.96% | 98.96% | 98.96% | 98.87% |
| **ViT-B/16** | 98.97% | 99.06% | 99.05% | 99.05% | 98.97% |
| **ConvNeXt-Tiny** | 99.03% | 99.14% | 99.08% | 99.11% | 99.03% |
| **Swin-Tiny** | **99.18%** | **99.29%** | **99.24%** | **99.26%** | **99.18%** |

> **Primary Benchmark Finding**: **Swin-Tiny achieved the best overall held-out test performance** across all key classification metrics, reaching **99.18% Accuracy** and **99.26% Macro F1**.

---

## Dataset Specifications & Split Integrity

- **Total Dataset Size**: 12,998 chest radiograph images
- **Split Distribution**:
  - **Train**: 9,097 images (70.0%)
  - **Validation**: 1,950 images (15.0%)
  - **Test (Held-Out)**: 1,951 images (15.0%)
- **Number of Classes**: 3 (`NORMAL`, `PNEUMONIA`, `TUBERCULOSIS`)
- **Image-Level Split Separation**: Verified. Hash-based and filename-based checks confirm zero data leakage between train, validation, and test partitions at the image level.
- **Patient-Level Separation Note**: Patient-level separation could not be independently verified due to anonymized public dataset naming conventions.
- **Backbone Initialization**: All architectures utilize official ImageNet-pretrained backbone weights before domain fine-tuning.
- **Evaluation Protocol**: The test set was strictly held out and evaluated only once after final convergence.

---

## Repository Structure

```
medvision_trained_models/
├── README.md                          # Top-level benchmark documentation (this file)
├── medvision_resnet50/                # Standalone ResNet-50 project
│   ├── configs/                       # resnet50.yaml configuration
│   ├── models/chest_xray_resnet50/    # best_model.pth (90.00 MB), historical_best_model.pth
│   ├── results/chest_xray_resnet50/   # Confusion matrix, curves, audit & evaluation reports
│   ├── src/                           # dataset, evaluate, model, train, utils source code
│   ├── predict_resnet50.py            # Standalone single/batch inference CLI
│   ├── smoke_test.py                  # Integration smoke test suite
│   ├── requirements.txt               # Dependency specifications
│   └── README.md                      # Model-specific technical documentation
├── medvision_vit_b16/                 # Standalone Vision Transformer (ViT-B/16) project
│   ├── configs/                       # vit_b16.yaml configuration
│   ├── models/chest_xray_vit_b16/     # best_model.pth (327.36 MB, Git LFS), config.json
│   ├── results/chest_xray_vit_b16/    # Confusion matrix, curves, final evaluation report
│   ├── src/                           # dataset, evaluate, model, train, utils source code
│   ├── predict_vit_b16.py             # Standalone single/batch inference CLI
│   ├── smoke_test.py                  # Integration smoke test suite
│   ├── requirements.txt               # Dependency specifications
│   └── README.md                      # Model-specific technical documentation
├── medvision_swin_tiny/               # Standalone Swin Transformer (Swin-T) project (Best Performer)
│   ├── configs/                       # swin_tiny.yaml configuration, baseline_config.json
│   ├── models/chest_xray_swin_tiny/   # best_model.pth (315.39 MB, Git LFS)
│   ├── results/chest_xray_swin_tiny/  # Confusion matrix, curves, audit & evaluation reports
│   ├── src/                           # dataset, evaluate, model, train, utils source code
│   ├── predict_swin_tiny.py           # Standalone single/batch inference CLI
│   ├── smoke_test.py                  # Integration smoke test suite
│   ├── requirements.txt               # Dependency specifications
│   └── README.md                      # Model-specific technical documentation
└── medvision_convnext_tiny/           # Standalone Modern ConvNet (ConvNeXt-Tiny) project
    ├── configs/                       # convnext_tiny.yaml configuration
    ├── models/chest_xray_convnext_tiny/ # best_model.pth (318.62 MB, Git LFS)
    ├── results/chest_xray_convnext_tiny/ # Confusion matrix, curves, audit & evaluation reports
    ├── src/                           # dataset, evaluate, model, train, utils source code
    ├── predict_convnext_tiny.py       # Standalone single/batch inference CLI
    ├── smoke_test.py                  # Integration smoke test suite
    ├── requirements.txt               # Dependency specifications
    └── README.md                      # Model-specific technical documentation
```

---

## Git LFS Tracking Notice

The PyTorch checkpoint weights (`*.pth`) across these four standalone models total **~1.14 GB**:
- Several individual checkpoint files exceed GitHub's standard 100 MB file limit (ViT-B/16: 327.36 MB, Swin-Tiny: 315.39 MB, ConvNeXt-Tiny: 318.62 MB).
- All `.pth` checkpoint files are managed and tracked via **Git LFS** (`git lfs track "*.pth"`) via `.gitattributes`.
- Clone this repository using `git lfs clone` or ensure `git lfs install` has been run before pulling checkpoints.

---

## Clinical Disclaimer

> **IMPORTANT MEDICAL NOTICE**:
> 
> - **Model confidence is not clinical certainty**: Softmax probabilities output by these models reflect mathematical distance in feature space, not clinical certainty or absolute medical truth.
> - **Research and experimental use only**: These models and code are developed solely for research and experimental purposes.
> - **Not validated for primary clinical diagnosis**: These models are **not FDA/CE certified medical devices** and must **never** be used as a standalone or primary tool for diagnostic medical decision-making without independent review by a board-certified radiologist.
