# Model Card: MedVision-ChestXRay-Classifier

---

## 1. Model Details
- **Model Name:** MedVision-ChestXRay-Classifier
- **Version:** v1.0
- **Model Architecture:** ResNet-50 (Deep Residual Learning with 50 layers)
- **Framework:** PyTorch & torchvision
- **Author / Developer:** K. Gnana Saaketh (MedVision AI Medical Imaging Assistant Team)
- **Role:** Deep Learning / Disease Prediction Feature Owner
- **Release Date:** September 2026
- **License:** Educational / Non-Commercial Research License

---

## 2. Intended Use & Clinical Scope
- **Primary Intended Use:** Automated binary screening of frontal chest radiographs (PA / AP projections) for the presence of Pneumonia vs. Normal findings.
- **Intended Users:** Medical research students, computer vision engineers, and healthcare AI demonstrators.
- **Out-of-Scope Use:** Direct clinical diagnosis or treatment planning in emergency or ICU settings without board-certified radiologist oversight.

> [!CAUTION]
> **STUDENT PROTOTYPE DISCLAIMER**  
> This model is developed as an academic deep learning engineering prototype for a collegiate engineering project. It is **NOT** an FDA-approved or CE-marked medical device. It should never be used as a standalone diagnostic tool or to replace the clinical judgment of a licensed medical practitioner.

---

## 3. Training & Dataset Details
- **Source Dataset:** National Institutes of Health (NIH) Clinical Center ChestX-ray14 Database (CVPR 2017).
- **Target Pathologies:**
  - `Normal`: Confirmed "No Finding" radiographs without detectable pulmonary or cardiac abnormalities.
  - `Pneumonia`: Radiographically confirmed consolidation, infiltrates, or air bronchograms indicative of pulmonary infection.
- **Data Splitting Strategy:**
  - 70% Training Set
  - 15% Validation Set
  - 15% Testing Set
  - Patient-level cohort partitioning to prevent data leakage between training and testing sets.
- **Preprocessing & Augmentation:**
  - Resolution: 224 x 224 pixels, 3-channel RGB.
  - ImageNet normalization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
  - Clinically safe augmentations: $\pm 10^\circ$ rotation, $p=0.5$ horizontal flip, subtle brightness/contrast adjustments.
- **Optimization:**
  - Optimizer: Adam ($\text{lr}=10^{-4}, \text{weight\_decay}=10^{-4}$)
  - Loss: Class-Weighted Cross-Entropy Loss
  - Transfer Learning: Pretrained ImageNet weights; fine-tuned `layer3`, `layer4`, and classification head with dropout ($p=0.3$).

---

## 4. Evaluation Metrics
Evaluated on an independent, held-out test split:
- Overall Accuracy
- Precision (Macro & Weighted)
- Recall / Sensitivity (Macro & Weighted)
- F1-Score (Macro & Weighted)
- Confusion Matrix

Numerical results and confusion matrix plots are saved dynamically in `outputs/metrics/test_metrics.json` and `outputs/plots/confusion_matrix.png`.

---

## 5. Teammate & System Integration
- **Output Interface:** Produces standard JSON payload containing:
  - Predicted disease label (`Normal` or `Pneumonia`)
  - Continuous Softmax confidence score ($[0.0, 1.0]$)
  - Full posterior probability distribution across all classes
- **Explainability Interface:** Provides direct programmatic access to `backbone.layer4[-1]` via `get_target_layer()` for downstream Grad-CAM saliency mapping.

---

## 6. Known Limitations & Failure Modes
1. **Co-occurring Pathologies:** In multi-morbid patients (e.g. Pneumonia co-presenting with Effusion or Atelectasis), focal opacities may be confounded.
2. **Pediatric vs. Adult Variance:** Anatomical scale differences across pediatric age groups may introduce distribution shift unless fine-tuned on dedicated pediatric cohorts.
3. **Hardware Requirements:** Inference requires ~200 MB of RAM on CPU; GPU acceleration is automatically leveraged when CUDA is present.
