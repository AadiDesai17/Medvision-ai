"""
MedVision Disease Prediction - Master Pipeline Runner
Author: K. Gnana Saaketh (ROLE: Deep Learning / Disease Prediction)
Project: Medvision — AI Medical Imaging Assistant

Executes the complete workflow end-to-end:
1. Dataset Preparation & Stratified Splitting
2. Model Training with Validation & Checkpointing
3. Test Set Evaluation & Metric Generation
4. Standardized Prediction on a Test Image
"""

import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BEST_MODEL_PATH, SPLITS_DIR, DEFAULT_PREDICTION_OUTPUT
from src.dataset import prepare_dataset
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import generate_model_output

def main():
    print("=" * 70)
    print("      MEDVISION — AI MEDICAL IMAGING ASSISTANT (DISEASE PREDICTION)      ")
    print("      Developer: K. Gnana Saaketh | Architecture: ResNet50 Transfer Learning")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # STEP 1: Dataset Preparation
    # ---------------------------------------------------------------------
    print("\n>>> STEP 1: PREPARING DATASET & GENERATING SPLITS <<<")
    train_df, val_df, test_df = prepare_dataset(force_rebuild=False)

    # ---------------------------------------------------------------------
    # STEP 2: Model Training
    # ---------------------------------------------------------------------
    print("\n>>> STEP 2: TRAINING RESNET50 CLASSIFIER <<<")
    model, history = train_model(epochs=5, batch_size=16)

    # ---------------------------------------------------------------------
    # STEP 3: Evaluation on Unseen Test Set
    # ---------------------------------------------------------------------
    print("\n>>> STEP 3: EVALUATING ON TEST SPLIT <<<")
    metrics = evaluate_model(checkpoint_path=BEST_MODEL_PATH, splits_dir=SPLITS_DIR)

    # ---------------------------------------------------------------------
    # STEP 4: Standardized Prediction on Sample Test Image
    # ---------------------------------------------------------------------
    print("\n>>> STEP 4: GENERATING STANDARDIZED INTEGRATION PREDICTION <<<")
    sample_row = test_df.iloc[0]
    sample_image_path = sample_row["image_path"]
    sample_image_id = sample_row["image_id"]
    true_label = sample_row["class_name"]

    print(f"Testing on actual sample: {sample_image_id} (Ground Truth: {true_label})")
    payload = generate_model_output(
        image_path=sample_image_path,
        image_id=sample_image_id,
        case_id="MV-CASE-DEMO-001",
        output_json_path=DEFAULT_PREDICTION_OUTPUT
    )

    print("\n" + "=" * 70)
    print("                     PIPELINE EXECUTION COMPLETE                     ")
    print("=" * 70)
    print(json.dumps(payload, indent=2))
    print("=" * 70)
    print(f"[Summary] Artifacts successfully created:")
    print(f"  - Model Weights     : models/best_model.pth")
    print(f"  - Model Config      : models/model_config.json")
    print(f"  - Test Metrics      : outputs/metrics/test_metrics.json")
    print(f"  - Loss Curve Plot   : outputs/plots/loss_curve.png")
    print(f"  - Accuracy Curve    : outputs/plots/accuracy_curve.png")
    print(f"  - Confusion Matrix  : outputs/plots/confusion_matrix.png")
    print(f"  - Integration JSON  : outputs/predictions/model_output.json")
    print("=" * 70)

if __name__ == "__main__":
    main()
