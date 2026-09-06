"""
MedVision Disease Prediction - Inference & Integration Contract
Author: K. Gnana Saaketh (ROLE: Deep Learning / Disease Prediction)
Project: Medvision — AI Medical Imaging Assistant
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

from src.config import (
    BEST_MODEL_PATH,
    MODEL_NAME,
    ARCHITECTURE,
    MODEL_VERSION,
    MODULE_NAME,
    CLASS_NAMES,
    DEFAULT_PREDICTION_OUTPUT,
    DEVICE
)
from src.utils import save_json
from src.preprocessing import preprocess_image
from src.model import load_trained_model, build_model

# Global model cache to avoid re-loading weights on repeated inferences
_CACHED_MODEL = None

def get_inference_model(model_path=None, device=DEVICE):
    """Retrieves cached model or instantiates and loads from checkpoint."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None and model_path is None:
        return _CACHED_MODEL

    if model_path is None:
        model_path = BEST_MODEL_PATH
        
    model_path = Path(model_path)
    if not model_path.exists():
        # Fallback to untrained architecture with warning if model file not yet trained
        print(f"[Warning] Checkpoint '{model_path}' not found. Using untrained ResNet50 for demonstration.")
        model = build_model(pretrained=True, device=device)
    else:
        model = load_trained_model(checkpoint_path=model_path, device=device)

    model.eval()
    if model_path == BEST_MODEL_PATH:
        _CACHED_MODEL = model
    return model

def predict_image(image_path, model=None, model_path=None, device=DEVICE):
    """
    Core Prediction API for team members (Streamlit UI, RAG, Grad-CAM).
    
    Parameters:
        image_path (str, Path, or PIL.Image): Input medical radiograph.
        model (nn.Module, optional): Pre-loaded model instance.
        model_path (str or Path, optional): Custom checkpoint path.
        device (torch.device): Inference device (CUDA or CPU).
        
    Returns:
        dict: Standard prediction dictionary containing:
            - 'class': Predicted diagnosis ("Normal" or "Pneumonia")
            - 'confidence': Softmax confidence float
            - 'probabilities': Dict of probabilities for every class
    """
    if model is None:
        model = get_inference_model(model_path=model_path, device=device)

    # 1. Preprocess Image
    tensor = preprocess_image(image_path, device=device)

    # 2. Run Inference
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)  # Shape: [num_classes]

    # 3. Format Probabilities and Prediction
    probabilities = {
        class_name: round(float(probs[idx].item()), 4)
        for idx, class_name in enumerate(CLASS_NAMES)
    }

    pred_idx = int(torch.argmax(probs).item())
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probabilities[pred_class]

    return {
        "class": pred_class,
        "confidence": confidence,
        "probabilities": probabilities
    }

def generate_model_output(
    image_path,
    image_id="image_001",
    case_id="MV-MOCK-001",
    model=None,
    model_path=None,
    output_json_path=DEFAULT_PREDICTION_OUTPUT,
    device=DEVICE
):
    """
    Produces the exact standardized JSON integration contract output.
    
    Parameters:
        image_path (str or Path): Path to image.
        image_id (str): Identifier for input image.
        case_id (str): MedVision patient / case identification.
        model (nn.Module, optional): Pre-loaded model.
        model_path (str, optional): Model checkpoint path.
        output_json_path (Path, optional): Destination file path for JSON output.
        device (torch.device): Device.
        
    Returns:
        dict: The complete standardized MedVision payload.
    """
    prediction_result = predict_image(
        image_path=image_path,
        model=model,
        model_path=model_path,
        device=device
    )

    output_payload = {
        "case_id": case_id,
        "model": {
            "name": MODEL_NAME,
            "architecture": ARCHITECTURE,
            "version": MODEL_VERSION
        },
        "input": {
            "image_id": str(image_id),
            "module": MODULE_NAME,
            "preprocessed": True
        },
        "prediction": {
            "class": prediction_result["class"],
            "confidence": prediction_result["confidence"],
            "probabilities": prediction_result["probabilities"]
        },
        "status": "success"
    }

    if output_json_path:
        save_json(output_payload, output_json_path)
        print(f"[Prediction] Output payload saved to: {output_json_path}")

    return output_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedVision Chest X-Ray Disease Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to input chest radiograph image")
    parser.add_argument("--image_id", type=str, default="image_001", help="Image ID")
    parser.add_argument("--case_id", type=str, default="MV-MOCK-001", help="Case ID")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path")
    parser.add_argument("--output", type=str, default=str(DEFAULT_PREDICTION_OUTPUT), help="Output JSON path")
    args = parser.parse_args()

    result = generate_model_output(
        image_path=args.image,
        image_id=args.image_id,
        case_id=args.case_id,
        model_path=args.checkpoint,
        output_json_path=args.output
    )

    import json
    print("\n--- STANDARDIZED INTEGRATION OUTPUT ---")
    print(json.dumps(result, indent=2))
