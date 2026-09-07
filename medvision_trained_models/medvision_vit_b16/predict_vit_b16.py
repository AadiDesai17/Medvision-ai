import os
import sys
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_vit_b16

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "chest_xray_vit_b16" / "best_model.pth"

# Exact deterministic preprocessing matching validation/test
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_inference_model(checkpoint_path=CHECKPOINT_PATH, device="cpu"):
    """Load trained ViT-B/16 checkpoint for inference."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at: {path}\n"
            "Please ensure the model is trained before running predictions."
        )
    checkpoint = torch.load(path, map_location=device)
    class_names = checkpoint.get("class_names", ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"])
    num_classes = len(class_names)
    
    model = create_vit_b16(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names

def predict_single_image(image_path, model, class_names, device="cpu"):
    """Run deterministic inference on a single image file."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    
    try:
        with Image.open(path) as img:
            # Safely and consistently convert Grayscale ('L') to 3-channel RGB
            img_rgb = img.convert("RGB")
            tensor = INFERENCE_TRANSFORM(img_rgb).unsqueeze(0).to(device)
    except Exception as e:
        raise ValueError(f"Unable to read image file: {e}")
        
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        pred_idx = probs.argmax()
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx] * 100.0
        
    return {
        "filename": path.name,
        "prediction": pred_class,
        "confidence": confidence,
        "probabilities": {class_names[i]: float(probs[i] * 100.0) for i in range(len(class_names))}
    }

def run_interactive_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    
    print("========================================")
    print("        MEDVISION")
    print("        ViT-B/16")
    print("   CHEST X-RAY CLASSIFIER")
    print("========================================\n")
    print("Model:       ViT-B/16")
    print("Dataset:     Chest X-ray")
    print("Classes:     NORMAL / PNEUMONIA / TUBERCULOSIS")
    print(f"Checkpoint:  {CHECKPOINT_PATH.as_posix()}")
    print(f"Device:      {device_name}\n")
    
    try:
        model, class_names = load_inference_model(CHECKPOINT_PATH, device=device)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    # Check if image path provided via command line argument
    if len(sys.argv) > 1 and sys.argv[1].strip():
        raw_path = sys.argv[1].strip().strip('"').strip("'")
        paths_to_process = [raw_path]
        interactive = False
    else:
        interactive = True

    while True:
        if interactive:
            try:
                raw_input_path = input("Enter path to Chest X-ray image (or 'q' to quit): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
                
            if not raw_input_path:
                continue
            if raw_input_path.lower() in ["q", "quit", "exit"]:
                print("Exiting.")
                break
                
            image_path_str = raw_input_path.strip('"').strip("'")
        else:
            image_path_str = paths_to_process.pop(0)

        image_path = Path(image_path_str)
        if not image_path.exists() or not image_path.is_file():
            print(f"\n[Error] Invalid file path: '{image_path}'. Please try again.\n")
            if not interactive:
                break
            continue
            
        try:
            result = predict_single_image(image_path, model, class_names, device=device)
            
            print("\n========================================")
            print("MEDVISION - ViT-B/16")
            print("Chest X-ray Classification")
            print("========================================")
            print("\nImage:")
            print(result['filename'])
            print("\nPrediction:")
            print(result['prediction'])
            print("\nConfidence:")
            print(f"{result['confidence']:.2f}%\n")
            print("Class Probabilities:\n")
            for c_name in class_names:
                prob = result["probabilities"][c_name]
                print(f"{c_name + ':':<15} {prob:>6.2f}%")
            print("\n========================================\n")
            
        except Exception as e:
            print(f"\n[Error processing image]: {e}\n")
            
        if not interactive:
            break

if __name__ == "__main__":
    run_interactive_demo()
