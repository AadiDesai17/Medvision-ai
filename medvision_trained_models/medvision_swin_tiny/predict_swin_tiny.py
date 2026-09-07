"""
Inference script for Swin-Tiny Chest X-ray model.
Accepts an input image and outputs predicted disease class and probabilities.
Loads trained checkpoint from models/chest_xray_swin_tiny/best_model.pth.
Includes interactive mode and CLI mode.
"""
import argparse
import sys
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F

# Ensure src is in Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from dataset import CLASSES, IDX_TO_CLASS, get_transforms
from model import create_swin_tiny
from utils import load_checkpoint


def predict_image(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    transform: any,
) -> None:
    img_file = Path(image_path)
    if not img_file.exists():
        print(f"[ERROR] File not found: {img_file}")
        return

    try:
        with Image.open(img_file) as img:
            img_rgb = img.convert("RGB")
    except Exception as e:
        print(f"[ERROR] Unable to open image {img_file}: {e}")
        return

    tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(tensor)
        else:
            logits = model(tensor)

        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        pred_idx = int(torch.argmax(logits, dim=1).item())
        pred_class = IDX_TO_CLASS[pred_idx]
        confidence = probs[pred_idx]

    print("\n" + "-" * 40)
    print(f"Image: {img_file.name}")
    print("\nPrediction:")
    print(f"{pred_class}")
    print("\nConfidence:")
    print(f"{confidence * 100:.2f}% (exact: {confidence:.6f})")
    print("\nClass probabilities:")
    print(f"\nNORMAL:\n{probs[0] * 100:.2f}% (exact: {probs[0]:.6f})")
    print(f"\nPNEUMONIA:\n{probs[1] * 100:.2f}% (exact: {probs[1]:.6f})")
    print(f"\nTUBERCULOSIS:\n{probs[2] * 100:.2f}% (exact: {probs[2]:.6f})")
    print("-" * 40)
    print("Notice: Model prediction probabilities reflect statistical confidence based on training")
    print("data distributions and must not be interpreted as a clinical medical diagnosis.")


def main():
    parser = argparse.ArgumentParser(description="Run Swin-Tiny inference on a Chest X-ray image")
    parser.add_argument("--image", type=str, default=None, help="Path to input X-ray image (.png)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/chest_xray_swin_tiny/best_model.pth",
        help="Path to trained checkpoint (.pth)",
    )
    args = parser.parse_args()

    ckpt_file = Path(args.checkpoint)
    if not ckpt_file.exists():
        print(f"[ERROR] Checkpoint not found at: {ckpt_file}")
        print("Training must be completed before running inference.")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_swin_tiny(num_classes=3, pretrained=False).to(device)
    load_checkpoint(ckpt_file, model, map_location=str(device))
    model.eval()

    transform = get_transforms("test", img_size=224)

    if args.image:
        predict_image(args.image, model, device, transform)
    else:
        print(f"Swin-Tiny Inference Demo (Device: {device})")
        print(f"Loaded checkpoint: {ckpt_file}")
        while True:
            try:
                user_input = input("\nEnter path to Chest X-ray image (or 'q' to quit): ").strip().strip('"\'')
                if not user_input or user_input.lower() == "q":
                    print("Exiting inference demo.")
                    break
                predict_image(user_input, model, device, transform)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting inference demo.")
                break


if __name__ == "__main__":
    main()
