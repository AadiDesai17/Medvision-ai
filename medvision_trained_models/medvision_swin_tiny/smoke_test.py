"""
Stage 2 GPU Smoke Test for Swin-Tiny on Chest X-ray dataset.
Executes a single controlled forward + backward step on a real data batch (B=16)
using CUDA, AMP FP16, and weighted CrossEntropyLoss.
Verifies finite loss, finite gradients, memory safety, and model adaptability.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from dataset import (
    CLASSES,
    IDX_TO_CLASS,
    ChestXRayDataset,
    compute_class_weights,
    get_transforms,
)
from model import count_parameters, create_swin_tiny
from utils import set_seed


def run_gpu_smoke_test(batch_size: int = 16) -> bool:
    print("=" * 70)
    print("MEDVISION SWIN-TINY: STAGE 2 REAL-DATA GPU SMOKE TEST")
    print("=" * 70)

    # 1. Deterministic Seed
    set_seed(42)
    print("[1/10] Seed fixed to 42.")

    # 2. CUDA & Device Check
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. GPU smoke test cannot proceed.")
        return False

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"[2/10] CUDA Device: {gpu_name} ({vram_gb:.2f} GB VRAM, Compute Capability: {compute_cap})")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    # 3. Real Training Dataset & DataLoader
    dataset_root = Path(r"C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia")
    train_dir = dataset_root / "train"
    if not train_dir.exists():
        print(f"[ERROR] Dataset train directory does not exist: {train_dir}")
        return False

    train_ds = ChestXRayDataset(train_dir, transform=get_transforms("train", img_size=224))
    print(f"[3/10] Training dataset loaded: {len(train_ds):,} real images across {len(CLASSES)} classes.")

    # Programmatic class weights
    class_weights = compute_class_weights(train_ds).to(device)
    for idx, (cls, w) in enumerate(zip(CLASSES, class_weights.tolist())):
        print(f"       Class {idx} ({cls:<12}): weight = {w:.4f}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # 4. Fetch One Real Batch
    batch_images, batch_targets = next(iter(train_loader))
    print(f"[4/10] Real data batch fetched: Images shape = {list(batch_images.shape)}, Targets = {batch_targets.tolist()}")
    assert batch_images.shape == (batch_size, 3, 224, 224), f"Expected shape ({batch_size}, 3, 224, 224), got {batch_images.shape}"
    assert len(batch_targets) == batch_size, f"Expected {batch_size} targets, got {len(batch_targets)}"

    # 5. Load ImageNet Pretrained Swin-Tiny with 3-Class Head
    print("[5/10] Instantiating ImageNet-pretrained Swin-Tiny model...")
    model = create_swin_tiny(num_classes=3, pretrained=True).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"       Total Parameters:     {total_params:,}")
    print(f"       Trainable Parameters: {trainable_params:,}")
    assert trainable_params > 0, "Model has 0 trainable parameters!"

    # 6. Optimizer, Criterion, and AMP GradScaler
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    print("[6/10] Configured AdamW (lr=3e-5) + Weighted CE Loss + AMP GradScaler (init_scale=1024.0).")

    # 7. Move Batch to GPU & Execute Forward Pass with AMP FP16
    batch_images = batch_images.to(device)
    batch_targets = batch_targets.to(device)

    model.train()
    optimizer.zero_grad()

    print("[7/10] Running forward pass under torch.autocast('cuda', dtype=torch.float16)...")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(batch_images)
        print(f"       Forward Output Shape: {list(outputs.shape)} (Expected: [{batch_size}, 3])")
        assert outputs.shape == (batch_size, 3), f"Unexpected output shape: {outputs.shape}"
        assert torch.isfinite(outputs).all(), "Non-finite values detected in forward pass outputs!"

        loss = criterion(outputs, batch_targets)
        loss_val = float(loss.item())
        print(f"       Weighted CE Loss:     {loss_val:.4f}")
        assert torch.isfinite(loss), f"Non-finite loss detected: {loss_val}"

    # 8. Backward Pass & Gradient Verification
    print("[8/10] Running backward pass via scaler.scale(loss).backward()...")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    # Check finite gradients
    grad_norms = []
    has_nan_or_inf = False
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            if not torch.isfinite(p.grad).all():
                print(f"[ERROR] Non-finite gradient in: {name}")
                has_nan_or_inf = True
            grad_norms.append(p.grad.norm().item())

    assert not has_nan_or_inf, "Non-finite gradients encountered during backward pass!"
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
    print(f"       Gradients verified finite across {len(grad_norms)} tensors. Total norm: {total_norm:.4f}")

    # 9. Optimizer Step & Scaler Update
    print("[9/10] Performing single optimizer step (scaler.step + scaler.update)...")
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    print("       Optimizer step successful. Weights updated cleanly.")

    # 10. Memory Usage & Safety Audit
    mem_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
    mem_res = torch.cuda.memory_reserved(0) / (1024 ** 2)
    max_mem_alloc = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    print("[10/10] GPU VRAM Metrics:")
    print(f"        Allocated VRAM: {mem_alloc:.2f} MB")
    print(f"        Reserved VRAM:  {mem_res:.2f} MB")
    print(f"        Peak VRAM:      {max_mem_alloc:.2f} MB ({max_mem_alloc / 1024:.2f} GB / {vram_gb:.2f} GB)")

    # Cleanup
    del batch_images, batch_targets, outputs, loss, model, optimizer, scaler
    torch.cuda.empty_cache()

    # Safety checks
    models_dir = Path(r"C:\Users\AADI\.gemini\antigravity\scratch\medvision_swin_tiny\models\chest_xray_swin_tiny")
    results_dir = Path(r"C:\Users\AADI\.gemini\antigravity\scratch\medvision_swin_tiny\results\chest_xray_swin_tiny")
    created_checkpoints = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt"))
    created_results = list(results_dir.glob("*.json"))

    print("\n--- SAFETY & ISOLATION VERIFICATION ---")
    print(f"Checkpoints created in models/: {len(created_checkpoints)} (Expected: 0)")
    print(f"Metrics created in results/:   {len(created_results)} (Expected: 0)")
    assert len(created_checkpoints) == 0, "Violated constraint: checkpoint created during smoke test!"
    assert len(created_results) == 0, "Violated constraint: results created during smoke test!"
    print("Full training performed:      FALSE (1 step only)")

    print("\n" + "=" * 70)
    print("STAGE 2 REAL-DATA GPU SMOKE TEST PASSED WITH 100% SUCCESS!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_gpu_smoke_test(batch_size=16)
    if not success:
        sys.exit(1)
