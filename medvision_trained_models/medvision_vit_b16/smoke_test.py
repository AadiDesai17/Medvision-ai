import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW

# Set path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed, load_config, calculate_class_weights, get_gpu_memory
from src.model import create_vit_b16, count_parameters
from src.dataset import get_dataloaders

def run_smoke_test():
    print("==================================================")
    print("MedVision ViT-B/16 Pre-Training Smoke Test")
    print("==================================================")
    
    set_seed(42)
    config = load_config(PROJECT_ROOT / "configs" / "vit_b16.yaml")
    
    # Check CUDA
    assert torch.cuda.is_available(), "CUDA is required for GPU smoke test!"
    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    
    # Step 1 & 2: Load pretrained weights & replace head with 3 classes
    print("\n1 & 2. Loading torchvision ViT-B/16 pretrained weights and replacing head with 3 classes...")
    model = create_vit_b16(num_classes=3, pretrained=True).to(device)
    params = count_parameters(model)
    print(f"   Architecture: ViT-B/16")
    print(f"   Total Parameters:     {params['total']:,}")
    print(f"   Trainable Parameters: {params['trainable']:,}")
    print(f"   Classification Head:  {model.heads.head}")
    assert model.heads.head.out_features == 3, f"Expected 3 out features, got {model.heads.head.out_features}"
    
    # Step 3: Move model to CUDA
    print("\n3. Moving model to CUDA...")
    print(f"   Model device: {next(model.parameters()).device}")
    
    # Step 4: Load a very small batch of real Chest X-ray images
    print("\n4. Loading real Chest X-ray batch from training split...")
    train_loader, val_loader, test_loader, class_names, class_to_idx = get_dataloaders(config)
    
    # Verify split counts
    print(f"   Train samples verified: {len(train_loader.dataset)} (expected: 9097)")
    print(f"   Val samples verified:   {len(val_loader.dataset)} (expected: 1950)")
    print(f"   Test samples verified:  {len(test_loader.dataset)} (expected: 1951)")
    assert len(train_loader.dataset) == 9097, f"Train count mismatch: {len(train_loader.dataset)}"
    assert len(val_loader.dataset) == 1950, f"Val count mismatch: {len(val_loader.dataset)}"
    assert len(test_loader.dataset) == 1951, f"Test count mismatch: {len(test_loader.dataset)}"
    
    # Fetch real batch
    real_images, real_targets = next(iter(train_loader))
    # Slice to small batch of 4 for smoke test
    batch_size = min(4, real_images.size(0))
    real_images = real_images[:batch_size].to(device)
    real_targets = real_targets[:batch_size].to(device)
    print(f"   Batch shape: {real_images.shape} (dtype: {real_images.dtype})")
    print(f"   Batch targets: {real_targets.tolist()} ({[class_names[t] for t in real_targets.tolist()]})")
    
    # Setup criterion and optimizer
    class_weights = calculate_class_weights(train_loader.dataset.targets, num_classes=3, method="sqrt_inverse").to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    init_scale = float(config["training"]["amp"].get("init_scale", 1024.0))
    scaler = torch.amp.GradScaler('cuda', init_scale=init_scale)
    
    # Step 5: Perform forward pass with AMP
    print("\n5. Performing forward pass under torch.amp.autocast('cuda')...")
    torch.cuda.reset_peak_memory_stats(0)
    optimizer.zero_grad()
    
    with torch.amp.autocast('cuda'):
        outputs = model(real_images)
        print(f"   Forward output shape: {outputs.shape}")
        assert outputs.shape == (batch_size, 3), f"Expected shape ({batch_size}, 3), got {outputs.shape}"
        
        # Step 6: Calculate loss
        print("\n6. Calculating loss...")
        loss = criterion(outputs, real_targets)
        print(f"   Loss value: {loss.item():.4f}")
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        
    # Step 7: Perform backward pass
    print("\n7. Performing backward pass with GradScaler...")
    scaler.scale(loss).backward()
    
    # Step 8: Optimizer step
    print("\n8. Performing optimizer step...")
    scaler.unscale_(optimizer)
    
    # Step 10: Confirm gradients are finite
    print("\n10. Confirming all gradients are finite...")
    all_finite = True
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if not torch.isfinite(param.grad).all():
                print(f"   Gradient non-finite in parameter: {name}")
                all_finite = False
                break
    assert all_finite, "Found non-finite gradients during smoke test!"
    print("   All trainable parameter gradients are verified FINITE.")
    
    scaler.step(optimizer)
    scaler.update()
    print("   Optimizer step and scaler update completed successfully.")
    
    # Step 9: Confirm AMP works
    print("\n9. Confirming AMP operation...")
    current_scale = scaler.get_scale()
    print(f"   GradScaler current scale: {current_scale}")
    assert current_scale > 0, "GradScaler scale invalid!"
    print("   AMP is functioning correctly.")
    
    # Step 11: Report GPU memory usage
    print("\n11. GPU Memory Report:")
    allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
    peak = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    print(f"   Currently Allocated: {allocated:.2f} MB")
    print(f"   Peak Allocated:      {peak:.2f} MB ({peak/1024:.2f} GB)")
    print(f"   Currently Reserved:  {reserved:.2f} MB")
    print(f"   Total VRAM Available: {total:.2f} MB ({total/1024:.2f} GB)")
    print(f"   VRAM Utilization:    {peak/total*100:.1f}% of 8 GB")
    
    print("\n==================================================")
    print("SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_smoke_test()
