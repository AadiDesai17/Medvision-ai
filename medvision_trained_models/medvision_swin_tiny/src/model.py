"""
Model factory for Swin Transformer Tiny (Swin-Tiny) architecture.
Uses torchvision official implementation and pretrained weights.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models


def create_swin_tiny(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """
    Creates a Swin-Tiny model with a custom classification head.
    
    Args:
        num_classes: Number of target output classes (default: 3).
        pretrained: Whether to load ImageNet-1K pretrained weights.
        
    Returns:
        nn.Module: Swin-Tiny model with adapted classification head.
    """
    if pretrained:
        weights = models.Swin_T_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.swin_t(weights=weights)

    # Inspect and dynamically obtain the classifier in_features (expected 768)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Counts total and trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Tuple[int, int]: (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    print("Testing create_swin_tiny factory...")
    net = create_swin_tiny(num_classes=3, pretrained=False)
    total, trainable = count_parameters(net)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    dummy_input = torch.randn(2, 3, 224, 224)
    out = net(dummy_input)
    print(f"Input shape: {dummy_input.shape} -> Output shape: {out.shape}")
    assert out.shape == (2, 3), f"Expected shape (2, 3), got {out.shape}"
    print("[OK] Swin-Tiny model factory verified.")
