"""
MedVision ConvNeXt-Tiny Model Definition
Chest X-Ray 3-Class Disease Prediction (NORMAL, PNEUMONIA, TUBERCULOSIS)

Architecture: torchvision.models.convnext_tiny
Pretrained Weights: ConvNeXt_Tiny_Weights.DEFAULT (ImageNet-1K v1)
Head: 3-class linear classifier replacing original 1000-class head
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


def build_convnext_tiny(
    num_classes: int = 3,
    pretrained: bool = True
) -> nn.Module:
    """
    Builds the ConvNeXt-Tiny model with ImageNet pretrained weights and
    replaces the final classification layer with a 3-class output head.

    Args:
        num_classes: Number of target categories (default: 3).
        pretrained: Whether to load ImageNet-1K pretrained weights.

    Returns:
        torch.nn.Module: ConvNeXt-Tiny model adapted for 3-class prediction.
    """
    if pretrained:
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    else:
        weights = None

    model = models.convnext_tiny(weights=weights)

    # In torchvision convnext_tiny, model.classifier is:
    # Sequential(
    #   (0): LayerNorm2d((768,), eps=1e-06, elementwise_affine=True),
    #   (1): Flatten(start_dim=1, end_dim=-1),
    #   (2): Linear(in_features=768, out_features=1000, bias=True)
    # )
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes, bias=True)

    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Counts total and trainable parameters in the model.

    Args:
        model: PyTorch neural network module.

    Returns:
        Tuple[int, int]: (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_target_layer_for_gradcam(model: nn.Module) -> nn.Module:
    """
    Returns the target convolutional block for Grad-CAM / XAI feature visualization.
    In ConvNeXt-Tiny, the final convolutional feature block resides in model.features[-1][-1].
    """
    return model.features[-1][-1]


if __name__ == "__main__":
    print("Testing ConvNeXt-Tiny architecture...")
    model = build_convnext_tiny(num_classes=3, pretrained=True)
    total, trainable = count_parameters(model)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Original classifier layer 2 in_features: 768")
    print(f"Modified classifier: {model.classifier}")
    
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print(f"Forward output shape: {out.shape}")
    assert out.shape == (2, 3), f"Expected shape (2, 3), got {out.shape}"
    print("ConvNeXt-Tiny architecture test PASSED.")
