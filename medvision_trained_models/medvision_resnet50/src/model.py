"""
Model factory for ResNet50 architecture on Chest X-ray classification.
Uses official torchvision.models.resnet50 with ImageNet-1K pretrained weights.
Classifier head is adapted to 3 classes (NORMAL, PNEUMONIA, TUBERCULOSIS)
with Dropout(p=0.2) + Linear(2048, 3), exactly matching the saved checkpoint.
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


def create_resnet50(
    num_classes: int = 3,
    pretrained: bool = True,
    dropout: float = 0.2
) -> nn.Module:
    """
    Constructs a ResNet50 model with custom 3-class classification head.
    
    Architecture:
      Backbone: torchvision.models.resnet50 (ImageNet-1K pretrained weights)
      Global Average Pooling: (B, 2048, 7, 7) -> (B, 2048)
      Head (model.fc): Sequential(
          (0): Dropout(p=dropout)
          (1): Linear(in_features=2048, out_features=num_classes)
      )
      
    This head structure perfectly matches the checkpoint keys:
      'fc.1.weight' -> shape [3, 2048]
      'fc.1.bias'   -> shape [3]
    """
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    # Attach metadata for inspection
    model.dataset_type = "chest_xray"
    model.num_classes = num_classes

    return model


def get_target_conv_layer(model: nn.Module) -> nn.Module:
    """
    Returns the final convolutional bottleneck block of ResNet50 (layer4[-1]).
    Exposed for Grad-CAM explainability and activation mapping.
    """
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    raise AttributeError("Model does not have 'layer4'. Grad-CAM layer extraction failed.")


def count_parameters(model: nn.Module) -> Tuple[int, int, int]:
    """
    Counts total, trainable, and frozen parameters in a PyTorch model.
    
    Returns:
        Tuple[int, int, int]: (total_params, trainable_params, frozen_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


if __name__ == "__main__":
    print("Testing create_resnet50 factory...")
    net = create_resnet50(num_classes=3, pretrained=False, dropout=0.2)
    total, trainable, frozen = count_parameters(net)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")
    print(f"Classification head:  {net.fc}")
    
    target_layer = get_target_conv_layer(net)
    print(f"Target Grad-CAM layer: {target_layer.__class__.__name__}")

    dummy_input = torch.randn(2, 3, 224, 224)
    out = net(dummy_input)
    print(f"Input shape: {list(dummy_input.shape)} -> Output shape: {list(out.shape)}")
    assert out.shape == (2, 3), f"Expected shape (2, 3), got {out.shape}"
    print("[OK] ResNet50 model factory verified successfully.")
