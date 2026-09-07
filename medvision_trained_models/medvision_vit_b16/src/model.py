import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights

def create_vit_b16(num_classes=3, pretrained=True, dropout=0.0):
    """
    Build and return a Vision Transformer ViT-B/16 model.
    
    Args:
        num_classes (int): Number of output classes (default 3: NORMAL, PNEUMONIA, TUBERCULOSIS).
        pretrained (bool): Whether to initialize backbone with ImageNet-1K pretrained weights.
        dropout (float): Dropout rate for the classification head and encoder.
        
    Returns:
        nn.Module: Configured ViT-B/16 model with adapted classification head.
    """
    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
    else:
        weights = None
        
    # Instantiate official torchvision ViT-B/16 architecture
    model = models.vit_b_16(weights=weights, dropout=dropout)
    
    # In torchvision's ViT implementation, the classification head is model.heads.head
    in_features = model.heads.head.in_features  # 768 for ViT-B/16
    
    # Replace linear projection head with exactly num_classes outputs
    model.heads.head = nn.Linear(in_features=in_features, out_features=num_classes)
    
    # Initialize head weights cleanly following standard ViT initialization
    nn.init.trunc_normal_(model.heads.head.weight, mean=0.0, std=0.02)
    nn.init.zeros_(model.heads.head.bias)
    
    return model

def count_parameters(model):
    """Return dictionary with parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params
    }

if __name__ == "__main__":
    # Self-test when invoked directly
    test_model = create_vit_b16(num_classes=3, pretrained=False)
    params = count_parameters(test_model)
    print("Model created successfully.")
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Head structure: {test_model.heads.head}")
    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_out = test_model(dummy_input)
    print(f"Forward pass output shape: {dummy_out.shape}")
    assert dummy_out.shape == (2, 3), f"Expected shape (2, 3), got {dummy_out.shape}"
    print("Self-test passed!")
