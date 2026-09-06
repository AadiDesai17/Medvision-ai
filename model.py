"""
MedVision Disease Prediction Model Architecture
Backbone: Pretrained ResNet50 Transfer Learning Classifier
Grad-CAM API: Exposes target convolutional layer (layer4[-1]) for teammate Aadi
Author: K. Gnana Saaketh (ROLE: Deep Learning / Disease Prediction)
Project: Medvision — AI Medical Imaging Assistant
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from src.config import (
    NUM_CLASSES,
    DROPOUT_RATE,
    MODEL_NAME,
    ARCHITECTURE,
    MODEL_VERSION,
    BEST_MODEL_PATH,
    DEVICE
)

class MedVisionResNet50(nn.Module):
    """
    ResNet50 Transfer Learning Classifier for Medical Chest X-Ray Diagnosis.
    
    Architecture:
        - Feature Extractor: ResNet50 (conv1 through layer4)
        - Grad-CAM Target: layer4[-1] (the final bottleneck convolutional block)
        - Classification Head: AdaptiveAvgPool2d -> Dropout(0.3) -> Linear(2048, num_classes)
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, freeze_early_layers=True):
        super(MedVisionResNet50, self).__init__()
        self.num_classes = num_classes
        self.model_name = MODEL_NAME
        self.architecture = ARCHITECTURE
        self.version = MODEL_VERSION
        
        # Load weights
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # Freezing strategy:
        # Freeze initial low-level feature extractors (conv1, bn1, layer1, layer2)
        # Keep layer3, layer4, and fc unfrozen for high-level pathology adaptation
        if freeze_early_layers:
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
            for param in self.backbone.layer1.parameters():
                param.requires_grad = False
            for param in self.backbone.layer2.parameters():
                param.requires_grad = False
                
        # Replace the final fully-connected classification layer
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """Standard forward pass returning raw logits of shape [batch_size, num_classes]."""
        return self.backbone(x)

    def get_target_layer(self):
        """
        Integration Helper for Grad-CAM (Module Owner: Aadi):
        Returns the target convolutional layer for visual activation mapping.
        For ResNet50, this is the final Bottleneck block of layer4.
        """
        return self.backbone.layer4[-1]

    def get_config(self):
        """Returns model metadata and architecture specifications."""
        return {
            "name": self.model_name,
            "architecture": self.architecture,
            "version": self.version,
            "num_classes": self.num_classes,
            "in_features": 2048,
            "target_layer": "backbone.layer4[-1]"
        }

def build_model(num_classes=NUM_CLASSES, pretrained=True, freeze_early_layers=True, device=DEVICE):
    """Factory function to build and move model to the target device."""
    model = MedVisionResNet50(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_early_layers=freeze_early_layers
    )
    return model.to(device)

def load_trained_model(checkpoint_path=None, device=DEVICE):
    """
    Loads model weights from checkpoint.
    If checkpoint doesn't exist, raises a clean FileNotFoundError.
    """
    if checkpoint_path is None:
        checkpoint_path = BEST_MODEL_PATH
        
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: {checkpoint_path}.\n"
            f"Please run 'python -m src.train' or 'python run_pipeline.py' first to train the model."
        )
        
    model = build_model(pretrained=False, device=device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
