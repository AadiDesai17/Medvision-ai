"""
Medical Image Preprocessing & Clinical Data Augmentation
Author: K. Gnana Saaketh (ROLE: Deep Learning / Disease Prediction)
Project: Medvision — AI Medical Imaging Assistant
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

from src.config import (
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    DEVICE
)

def get_train_transforms():
    """
    Returns training transformations with safe medical imaging augmentations.
    Excludes unrealistic transformations (e.g. vertical flip or extreme color distortion).
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])

def get_val_test_transforms():
    """
    Returns deterministic validation and test transformations.
    Converts image to RGB, resizes to target dimension, and normalizes.
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])

def load_image_rgb(image_input):
    """
    Loads an image from a file path or accepts a PIL Image.
    Ensures the image is converted to a 3-channel RGB representation.
    """
    if isinstance(image_input, (str, Path)):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Expected file path or PIL.Image, got {type(image_input)}")
    
    # Radiographs may be Grayscale ('L'), RGBA, or Palette-based. Convert to RGB.
    return img.convert("RGB")

def preprocess_image(image_input, device=DEVICE):
    """
    Standard single-image preprocessing function for inference and Grad-CAM.
    
    Parameters:
        image_input (str, Path, or PIL.Image): Input image path or PIL image.
        device (torch.device): Device to place the preprocessed tensor on.
        
    Returns:
        torch.Tensor: Preprocessed tensor of shape [1, 3, 224, 224] ready for model input.
    """
    img = load_image_rgb(image_input)
    transform = get_val_test_transforms()
    tensor = transform(img)  # Shape: [3, 224, 224]
    tensor = tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]
    return tensor.to(device)
