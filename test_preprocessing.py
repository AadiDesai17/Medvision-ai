"""
Unit Tests for Image Preprocessing
Author: K. Gnana Saaketh (ROLE: Deep Learning / Disease Prediction)
"""

import pytest
import numpy as np
from PIL import Image
import torch

from src.preprocessing import (
    get_train_transforms,
    get_val_test_transforms,
    load_image_rgb,
    preprocess_image
)

def test_grayscale_to_rgb_conversion(tmp_path):
    """Ensure single-channel grayscale radiographs are properly converted to 3-channel RGB."""
    gray_img = Image.fromarray((np.random.rand(300, 300) * 255).astype(np.uint8), mode="L")
    img_path = tmp_path / "test_gray.png"
    gray_img.save(img_path)

    loaded = load_image_rgb(img_path)
    assert loaded.mode == "RGB"
    assert loaded.size == (300, 300)

def test_preprocess_image_tensor_properties(tmp_path):
    """Verify output tensor shape, data type, and normalization."""
    dummy_img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
    img_path = tmp_path / "dummy.png"
    dummy_img.save(img_path)

    tensor = preprocess_image(img_path, device=torch.device("cpu"))
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32

    # Verify not all zeros or NaNs
    assert not torch.isnan(tensor).any()
    assert tensor.abs().sum() > 0

def test_preprocess_invalid_image_path():
    """Verify clean FileNotFoundError when image doesn't exist."""
    with pytest.raises(FileNotFoundError):
        preprocess_image("C:/non_existent_folder/missing_image.png")
