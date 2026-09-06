"""
Unit Tests for Dataset Preparation & PyTorch Dataset
Author: K. Gnana Saaketh (ROLE: Deep Learning / Disease Prediction)
"""

import pytest
import pandas as pd
from pathlib import Path
import torch

from src.config import SPLITS_DIR, LOCAL_NIH_IMAGES_DIR, LOCAL_NIH_CSV_PATH
from src.dataset import prepare_dataset, ChestXRayDataset
from src.preprocessing import get_val_test_transforms

@pytest.fixture(scope="module")
def dataset_splits():
    """Builds or loads dataset splits."""
    if not LOCAL_NIH_IMAGES_DIR.exists() or not LOCAL_NIH_CSV_PATH.exists():
        pytest.skip("Local NIH dataset not found; skipping dataset integration test.")
    train_df, val_df, test_df = prepare_dataset(force_rebuild=False)
    return train_df, val_df, test_df

def test_split_files_exist(dataset_splits):
    """Verify split manifests exist on disk."""
    train_df, val_df, test_df = dataset_splits
    assert (SPLITS_DIR / "train_split.csv").exists()
    assert (SPLITS_DIR / "val_split.csv").exists()
    assert (SPLITS_DIR / "test_split.csv").exists()
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

def test_no_data_leakage(dataset_splits):
    """Ensure zero image leakage between train, validation, and test splits."""
    train_df, val_df, test_df = dataset_splits
    train_imgs = set(train_df["image_id"])
    val_imgs = set(val_df["image_id"])
    test_imgs = set(test_df["image_id"])

    assert len(train_imgs.intersection(val_imgs)) == 0, "Data leakage between Train and Val!"
    assert len(train_imgs.intersection(test_imgs)) == 0, "Data leakage between Train and Test!"
    assert len(val_imgs.intersection(test_imgs)) == 0, "Data leakage between Val and Test!"

def test_classes_present_in_splits(dataset_splits):
    """Ensure both Normal (0) and Pneumonia (1) are present in every split."""
    train_df, val_df, test_df = dataset_splits
    for s_df in [train_df, val_df, test_df]:
        labels = set(s_df["label"])
        assert 0 in labels, "Normal class missing from split!"
        assert 1 in labels, "Pneumonia class missing from split!"

def test_pytorch_dataset_loader(dataset_splits):
    """Verify ChestXRayDataset loads images and outputs correct tensor shape."""
    _, _, test_df = dataset_splits
    dataset = ChestXRayDataset(test_df, transform=get_val_test_transforms())
    assert len(dataset) == len(test_df)

    img_tensor, label, img_id = dataset[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 224, 224)
    assert label in [0, 1]
    assert isinstance(img_id, str)
