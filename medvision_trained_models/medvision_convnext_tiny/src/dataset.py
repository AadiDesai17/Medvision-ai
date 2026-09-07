"""
MedVision Dataset & Transform Pipeline
Chest X-Ray 3-Class Disease Prediction (NORMAL, PNEUMONIA, TUBERCULOSIS)

Standardized Protocol:
- Convert grayscale X-rays to RGB in memory (SafeRGB)
- Input resolution: 224x224
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Train augmentations: RandomRotation(7 degrees), ColorJitter(brightness=0.05, contrast=0.05)
- Strictly prohibited: RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop
- Deterministic Validation & Test transforms
"""

import os
from typing import Tuple, Dict, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Predefined classes and standardized mappings
CLASS_NAMES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Standardized class weights from moderated square-root inverse frequencies
STANDARDIZED_CLASS_WEIGHTS = [0.8630, 0.9902, 1.1468]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SafeRGB:
    """
    Converts any single-channel grayscale or non-RGB PIL Image to 3-channel RGB in memory.
    Strictly read-only; does not alter files on disk.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            return img.convert("RGB")
        return img


def get_transforms() -> Dict[str, transforms.Compose]:
    """
    Builds torchvision transforms adhering strictly to the standardized experimental protocol.
    """
    train_transform = transforms.Compose([
        SafeRGB(),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomRotation(degrees=7),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        SafeRGB(),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
    }


class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for MedVision Chest X-Ray cohort.
    Loads images directly from split directories (train / val / test).
    """
    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Class directory not found: {cls_dir}")
            
            cls_idx = CLASS_TO_IDX[cls_name]
            fnames = sorted(os.listdir(cls_dir))
            for fname in fnames:
                if fname.lower().endswith(".png"):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        with Image.open(path) as img:
            if self.transform is not None:
                img_tensor = self.transform(img)
            else:
                img_tensor = transforms.functional.to_tensor(SafeRGB()(img))
        return img_tensor, label, path


def get_standardized_class_weights(device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Returns the exact standardized class weights tensor:
    NORMAL: 0.8630, PNEUMONIA: 0.9902, TUBERCULOSIS: 1.1468
    """
    weights = torch.tensor(STANDARDIZED_CLASS_WEIGHTS, dtype=torch.float32)
    if device is not None:
        weights = weights.to(device)
    return weights


def get_dataloaders(
    dataset_root: str,
    physical_batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Constructs DataLoaders for train, val, and test splits.
    """
    transform_dict = get_transforms()
    loaders = {}

    for split in ["train", "val", "test"]:
        shuffle = (split == "train")
        ds = ChestXRayDataset(
            dataset_root=dataset_root,
            split=split,
            transform=transform_dict[split],
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=physical_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return loaders


if __name__ == "__main__":
    dataset_path = r"C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia"
    print("Testing ChestXRayDataset & Transforms...")
    loaders = get_dataloaders(dataset_path, physical_batch_size=16, num_workers=0)
    for split, loader in loaders.items():
        print(f"Split {split}: {len(loader.dataset)} samples across {len(loader)} batches.")
        batch_imgs, batch_lbls, batch_paths = next(iter(loader))
        print(f"  Batch images shape: {batch_imgs.shape}, dtype: {batch_imgs.dtype}")
        print(f"  Batch labels shape: {batch_lbls.shape}")
        print(f"  Min value: {batch_imgs.min():.3f}, Max value: {batch_imgs.max():.3f}")
    
    weights = get_standardized_class_weights()
    print(f"Standardized Class Weights: {weights}")
    print("Dataset module self-test PASSED.")
