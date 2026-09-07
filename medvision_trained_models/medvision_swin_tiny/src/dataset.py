"""
Dataset loader and preprocessing pipeline for Chest X-ray classification.
Supports 3 classes: NORMAL (0), PNEUMONIA (1), TUBERCULOSIS (2).
Ensures fair comparison with ViT-B/16 by maintaining identical transforms and class weights.
"""
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

CLASS_TO_IDX: Dict[str, int] = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
    "TUBERCULOSIS": 2,
}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}
CLASSES: List[str] = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str, img_size: int = 224) -> transforms.Compose:
    """
    Returns image transformations for train, val, or test splits.
    Strictly adheres to fair comparison: no horizontal/vertical flips.
    
    Args:
        split: 'train', 'val', or 'test'.
        img_size: Target square image dimension (default: 224).
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class ChestXRayDataset(Dataset):
    """
    Custom PyTorch Dataset for Chest X-ray images.
    Converts 1-channel grayscale to 3-channel RGB before transforms.
    """

    def __init__(self, split_dir: Path, transform: Optional[transforms.Compose] = None):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_counts: Dict[str, int] = {cls: 0 for cls in CLASSES}

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        for cls_name, cls_idx in CLASS_TO_IDX.items():
            cls_folder = self.split_dir / cls_name
            if not cls_folder.exists():
                raise FileNotFoundError(f"Class folder not found: {cls_folder}")

            for file_path in cls_folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == ".png":
                    self.samples.append((file_path, cls_idx))
                    self.class_counts[cls_name] += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]
        with Image.open(file_path) as img:
            # Source images are grayscale ('L'); convert to 'RGB' for 3-channel backbone
            img_rgb = img.convert("RGB")

        if self.transform:
            tensor = self.transform(img_rgb)
        else:
            tensor = transforms.ToTensor()(img_rgb)

        return tensor, label


def compute_class_weights(train_dataset: ChestXRayDataset) -> torch.Tensor:
    """
    Computes moderated square-root inverse-frequency weights from training counts:
    w_c = (1 / sqrt(N_c)) / (mean_k(1 / sqrt(N_k)))
    """
    counts = [train_dataset.class_counts[cls] for cls in CLASSES]
    for c, cnt in zip(CLASSES, counts):
        if cnt <= 0:
            raise ValueError(f"Class {c} has invalid count {cnt}")

    inv_sqrts = [1.0 / math.sqrt(n) for n in counts]
    mean_inv_sqrt = sum(inv_sqrts) / len(counts)
    weights = [inv / mean_inv_sqrt for inv in inv_sqrts]
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    dataset_root: Path,
    batch_size: int = 16,
    num_workers: int = 2,
    pin_memory: bool = True,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Initializes train, validation, and test DataLoaders along with calculated class weights.
    """
    dataset_root = Path(dataset_root)
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    train_ds = ChestXRayDataset(train_dir, transform=get_transforms("train", img_size))
    val_ds = ChestXRayDataset(val_dir, transform=get_transforms("val", img_size))
    test_ds = ChestXRayDataset(test_dir, transform=get_transforms("test", img_size))

    class_weights = compute_class_weights(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    dataset_path = Path(r"C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia")
    print(f"Testing ChestXRayDataset from {dataset_path}...")
    train_ds = ChestXRayDataset(dataset_path / "train", transform=get_transforms("train"))
    print(f"Train samples: {len(train_ds)}")
    print(f"Class counts: {train_ds.class_counts}")
    weights = compute_class_weights(train_ds)
    print(f"Computed class weights: {weights.tolist()}")
    sample_tensor, sample_label = train_ds[0]
    print(f"Sample tensor shape: {sample_tensor.shape}, label: {sample_label} ({IDX_TO_CLASS[sample_label]})")
    print("[OK] Dataset module verified.")
