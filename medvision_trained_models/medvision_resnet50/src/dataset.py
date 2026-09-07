"""
Dataset loader and standardized preprocessing pipeline for Chest X-ray classification.
Supports 3 classes:
  0: NORMAL
  1: PNEUMONIA
  2: TUBERCULOSIS

Standardized Preprocessing:
- SafeRGB: Converts 1-channel grayscale ('L') to 3-channel RGB via channel replication.
- Spatial Resolution: 224x224
- Training Augmentation:
    * Resize(224, 224)
    * RandomRotation(degrees=7)
    * ColorJitter(brightness=0.05, contrast=0.05)
    * ToTensor()
    * ImageNet Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    * STRICTLY NO RandomHorizontalFlip, NO RandomVerticalFlip, NO RandomResizedCrop.
- Validation/Test/Inference:
    * Resize(224, 224)
    * ToTensor()
    * ImageNet Normalization
- Standardized Class Weights:
    * NORMAL:       0.8630
    * PNEUMONIA:    0.9902
    * TUBERCULOSIS: 1.1468
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

# Standardized Class Weights from moderation formula:
# w_c = (1 / sqrt(N_c)) / (mean_k(1 / sqrt(N_k)))
STANDARDIZED_CLASS_WEIGHTS = {
    "NORMAL": 0.8630,
    "PNEUMONIA": 0.9902,
    "TUBERCULOSIS": 1.1468,
}


class SafeRGB:
    """Ensures input PIL image is strictly 3-channel RGB via channel replication."""
    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            return img.convert("RGB")
        return img


def get_transforms(
    split: str,
    img_size: int = 224,
) -> transforms.Compose:
    """
    Returns image transformations strictly adhering to the standardized protocol.
    
    Args:
        split: 'train', 'val', 'test', or 'infer'
        img_size: Target square image dimension (default: 224)
    """
    if split in ["val", "test", "infer"]:
        return transforms.Compose([
            SafeRGB(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    elif split == "train":
        # Standardized train augmentations aligned with ViT-B/16 and Swin-Tiny
        return transforms.Compose([
            SafeRGB(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', 'test', or 'infer'.")


class ChestXRayDataset(Dataset):
    """
    Custom PyTorch Dataset for 3-Class Chest X-ray images.
    Reads images directly from class subdirectories: NORMAL, PNEUMONIA, TUBERCULOSIS.
    Converts 1-channel grayscale ('L') to 3-channel ('RGB') before transforms.
    """
    def __init__(
        self,
        split_dir: Path,
        transform: Optional[transforms.Compose] = None
    ):
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

            for file_path in sorted(cls_folder.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() == ".png":
                    self.samples.append((file_path, cls_idx))
                    self.class_counts[cls_name] += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]
        with Image.open(file_path) as raw_img:
            img = raw_img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label


def compute_class_weights(train_dataset: Optional[ChestXRayDataset] = None) -> torch.Tensor:
    """
    Returns standardized square-root inverse-frequency weights:
      NORMAL:       0.8630
      PNEUMONIA:    0.9902
      TUBERCULOSIS: 1.1468
    """
    if train_dataset is not None:
        counts = [train_dataset.class_counts[cls] for cls in CLASSES]
        for c, cnt in zip(CLASSES, counts):
            if cnt <= 0:
                raise ValueError(f"Class {c} has invalid count {cnt}")
        inv_sqrts = [1.0 / math.sqrt(n) for n in counts]
        mean_inv_sqrt = sum(inv_sqrts) / len(counts)
        weights = [round(inv / mean_inv_sqrt, 4) for inv in inv_sqrts]
        return torch.tensor(weights, dtype=torch.float32)

    return torch.tensor([
        STANDARDIZED_CLASS_WEIGHTS["NORMAL"],
        STANDARDIZED_CLASS_WEIGHTS["PNEUMONIA"],
        STANDARDIZED_CLASS_WEIGHTS["TUBERCULOSIS"],
    ], dtype=torch.float32)


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
    dataset_path = Path(r"C:/Users/AADI/Downloads/dataset/chest-xray-tb-pneumonia")
    if dataset_path.exists():
        print("Testing standardized dataset loading...")
        train_ds = ChestXRayDataset(dataset_path / "train", transform=get_transforms("train", 224))
        print(f"Loaded {len(train_ds)} train samples.")
        print(f"Class counts: {train_ds.class_counts}")
        weights = compute_class_weights(train_ds)
        print(f"Class weights: {weights.tolist()}")
        for cls, w in zip(CLASSES, weights.tolist()):
            print(f"  {cls:<14}: {w:.4f}")
        print("[OK] Standardized dataset tests passed successfully.")
    else:
        print(f"Dataset path does not exist: {dataset_path}")
