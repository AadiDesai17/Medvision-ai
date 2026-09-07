from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Explicit standard class ordering
CLASS_NAMES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def get_transforms(config=None):
    """
    Construct deterministic and augmentation transform pipelines for ViT-B/16.
    
    All images are converted from Grayscale ('L') to 3-channel ('RGB') 
    before tensor transformation to match ImageNet pretrained input expectations.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_size = (224, 224)
    
    rot_deg = 7
    bright = 0.05
    contrast = 0.05
    
    if config:
        img_size = tuple(config.get("dataset", {}).get("image_size", [224, 224]))
        mean = config.get("dataset", {}).get("mean", mean)
        std = config.get("dataset", {}).get("std", std)
        aug_cfg = config.get("augmentation", {})
        rot_deg = aug_cfg.get("rotation_degrees", rot_deg)
        bright = aug_cfg.get("brightness_jitter", bright)
        contrast = aug_cfg.get("contrast_jitter", contrast)

    # Medically sensible, conservative training augmentations
    # Omits vertical/horizontal flips to preserve anatomical laterality and cardiac silhouette
    train_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomRotation(degrees=rot_deg),
        transforms.ColorJitter(brightness=bright, contrast=contrast),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Pure deterministic pipeline for validation, test, and live inference
    val_test_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_test_transform

class ChestXrayDataset(Dataset):
    """
    Dataset wrapper for Chest X-ray classification.
    Reads 512x512 Grayscale PNGs and deterministically converts to RGB for ViT-B/16.
    """
    def __init__(self, split_dir, transform=None):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.samples = []
        self.targets = []
        
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
            
        for class_name in CLASS_NAMES:
            class_folder = self.split_dir / class_name
            if not class_folder.exists():
                raise FileNotFoundError(f"Class folder not found: {class_folder}")
            label = CLASS_TO_IDX[class_name]
            
            # Collect and sort for deterministic loading order
            files = sorted([f for f in class_folder.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
            for f in files:
                self.samples.append((f, label))
                self.targets.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            # Safely and consistently convert Grayscale ('L') to 3-channel RGB
            img = img.convert("RGB")
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_dataloaders(config):
    """
    Build DataLoaders for train, val, and test splits according to config.
    """
    root_dir = Path(config["dataset"]["root_dir"])
    train_transform, eval_transform = get_transforms(config)
    
    train_dataset = ChestXrayDataset(root_dir / "train", transform=train_transform)
    val_dataset = ChestXrayDataset(root_dir / "val", transform=eval_transform)
    test_dataset = ChestXrayDataset(root_dir / "test", transform=eval_transform)
    
    batch_size = config["training"]["batch_size"]
    num_workers = config["dataset"].get("num_workers", 2)
    pin_memory = config["dataset"].get("pin_memory", True) and torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader, CLASS_NAMES, CLASS_TO_IDX
