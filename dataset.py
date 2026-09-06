"""
Medical Dataset Preparation, Splitting & PyTorch Dataset
Author: K. Gnana Saaketh (ROLE: Deep Learning / Disease Prediction)
Project: Medvision — AI Medical Imaging Assistant
"""

import os
import shutil
import csv
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    LOCAL_NIH_IMAGES_DIR,
    LOCAL_NIH_CSV_PATH,
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    CLASS_NAMES,
    CLASS_TO_IDX,
    RANDOM_SEED,
    BATCH_SIZE,
    NUM_WORKERS
)
from src.preprocessing import get_train_transforms, get_val_test_transforms, load_image_rgb

class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for Chest X-Ray Normal vs. Pneumonia Classification.
    Returns:
        image (torch.Tensor): Preprocessed image tensor [3, 224, 224]
        label (int): 0 for Normal, 1 for Pneumonia
        image_id (str): Unique image filename/identifier
    """
    def __init__(self, df_or_csv_path, transform=None):
        if isinstance(df_or_csv_path, (str, Path)):
            self.df = pd.read_csv(df_or_csv_path)
        elif isinstance(df_or_csv_path, pd.DataFrame):
            self.df = df_or_csv_path.copy().reset_index(drop=True)
        else:
            raise TypeError("Expected pandas DataFrame or path to split CSV.")
            
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = int(row["label"])
        image_id = str(row["image_id"])

        img = load_image_rgb(image_path)
        if self.transform:
            img = self.transform(img)

        return img, label, image_id


def prepare_dataset(
    nih_images_dir=LOCAL_NIH_IMAGES_DIR,
    nih_csv_path=LOCAL_NIH_CSV_PATH,
    splits_dir=SPLITS_DIR,
    random_seed=RANDOM_SEED,
    max_normal_samples=250,
    force_rebuild=False
):
    """
    Reads local NIH ChestX-ray14 dataset, identifies legitimate Normal and Pneumonia cases,
    creates stratified, leakage-free Train/Val/Test splits, and saves manifest CSVs.
    
    Split Ratios:
        - Train: 70%
        - Val:   15%
        - Test:  15%
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv = splits_dir / "train_split.csv"
    val_csv = splits_dir / "val_split.csv"
    test_csv = splits_dir / "test_split.csv"

    if not force_rebuild and train_csv.exists() and val_csv.exists() and test_csv.exists():
        print(f"[Dataset] Existing split manifests found in {splits_dir}. Loading...")
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        return train_df, val_df, test_df

    nih_images_dir = Path(nih_images_dir)
    nih_csv_path = Path(nih_csv_path)

    if not nih_images_dir.exists() or not nih_csv_path.exists():
        raise FileNotFoundError(
            f"Local NIH ChestX-ray14 dataset not found at:\n"
            f"Images: {nih_images_dir}\nCSV: {nih_csv_path}\n"
            f"Please verify your Medical_Datasets folder."
        )

    print(f"[Dataset] Scanning local NIH ChestX-ray14 dataset...")
    existing_images = set(os.listdir(nih_images_dir))
    
    pneumonia_records = []
    normal_records = []

    with open(nih_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row["Image Index"]
            if img_name not in existing_images:
                continue

            findings = [x.strip() for x in row["Finding Labels"].split("|")]
            patient_id = row.get("Patient ID", "unknown")
            full_path = str(nih_images_dir / img_name)

            if "Pneumonia" in findings:
                pneumonia_records.append({
                    "image_id": img_name,
                    "image_path": full_path,
                    "patient_id": patient_id,
                    "class_name": "Pneumonia",
                    "label": CLASS_TO_IDX["Pneumonia"]
                })
            elif "No Finding" in findings:
                normal_records.append({
                    "image_id": img_name,
                    "image_path": full_path,
                    "patient_id": patient_id,
                    "class_name": "Normal",
                    "label": CLASS_TO_IDX["Normal"]
                })

    num_pneumonia = len(pneumonia_records)
    print(f"[Dataset] Found {num_pneumonia} Pneumonia images and {len(normal_records)} Normal images.")

    # Balance Normal cohort relative to Pneumonia cohort to prevent heavy bias
    np.random.seed(random_seed)
    target_normal_count = min(len(normal_records), max(num_pneumonia * 3, max_normal_samples))
    selected_indices = np.random.choice(len(normal_records), size=target_normal_count, replace=False)
    selected_normal_records = [normal_records[i] for i in selected_indices]

    # Combine into single dataframe
    all_records = pneumonia_records + selected_normal_records
    df = pd.DataFrame(all_records)

    # Patient-level leakage prevention: Group by patient_id
    patient_ids = df["patient_id"].unique()
    np.random.shuffle(patient_ids)

    # 70% Train, 15% Val, 15% Test
    n_total = len(patient_ids)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    train_patients = set(patient_ids[:n_train])
    val_patients = set(patient_ids[n_train:n_train + n_val])
    test_patients = set(patient_ids[n_train + n_val:])

    train_df = df[df["patient_id"].isin(train_patients)].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_patients)].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_patients)].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Ensure all splits have at least one sample of each class (if not, rebalance from train)
    for split_df, name in [(val_df, "Val"), (test_df, "Test")]:
        for cls_name, cls_idx in CLASS_TO_IDX.items():
            if (split_df["label"] == cls_idx).sum() == 0:
                donor_row = train_df[train_df["label"] == cls_idx].iloc[-1:]
                train_df = train_df.drop(donor_row.index).reset_index(drop=True)
                if name == "Val":
                    val_df = pd.concat([val_df, donor_row], ignore_index=True)
                else:
                    test_df = pd.concat([test_df, donor_row], ignore_index=True)

    # Save split manifests
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print("=" * 55)
    print("           MEDVISION DATASET SUMMARY REPORT           ")
    print("=" * 55)
    print(f"Total Cohort Images : {len(df)}")
    print(f"  - Training Split  : {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - Validation Split: {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  - Testing Split   : {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")
    print("-" * 55)
    print("Class Distribution Across Splits:")
    for split_name, s_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_norm = (s_df["label"] == CLASS_TO_IDX["Normal"]).sum()
        n_pneu = (s_df["label"] == CLASS_TO_IDX["Pneumonia"]).sum()
        print(f"  [{split_name:5s}] Normal: {n_norm:3d} | Pneumonia: {n_pneu:3d} (Total: {len(s_df):3d})")
    print("=" * 55)
    print(f"Split manifests saved to: {splits_dir}")

    return train_df, val_df, test_df


def get_data_loaders(train_df, val_df, test_df, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Creates PyTorch DataLoader instances for train, val, and test splits.
    """
    train_dataset = ChestXRayDataset(train_df, transform=get_train_transforms())
    val_dataset = ChestXRayDataset(val_df, transform=get_val_test_transforms())
    test_dataset = ChestXRayDataset(test_df, transform=get_val_test_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


def compute_class_weights(train_df, device):
    """
    Calculates balanced class weights for CrossEntropyLoss to counteract class imbalance.
    weight[c] = total_samples / (num_classes * count[c])
    """
    counts = np.array([
        (train_df["label"] == 0).sum(),
        (train_df["label"] == 1).sum()
    ], dtype=np.float32)
    
    total = np.sum(counts)
    weights = total / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


if __name__ == "__main__":
    prepare_dataset(force_rebuild=True)
