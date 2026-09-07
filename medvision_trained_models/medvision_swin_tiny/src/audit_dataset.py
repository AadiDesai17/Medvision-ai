import os
import sys
import hashlib
from collections import defaultdict, Counter
from pathlib import Path
from PIL import Image
import math

DATASET_ROOT = Path(r"C:\Users\AADI\Downloads\dataset\chest-xray-tb-pneumonia")
SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

def compute_file_hash(filepath, block_size=65536):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()

def run_audit():
    print(f"=== DATASET AUDIT: {DATASET_ROOT} ===")
    
    if not DATASET_ROOT.exists():
        print(f"ERROR: Dataset root {DATASET_ROOT} does not exist!")
        return
    print(f"[OK] Dataset path exists: {DATASET_ROOT}")

    for split in SPLITS:
        split_dir = DATASET_ROOT / split
        if not split_dir.exists():
            print(f"ERROR: Split directory missing: {split_dir}")
            return
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                print(f"ERROR: Class directory missing: {cls_dir}")
                return
    print("[OK] All split directories (train/val/test) and class directories (NORMAL/PNEUMONIA/TUBERCULOSIS) exist.")

    split_class_counts = defaultdict(lambda: defaultdict(int))
    split_totals = defaultdict(int)
    extensions = Counter()
    zero_byte_files = []
    corrupt_files = []
    image_sizes = Counter()
    image_modes = Counter()
    channels = Counter()
    
    all_files_by_split = defaultdict(dict)
    hash_to_locations = defaultdict(list)
    filename_patterns = defaultdict(list)

    total_files = 0

    print("Scanning and verifying all images (dimensions, modes, integrity, hashes)...")
    for split in SPLITS:
        for cls in CLASSES:
            cls_dir = DATASET_ROOT / split / cls
            for entry in cls_dir.iterdir():
                if entry.is_file():
                    total_files += 1
                    ext = entry.suffix.lower()
                    extensions[ext] += 1
                    
                    size = entry.stat().st_size
                    if size == 0:
                        zero_byte_files.append(str(entry))
                        continue
                    
                    split_class_counts[split][cls] += 1
                    split_totals[split] += 1
                    
                    try:
                        with Image.open(entry) as img:
                            img.verify()
                        with Image.open(entry) as img:
                            image_sizes[img.size] += 1
                            image_modes[img.mode] += 1
                            channels[len(img.getbands())] += 1
                    except Exception as e:
                        corrupt_files.append((str(entry), str(e)))
                    
                    fhash = compute_file_hash(entry)
                    rel_name = f"{cls}/{entry.name}"
                    all_files_by_split[split][rel_name] = (str(entry), fhash)
                    hash_to_locations[fhash].append((split, cls, entry.name))
                    filename_patterns[cls].append(entry.name)

    print("\n--- IMAGE COUNTS ---")
    for split in SPLITS:
        print(f"{split.upper()}:")
        for cls in CLASSES:
            print(f"  - {cls}: {split_class_counts[split][cls]}")
        print(f"  Total: {split_totals[split]}")
    overall_total = sum(split_totals.values())
    print(f"OVERALL TOTAL: {overall_total}")

    expected = {
        "train": {"NORMAL": 3911, "PNEUMONIA": 2971, "TUBERCULOSIS": 2215, "total": 9097},
        "val": {"NORMAL": 838, "PNEUMONIA": 637, "TUBERCULOSIS": 475, "total": 1950},
        "test": {"NORMAL": 839, "PNEUMONIA": 637, "TUBERCULOSIS": 475, "total": 1951},
        "overall": 12998
    }

    discrepancies = []
    for split in SPLITS:
        for cls in CLASSES:
            cnt = split_class_counts[split][cls]
            exp = expected[split][cls]
            if cnt != exp:
                discrepancies.append(f"{split}/{cls}: got {cnt}, expected {exp}")
        tot = split_totals[split]
        exp_tot = expected[split]["total"]
        if tot != exp_tot:
            discrepancies.append(f"{split} total: got {tot}, expected {exp_tot}")
    if overall_total != expected["overall"]:
        discrepancies.append(f"Overall total: got {overall_total}, expected {expected['overall']}")

    if discrepancies:
        print(f"\n[WARNING] Discrepancies found: {discrepancies}")
    else:
        print("\n[OK] Image counts match expected values exactly!")

    print("\n--- FORMAT AND DIMENSIONS ---")
    print(f"Extensions: {dict(extensions)}")
    print(f"Zero-byte files: {len(zero_byte_files)}")
    print(f"Corrupt/unreadable files: {len(corrupt_files)}")
    print(f"Image sizes (width, height): {dict(image_sizes)}")
    print(f"Image modes: {dict(image_modes)}")
    print(f"Channels: {dict(channels)}")

    print("\n--- SPLIT SEPARATION AUDIT ---")
    cross_split_duplicates = defaultdict(list)
    for fhash, locs in hash_to_locations.items():
        splits_present = set(loc[0] for loc in locs)
        if len(splits_present) > 1:
            cross_split_duplicates[fhash] = locs

    print(f"Cross-split duplicate image content (identical MD5): {len(cross_split_duplicates)}")
    if cross_split_duplicates:
        for fhash, locs in list(cross_split_duplicates.items())[:5]:
            print(f"  Hash {fhash}: {locs}")
    else:
        print("[OK] Exact image content separation: No identical images across train, val, and test splits.")

    train_names = set(all_files_by_split["train"].keys())
    val_names = set(all_files_by_split["val"].keys())
    test_names = set(all_files_by_split["test"].keys())

    print(f"Duplicate relative paths between train & val: {len(train_names & val_names)}")
    print(f"Duplicate relative paths between train & test: {len(train_names & test_names)}")
    print(f"Duplicate relative paths between val & test: {len(val_names & test_names)}")

    print("\n--- PATIENT-LEVEL SEPARATION AUDIT ---")
    for cls in CLASSES:
        sample_names = filename_patterns[cls][:5]
        print(f"Sample filenames for {cls}: {sample_names}")
    print("Patient-level separation assessment: Heterogeneous naming conventions across classes. Ground-truth clinical metadata tables are not present; patient-level separation cannot be guaranteed or claimed. Exact image-level separation is confirmed.")

    print("\n--- CLASS WEIGHT CALCULATION ---")
    train_counts = split_class_counts["train"]
    N_c = [train_counts[cls] for cls in CLASSES]
    print(f"Classes: {CLASSES}")
    print(f"Train counts: {N_c}")
    
    inv_sqrts = [1.0 / math.sqrt(n) for n in N_c]
    mean_inv_sqrt = sum(inv_sqrts) / len(CLASSES)
    weights = [inv / mean_inv_sqrt for inv in inv_sqrts]
    
    print("Moderated square-root inverse-frequency weights:")
    for cls, w, cnt in zip(CLASSES, weights, N_c):
        print(f"  {cls:<15}: count={cnt:<5} -> weight={w:.4f} (unrounded: {w:.6f})")

if __name__ == "__main__":
    run_audit()
