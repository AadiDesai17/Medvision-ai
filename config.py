"""
MedVision Disease Prediction - Central Configuration
Author: K. Gnana Saaketh (ROLE: Deep Learning / Disease Prediction)
Project: Medvision — AI Medical Imaging Assistant
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
DEFAULT_PREDICTION_OUTPUT = PREDICTIONS_DIR / "model_output.json"

# Local NIH Dataset Path in user's downloads
LOCAL_NIH_DATASET_DIR = Path(r"C:\Users\Saaketh\Downloads\Medical_Datasets\nih_chest_xray14")
LOCAL_NIH_IMAGES_DIR = LOCAL_NIH_DATASET_DIR / "images"
LOCAL_NIH_CSV_PATH = LOCAL_NIH_DATASET_DIR / "Data_Entry_2017_v2020.csv"

# Model Architecture & Identity
MODEL_NAME = "MedVision-ChestXRay-Classifier"
ARCHITECTURE = "ResNet50"
MODEL_VERSION = "v1.0"
MODULE_NAME = "chest_xray"

# Classes
CLASS_NAMES = ["Normal", "Pneumonia"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASS_NAMES)}

# Hyperparameters
RANDOM_SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
NUM_WORKERS = 0  # 0 is recommended on Windows to prevent IPC issues

# ImageNet Normalization Constants
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Device Detection
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
