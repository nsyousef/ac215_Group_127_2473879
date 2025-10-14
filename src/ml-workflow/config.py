"""Configuration settings for the dataloader pipeline"""

import os

# GCS Configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "derma-datasets-2")
GCS_METADATA_PATH = os.getenv("GCS_METADATA_PATH", "final/metadata_all_harmonized.csv")
GCS_IMAGE_PREFIX = os.getenv("GCS_IMAGE_PREFIX", f"gs://{GCS_BUCKET_NAME}/final/imgs")

# Local Storage Configuration
LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "../../data")
LOCAL_METADATA_PATH = os.getenv("LOCAL_METADATA_PATH", f"{LOCAL_DATA_DIR}/metadata_all_harmonized.csv")

# Data Filtering
MIN_IMAGES_PER_LABEL = int(os.getenv("MIN_IMAGES_PER_LABEL", "10"))

# Training Configuration
DEFAULT_SEED = int(os.getenv("DEFAULT_SEED", "42"))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "32"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

# Image Configuration
DEFAULT_IMAGE_SIZE = tuple(map(int, os.getenv("DEFAULT_IMAGE_SIZE", "224,224").split(",")))
DEFAULT_IMAGE_MODE = os.getenv("DEFAULT_IMAGE_MODE", "RGB")

# Augmentation Parameters
BRIGHTNESS_JITTER = float(os.getenv("BRIGHTNESS_JITTER", "0.1"))
CONTRAST_JITTER = float(os.getenv("CONTRAST_JITTER", "0.1"))
SATURATION_JITTER = float(os.getenv("SATURATION_JITTER", "0.1"))