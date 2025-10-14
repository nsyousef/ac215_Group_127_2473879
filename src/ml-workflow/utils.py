"""Utility functions for dataloader pipeline"""

import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Tuple, List

from config import (
    MIN_IMAGES_PER_LABEL, DEFAULT_SEED, DEFAULT_IMAGE_SIZE,
    BRIGHTNESS_JITTER, CONTRAST_JITTER, SATURATION_JITTER
)

# Setup logging
def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with consistent format"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logger()

# Transform functions
def get_basic_transform(img_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """Basic transform for computing statistics"""
    return transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

def get_train_transform(mean: List[float], std: List[float], img_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
                       use_advanced_augmentation: bool = False) -> transforms.Compose:
    """
    Training transform with augmentation
    
    Args:
        mean: Channel-wise mean for normalization
        std: Channel-wise std for normalization
        img_size: Target image size
        use_advanced_augmentation: If True, use additional augmentations (rotation, random crops, etc.)
    """
    if use_advanced_augmentation:
        transform_list = [
            transforms.Resize(int(img_size[0] * 1.1)),  # Slightly larger for random crop
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=BRIGHTNESS_JITTER, contrast=CONTRAST_JITTER, 
                                 saturation=SATURATION_JITTER, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    else:
        transform_list = [
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=BRIGHTNESS_JITTER, contrast=CONTRAST_JITTER, saturation=SATURATION_JITTER),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    
    return transforms.Compose(transform_list)

def get_test_valid_transform(mean: List[float], std: List[float], img_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """Validation transform without augmentation"""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def load_metadata(source: str, min_samples: int = MIN_IMAGES_PER_LABEL) -> pd.DataFrame:
    """
    Load metadata from GCS or local file and filter by minimum samples
    
    Args:
        source: Path to CSV (gs://bucket/path or local/path.csv)
        min_samples: Minimum images per label
    """
    logger.info(f"Loading metadata from: {source}")
    metadata = pd.read_csv(source)
    logger.info(f"Total images loaded: {len(metadata):,}")
    
    # Filter labels with insufficient samples
    label_counts = metadata["label"].value_counts()
    preserve_labels = label_counts[label_counts >= min_samples].index
    logger.info(f"Labels with >= {min_samples} images: {len(preserve_labels):,}")
    
    metadata = metadata[metadata['label'].isin(preserve_labels)].reset_index(drop=True)
    logger.info(f"Images after filtering: {len(metadata):,}")
    
    return metadata

def compute_dataset_stats(dataloader: DataLoader, num_channels: int = 3) -> Tuple[List[float], List[float]]:
    """Compute channel-wise mean and std"""
    logger.info("Computing dataset statistics...")
    mean = torch.zeros(num_channels)
    std = torch.zeros(num_channels)
    total_images = 0

    for imgs, _ in tqdm(dataloader, desc="Computing stats"):
        batch_samples = imgs.size(0)
        imgs = imgs.view(batch_samples, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    mean_list, std_list = mean.tolist(), std.tolist()
    
    logger.info(f"Mean: {mean_list}")
    logger.info(f"Std: {std_list}")
    
    return mean_list, std_list

def stratified_split(df: pd.DataFrame, label_col: str = "label", test_size: float = 0.2, val_size: float = None, seed: int = DEFAULT_SEED) -> Tuple[pd.DataFrame, ...]:
    """Split data into train/test or train/val/test sets with stratification"""
    if val_size is None:
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=seed)
        logger.info(f"Train samples: {len(train_df):,}, Test samples: {len(test_df):,}")
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    else:
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=seed)
        adjusted_val_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(train_df, test_size=adjusted_val_size,  stratify=train_df[label_col], random_state=seed)
        logger.info(f"Train samples: {len(train_df):,}, Val samples: {len(val_df):,}, Test samples: {len(test_df):,}")
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def analyze_class_distribution(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Analyze class distribution in dataset"""
    distribution = df[label_col].value_counts().reset_index()
    distribution.columns = ['class', 'count']
    distribution['percentage'] = (distribution['count'] / len(df) * 100).round(2)
    distribution = distribution.sort_values('count', ascending=False)
    
    logger.info(f"\nClass Distribution:")
    logger.info(f"Total classes: {len(distribution)}")
    logger.info(f"Total samples: {len(df):,}")
    logger.info(f"Min samples per class: {distribution['count'].min()}")
    logger.info(f"Max samples per class: {distribution['count'].max()}")
    logger.info(f"Mean samples per class: {distribution['count'].mean():.1f}")
    logger.info(f"Median samples per class: {distribution['count'].median():.1f}")
    
    return distribution