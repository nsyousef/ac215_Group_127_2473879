"""DataLoader with support for both GCS and local file storage"""

import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, List, Callable, Tuple, Dict, Any

from config import (DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_SIZE, LOCAL_DATA_DIR, DEFAULT_IMAGE_MODE)
from utils import (logger, get_basic_transform, get_train_transform, get_test_valid_transform, compute_dataset_stats, stratified_split)

class ImageDataset(Dataset):
    """PyTorch Dataset for images from GCS or local storage"""
    
    def __init__(self, df: pd.DataFrame, img_col: str, label_col: str, img_prefix: str = "", 
                 transform: Optional[Callable] = None, classes: Optional[List[str]] = None, 
                 convert_mode: str = DEFAULT_IMAGE_MODE, skip_errors: bool = True):
        """
        Args:
            df: DataFrame with image paths and labels
            img_col: Column name for image paths
            label_col: Column name for labels
            img_prefix: Prefix for image paths (GCS bucket or local dir)
            transform: Optional transform to apply to images
            classes: Fixed class list (inferred if None)
            convert_mode: PIL image mode ('RGB' or 'L')
            skip_errors: If True, skip corrupted images and log warning; if False, raise error
        """
        self.df = df
        self.img_col = img_col
        self.label_col = label_col
        self.img_prefix = img_prefix.rstrip("/")
        self.convert_mode = convert_mode
        self.transform = transform
        self.skip_errors = skip_errors
        self.failed_images = []
        
        # Determine if using GCS or local
        self.use_gcs = self.img_prefix.startswith("gs://")
        if self.use_gcs:
            import fsspec
            self.fs = fsspec.filesystem("gs")
            logger.info("Using GCS storage")
        else:
            logger.info(f"Using local storage: {self.img_prefix or 'relative paths'}")

        # Setup classes
        if classes is None:
            classes = sorted(self.df[label_col].astype(str).unique().tolist())
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        """Load and return image and label with error handling"""
        max_retries = 3
        attempts = 0
        
        while attempts < max_retries:
            try:
                # Get current index (may change if we skip)
                current_idx = (idx + attempts) % len(self.df)
                row = self.df.iloc[current_idx]
                filename = str(row[self.img_col]).lstrip("/")
                path = f"{self.img_prefix}/{filename}"
                label = self.class_to_idx[str(row[self.label_col])]
                
                # Load image from GCS or local
                if self.use_gcs:
                    with self.fs.open(path, "rb") as f:
                        img = Image.open(f).convert(self.convert_mode)
                else:
                    with open(path, "rb") as f:
                        img = Image.open(f).convert(self.convert_mode)
                
                # Verify image is valid
                img.verify()
                
                # Reload image after verify
                if self.use_gcs:
                    with self.fs.open(path, "rb") as f:
                        img = Image.open(f).convert(self.convert_mode)
                else:
                    with open(path, "rb") as f:
                        img = Image.open(f).convert(self.convert_mode)
                
                if self.transform:
                    img = self.transform(img)
                
                return img, label
                
            except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError, KeyError) as e:

                error_msg = f"Error loading image at index {current_idx} (path: {path}): {type(e).__name__}: {str(e)}"
                
                if current_idx not in self.failed_images:
                    self.failed_images.append(current_idx)
                    logger.warning(error_msg)
                
                if not self.skip_errors:
                    raise RuntimeError(error_msg) from e
                
                # Try next image
                attempts += 1
                
                if attempts >= max_retries:
                    # Return a random valid image as fallback
                    logger.error(f"Failed to load image after {max_retries} attempts. Using fallback.")
                    fallback_idx = (idx + max_retries) % len(self.df)
                    return self.__getitem__(fallback_idx)
        
        # This should never be reached, but just in case
        raise RuntimeError(f"Unable to load any image starting from index {idx}")


def create_dataloaders(
    metadata_df: pd.DataFrame,
    img_col: str = "filename",
    label_col: str = "label",
    img_prefix: str = "",
    use_local: bool = False,
    local_img_dir: str = LOCAL_DATA_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 4,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    seed: int = 42,
    compute_stats: bool = True,
    img_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    weighted_sampling: bool = True,
    skip_errors: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict[str, Any]]:
    """
    Create train, validation, and optional test DataLoaders.
    
    Args:
        metadata_df: DataFrame with image paths and labels
        img_col: Column for image filenames
        label_col: Column for labels
        img_prefix: Prefix for images (GCS or local)
        use_local: If True, use local files from local_img_dir
        local_img_dir: Local directory for images
        batch_size: Batch size
        num_workers: Number of workers
        test_size: Train/test split size
        val_size: If None, creates train/test split; if provided, creates train/val/test split
        seed: Random seed
        compute_stats: Compute mean/std from data
        img_size: Image size (height, width)
        mean: Optional custom mean (required if compute_stats=False)
        std: Optional custom std (required if compute_stats=False)
        weighted_sampling: Use weighted sampling for imbalanced data
        skip_errors: Skip corrupted images instead of raising errors

    Returns:
        (train_loader, val_loader, test_loader, info_dict)
        val_loader is None if val_size is not provided
    """
    logger.info("Creating DataLoaders")
    
    # Determine image prefix
    if use_local:
        img_prefix = local_img_dir
        logger.info(f"Using local images from: {img_prefix}")
    elif not img_prefix:
        logger.warning("No img_prefix provided, using paths as-is")
    
    # Split data (train/test or train/val/test)
    split_result = stratified_split(metadata_df, label_col, test_size, val_size, seed)
    
    if val_size is not None:
        train_df, val_df, test_df = split_result
    else:
        train_df, test_df = split_result
        val_df = None
    
    all_classes = sorted(metadata_df[label_col].astype(str).unique().tolist())
    num_classes = len(all_classes)
    logger.info(f"Total classes: {num_classes}")
    
    # Compute or validate statistics
    if compute_stats:
        logger.info("Computing dataset statistics from all training data")
        
        temp_dataset = ImageDataset(
            df=train_df, img_col=img_col, label_col=label_col, img_prefix=img_prefix, 
            transform=get_basic_transform(img_size), classes=all_classes, skip_errors=skip_errors
        )
        temp_loader = DataLoader(
            temp_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        mean, std = compute_dataset_stats(temp_loader)
    else:
        if mean is None or std is None:
            raise ValueError("When compute_stats=False, you must provide mean and std parameters")
        logger.info(f"Using provided stats - Mean: {mean}, Std: {std}")
    
    # Create datasets
    train_dataset = ImageDataset(
        df=train_df, img_col=img_col, label_col=label_col, img_prefix=img_prefix, 
        transform=get_train_transform(mean, std, img_size), classes=all_classes, skip_errors=skip_errors
    )
    
    test_dataset = ImageDataset(
        df=test_df, img_col=img_col, label_col=label_col, img_prefix=img_prefix, 
        transform=get_test_valid_transform(mean, std, img_size), classes=all_classes, skip_errors=skip_errors
    )
    
    if val_df is not None:
        val_dataset = ImageDataset(
            df=val_df, img_col=img_col, label_col=label_col, img_prefix=img_prefix, 
            transform=get_test_valid_transform(mean, std, img_size), classes=all_classes, skip_errors=skip_errors
        )
    else:
        val_dataset = None
    
    # Create dataloaders with optional weighted sampling
    dataloader_kwargs = {'batch_size': batch_size, 'num_workers': num_workers,'pin_memory': True}
    
    if weighted_sampling:
        # Compute class weights for balanced sampling
        class_counts = train_df[label_col].value_counts()
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_df[label_col]]
        
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, sampler=sampler, drop_last=True, **dataloader_kwargs)
        logger.info("Using weighted sampling for imbalanced data")
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **dataloader_kwargs)
    
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)
    
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    else:
        val_loader = None
    
    info = {
        'num_classes': num_classes, 'classes': all_classes,
        'class_to_idx': train_dataset.class_to_idx,
        'train_size': len(train_dataset), 
        'test_size': len(test_dataset),
        'val_size': len(val_dataset) if val_dataset is not None else 0,
        'mean': mean, 'std': std, 'img_size': img_size, 'batch_size': batch_size,
        'failed_images': {
            'train': train_dataset.failed_images,
            'test': test_dataset.failed_images,
            'val': val_dataset.failed_images if val_dataset is not None else []
        }
    }
    
    if val_loader is not None:
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    else:
        logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    logger.info("=" * 70)
    
    return train_loader, val_loader, test_loader, info
