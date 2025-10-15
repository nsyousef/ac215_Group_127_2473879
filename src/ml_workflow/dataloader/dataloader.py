"""DataLoader with support for both GCS and local file storage - FIXED for multiprocessing"""

import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, List, Callable, Tuple, Dict, Any
import torch 
import time
import fsspec

from constants import DEFAULT_IMAGE_MODE, IMG_COL, LABEL_COL
from utils import (logger, stratified_split)
from dataloader.transform_utils import (get_basic_transform, get_train_transform, get_test_valid_transform, compute_dataset_stats)
    
# top of your dataloader file
import gcsfs
_worker_fs = None

def _get_worker_fs():
    # One FS per worker process (safe with DataLoader(multiprocessing))
    global _worker_fs
    if _worker_fs is None:
        _worker_fs = gcsfs.GCSFileSystem(
            token="cloud",          # uses VM service account if on GCP
            cache_timeout=600,      # reuse bucket metadata
            default_block_size=2**22,   # 1MB blocks; tune if needed
            retry_reads=True
        )
    return _worker_fs

def _worker_init_fn(_):
    global _worker_fs
    _worker_fs = None
    _ = _get_worker_fs()
    try:
        import torch
        torch.set_num_threads(1)  # avoid oversubscribing CPU in workers
    except Exception:
        pass



class ImageDataset(Dataset):
    """PyTorch Dataset for images from GCS or local storage - multiprocessing safe"""
    
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
        # Convert DataFrame to lists for faster, multiprocessing-safe access
        self.image_paths = df[img_col].astype(str).tolist()
        self.labels_raw = df[label_col].astype(str).tolist()
        
        self.img_prefix = img_prefix.rstrip("/")
        self.convert_mode = convert_mode
        self.transform = transform
        self.skip_errors = skip_errors
        
        # Determine if using GCS or local
        self.use_gcs = self.img_prefix.startswith("gs://")
        
        # Don't store filesystem - create per access to avoid pickling issues
        
        if self.use_gcs:
            logger.info("Using GCS storage")
        else:
            logger.info(f"Using local storage: {self.img_prefix or 'relative paths'}")

        # Setup classes
        if classes is None:
            classes = sorted(set(self.labels_raw))
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Pre-compute integer labels for faster access
        self.labels = [self.class_to_idx[label] for label in self.labels_raw]

    def __len__(self) -> int:
        return len(self.image_paths)

    def _get_fs(self):
        if self.use_gcs:
            return _get_worker_fs()
        return None
    
    def __getitem__(self, idx: int):
        """Load and return image and label with error handling"""
        max_retries = 3
        attempts = 0
        last_error = None
        stime = time.time()
        
        while attempts < max_retries:
            path = "unknown"
            try:
                # Try current index first, then nearby indices as fallback
                current_idx = (idx + attempts) % len(self.image_paths)
                
                filename = self.image_paths[current_idx].lstrip("/")
                path = f"{self.img_prefix}/{filename}" if self.img_prefix else filename
                label = self.labels[current_idx]
                
                # Load image from GCS or local
                if self.use_gcs:
                    ptime = time.time()
                    fs = self._get_fs()
                    #print(f"Time taken to get fs: {time.time() - ptime}")
                    qtime = time.time()
                    with fs.open(path, "rb") as f:
                        img = Image.open(f)
                        img = img.convert(self.convert_mode)
                    #print(f"Time taken to open image: {time.time() - qtime}")
                else:
                    with open(path, "rb") as f:
                        img = Image.open(f).convert(self.convert_mode)
                        img.load()
                
                ttime = time.time()
                if self.transform:
                    img = self.transform(img)
                #print(f"Time taken to transform image: {time.time() - ttime}")
                #print(f"Time taken to get image: {time.time() - stime}")
                return img, label
                
            except Exception as e:
                last_error = e
                attempts += 1
                
                # Only print on first attempt to reduce spam (use print, not logger, for multiprocessing)
                if attempts == 1 and self.skip_errors:
                    print(f"Warning: Failed to load image at idx {current_idx} ({path}): {type(e).__name__}")
                
                if not self.skip_errors:
                    raise RuntimeError(f"Error loading image at index {idx} (path: {path}): {type(e).__name__}: {str(e)}") from e
        
        # If all retries failed
        if self.skip_errors:
            # Return a black/white image as fallback
            print(f"Error: All retries failed for index {idx}, returning blank image")
            if self.convert_mode == 'RGB':
                img = Image.new('RGB', (224, 224), (0, 0, 0))
            else:
                img = Image.new('L', (224, 224), 0)
            
            if self.transform:
                img = self.transform(img)
            
            return img, self.labels[idx % len(self.labels)]
        else:
            raise RuntimeError(f"Failed to load image at index {idx} after {max_retries} attempts") from last_error


def create_dataloaders(
    metadata_df: pd.DataFrame,
    img_prefix: str,
    data_config: dict,
    training_config: dict,
    splits_config: dict,
    image_config: dict,
    data_processing_config: dict,
    augmentation_config: dict,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict[str, Any]]:
    """
    Create train, validation, and optional test DataLoaders.
    
    Args:
        metadata_df: DataFrame with image paths and labels
        img_prefix: Prefix for images (GCS or local)
        data_config: Data configuration dictionary
        training_config: Training configuration dictionary
        splits_config: Data splits configuration dictionary
        image_config: Image configuration dictionary
        data_processing_config: Data processing configuration dictionary
        augmentation_config: Augmentation configuration dictionary
    Returns:
        (train_loader, val_loader, test_loader, info_dict)
        val_loader is None if val_size is not provided
    """
    logger.info("Creating DataLoaders")
    
    # Extract values from config dictionaries
    use_local = data_config['use_local']
    local_img_dir = data_config['img_prefix']
    batch_size = training_config['batch_size']
    num_workers = training_config['num_workers']
    prefetch_factor = training_config['prefetch_factor']
    test_size = splits_config['test_size']
    val_size = splits_config['val_size']
    seed = training_config['seed']
    compute_stats = data_processing_config['compute_stats']
    img_size = tuple(image_config['size'])
    weighted_sampling = data_processing_config['weighted_sampling']
    skip_errors = data_processing_config['skip_errors']
    
    # Determine image prefix
    if use_local:
        img_prefix = local_img_dir
        logger.info(f"Using local images from: {img_prefix}")
    elif not img_prefix:
        logger.warning("No img_prefix provided, using paths as-is")
    
    # Enable persistent_workers for better performance (but only if num_workers > 0)
    use_persistent_workers = num_workers > 0
    
    # Split data (train/test or train/val/test)
    split_result = stratified_split(metadata_df, LABEL_COL, test_size, val_size, seed)
    
    if val_size is not None:
        train_df, val_df, test_df = split_result
    else:
        train_df, test_df = split_result
        val_df = None
    
    all_classes = sorted(metadata_df[LABEL_COL].astype(str).unique().tolist())
    num_classes = len(all_classes)
    logger.info(f"Total classes: {num_classes}")
    
    # Compute or validate statistics
    if compute_stats:
        logger.info("Computing dataset statistics from all training data")
        
        temp_dataset = ImageDataset(
            df=train_df, img_col=IMG_COL, label_col=LABEL_COL, img_prefix=img_prefix, 
            transform=get_basic_transform(img_size), classes=all_classes, skip_errors=skip_errors
        )
        temp_loader = DataLoader(
            temp_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        mean, std = compute_dataset_stats(temp_loader)
    else:
        # Use default ImageNet statistics if not computing from data
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    # Create datasets
    train_dataset = ImageDataset(
        df=train_df, img_col=IMG_COL, label_col=LABEL_COL, img_prefix=img_prefix, 
        transform=get_train_transform(mean, std, img_size, augmentation_config), classes=all_classes, skip_errors=skip_errors
    )
    
    test_dataset = ImageDataset(
        df=test_df, img_col=IMG_COL, label_col=LABEL_COL, img_prefix=img_prefix, 
        transform=get_test_valid_transform(mean, std, img_size), classes=all_classes, skip_errors=skip_errors
    )
    
    if val_df is not None:
        val_dataset = ImageDataset(
            df=val_df, img_col=IMG_COL, label_col=LABEL_COL, img_prefix=img_prefix, 
            transform=get_test_valid_transform(mean, std, img_size), classes=all_classes, skip_errors=skip_errors
        )
    else:
        val_dataset = None
    
    # Create dataloaders with optional weighted sampling
    # CRITICAL: Use 'spawn' for GCS to avoid fsspec connection issues
    import multiprocessing
    mp_context = multiprocessing.get_context('spawn') if num_workers > 0 else None
    
    dataloader_kwargs = {
        'batch_size': batch_size, 
        'num_workers': num_workers, 
        'prefetch_factor': prefetch_factor if num_workers > 0 else None, 
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': use_persistent_workers,
        'multiprocessing_context': mp_context,
        'worker_init_fn': _worker_init_fn
    }
    
    if weighted_sampling:
        # Compute class weights for balanced sampling
        class_counts = train_df[LABEL_COL].value_counts()
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_df[LABEL_COL]]
        
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
        'mean': mean, 'std': std,
        'img_size': img_size, 'batch_size': batch_size,
    }
    
    if val_loader is not None:
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    else:
        logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    logger.info("=" * 70)
    
    return train_loader, val_loader, test_loader, info