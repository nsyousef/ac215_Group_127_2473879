import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, List, Callable, Tuple, Dict, Any
import torch 
import time
import fsspec
import multiprocessing

from constants import DEFAULT_IMAGE_MODE, IMG_COL, LABEL_COL, MAX_RETRIES
from utils import (logger, stratified_split)
from dataloader.transform_utils import (get_basic_transform, get_train_transform, get_test_valid_transform, compute_dataset_stats)
    

# If we use multiple workers with GCP, we need to ensure each worker has its own filesystem object
import gcsfs
_worker_fs = None

def _get_worker_fs():
    global _worker_fs
    if _worker_fs is None:
        _worker_fs = gcsfs.GCSFileSystem(
            token="cloud",          
            cache_timeout=600,      
            default_block_size=2**22,   # How much memory it loads in one packet
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
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 img_col: str, 
                 label_col: str, 
                 img_prefix: str = "", 
                 transform: Optional[Callable] = None, 
                 classes: Optional[List[str]] = None, 
                 use_local: bool = False,
                 ):        
        """
        Args:
            df: DataFrame with image paths and labels
            img_col: Column name for image paths
            label_col: Column name for labels
            img_prefix: Prefix for image paths (GCS bucket or local dir)
            transform: Optional transform to apply to images
            classes: Fixed class list (inferred if None)
        """
        # Convert DataFrame to lists for faster, multiprocessing-safe access
        self.image_paths = df[img_col].astype(str).tolist()
        self.labels_raw = df[label_col].astype(str).tolist()
        self.max_retries = MAX_RETRIES
        
        self.img_prefix = img_prefix.rstrip("/")
        self.convert_mode = DEFAULT_IMAGE_MODE
        self.transform = transform
        self.use_local = use_local

        # Setup classes
        if classes is None:
            classes = sorted(set(self.labels_raw))
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = [self.class_to_idx[label] for label in self.labels_raw]

    def __len__(self) -> int:
        return len(self.image_paths)

    def _get_fs(self):
        if not self.use_local:
            return _get_worker_fs()
        return None
    
    def __getitem__(self, idx: int):
        """Load and return image and label"""
        attempts = 0
        last_error = None

        while attempts < self.max_retries:
            try:
                current_idx = (idx + attempts) % len(self.image_paths)
                filename = self.image_paths[current_idx].lstrip("/")
                path = f"{self.img_prefix}/{filename}" if self.img_prefix else filename
                label = self.labels[current_idx]
                
                # Load image from GCS or local
                if not self.use_local:
                    fs = self._get_fs()
                    with fs.open(path, "rb") as f:
                        img = Image.open(f)
                        img = img.convert(self.convert_mode)
                else:
                    with open(path, "rb") as f:
                        img = Image.open(f).convert(self.convert_mode)
                        img.load()
                
                if self.transform:
                    img = self.transform(img)
                return img, label
                
            except Exception as e:
                last_error = e
                attempts += 1
                # Only print on first attempt to reduce spam (use print, not logger, for multiprocessing)
                if attempts == 1:
                    print(f"Warning: Failed to load image at idx {current_idx} ({path}): {type(e).__name__}")
                    raise RuntimeError(f"Error loading image at index {idx} (path: {path}): {type(e).__name__}: {str(e)}") from e

def create_dataloaders(
                metadata_df: pd.DataFrame,
                img_prefix: str,
                data_config: dict,
                training_config: dict,
                splits_config: dict,
                image_config: dict,
                data_processing_config: dict,
                augmentation_config: dict) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict[str, Any]]:
    """Create train, validation, and optional test DataLoaders with stratification"""
    logger.info("Creating DataLoaders")
    
    batch_size = training_config['batch_size']
    num_workers = training_config['num_workers']
    prefetch_factor = training_config['prefetch_factor']
    test_size = splits_config['test_size']
    val_size = splits_config['val_size']
    seed = training_config['seed']
    compute_stats = data_processing_config['compute_stats']
    img_size = tuple(image_config['size'])
    weighted_sampling = data_processing_config['weighted_sampling']
    use_local = data_config['use_local']
    
    # Determine image prefix
    if use_local:
        img_prefix = data_config['img_prefix']
        logger.info(f"Using local images from: {img_prefix}")
    elif not img_prefix:
        logger.warning("No img_prefix provided, using paths as-is")
    
    use_persistent_workers = num_workers > 0
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
                                df=train_df, 
                                img_col=IMG_COL, 
                                label_col=LABEL_COL, 
                                img_prefix=img_prefix, 
                                use_local=use_local,
                                transform=get_basic_transform(img_size), 
                                classes=all_classes
                                )
        temp_loader = DataLoader(
                            temp_dataset, 
                            batch_size=batch_size, 
                            num_workers=num_workers, 
                            shuffle=False
                            )
        mean, std = compute_dataset_stats(temp_loader)
    else:
        # Use default ImageNet statistics if not computing from data
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    # Create datasets
    train_dataset = ImageDataset(
                        df=train_df, 
                        img_col=IMG_COL, 
                        label_col=LABEL_COL, 
                        img_prefix=img_prefix, 
                        use_local=use_local,
                        transform=get_train_transform(mean, std, img_size, augmentation_config), 
                        classes=all_classes
                        )
    
    test_dataset = ImageDataset(
                        df=test_df, 
                        img_col=IMG_COL, 
                        label_col=LABEL_COL, 
                        img_prefix=img_prefix, 
                        use_local=use_local,
                        transform=get_test_valid_transform(mean, std, img_size), 
                        classes=all_classes
                        )
    
    if val_df is not None:
        val_dataset = ImageDataset(
                        df=val_df, 
                        img_col=IMG_COL, 
                        label_col=LABEL_COL, 
                        img_prefix=img_prefix, 
                        use_local=use_local,
                        transform=get_test_valid_transform(mean, std, img_size), 
                        classes=all_classes
                        )
    else:
        val_dataset = None
    
    # Create dataloaders with optional weighted sampling    
    if not use_local:
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
    else:
        dataloader_kwargs = {
            'batch_size': batch_size, 
            'num_workers': num_workers, 
            'prefetch_factor': prefetch_factor if num_workers > 0 else None, 
            'pin_memory': torch.cuda.is_available(),
            'persistent_workers': use_persistent_workers,
        }

    # Compute class frequency weights, also used in focal loss
    class_counts = train_df[LABEL_COL].value_counts()
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_df[LABEL_COL]]

    
    if weighted_sampling:
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, sampler=sampler, drop_last=True, **dataloader_kwargs)
        logger.info("Using weighted sampling for imbalanced data")
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **dataloader_kwargs)    
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs) if val_dataset is not None else None
    info = {
        'num_classes': num_classes, 'classes': all_classes,
        'class_to_idx': train_dataset.class_to_idx,
        'train_size': len(train_dataset), 
        'test_size': len(test_dataset),
        'val_size': len(val_dataset) if val_dataset is not None else 0,
        'mean': mean, 'std': std,
        'img_size': img_size, 'batch_size': batch_size,
        'sample_weights': sample_weights,
    }
    
    return train_loader, val_loader, test_loader, info