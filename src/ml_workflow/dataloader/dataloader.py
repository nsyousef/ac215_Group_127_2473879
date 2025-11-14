import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, List, Callable, Tuple, Dict, Any
import torch
import multiprocessing

# Try relative imports (package mode), fall back to absolute (script mode)
try:
    from ..constants import DEFAULT_IMAGE_MODE, IMG_COL, LABEL_COL, MAX_RETRIES, EMBEDDING_COL
    from ..utils import (logger, stratified_split)
    from .transform_utils import (get_basic_transform, get_train_transform, get_test_valid_transform, compute_dataset_stats)
    from .embedding_utils import embedding_to_array
except ImportError:
    from ml_workflow.constants import DEFAULT_IMAGE_MODE, IMG_COL, LABEL_COL, MAX_RETRIES, EMBEDDING_COL
    from ml_workflow.utils import (logger, stratified_split)
    from ml_workflow.dataloader.transform_utils import (get_basic_transform, get_train_transform, get_test_valid_transform, compute_dataset_stats)
    from ml_workflow.dataloader.embedding_utils import embedding_to_array
    
# If we use multiple workers with GCP, we need to ensure each worker has its own filesystem object
import gcsfs
_worker_fs = None

def _get_worker_fs():
    global _worker_fs
    if _worker_fs is None:
        try:
            # Try to use the same credentials as google.cloud.storage
            # This allows gcsfs to work with any credentials that google.cloud.storage can use
            from google.auth import default as google_auth_default
            from google.auth.exceptions import DefaultCredentialsError
            
            try:
                # Get credentials using the same method as google.cloud.storage
                credentials, project = google_auth_default()
                # Pass credentials explicitly to gcsfs
                _worker_fs = gcsfs.GCSFileSystem(
                    token=credentials,  # Pass credentials object directly
                    cache_timeout=600,      
                    default_block_size=2**22,
                    retry_reads=True
                )
            except DefaultCredentialsError:
                # Fall back to "cloud" token (application default credentials)
                # This requires: gcloud auth application-default login
                _worker_fs = gcsfs.GCSFileSystem(
                    token="cloud",
                    cache_timeout=600,
                    default_block_size=2**22,
                    retry_reads=True
                )
        except ImportError:
            # If google.auth is not available, use "cloud" token
            _worker_fs = gcsfs.GCSFileSystem(
                token="cloud",
                cache_timeout=600,
                default_block_size=2**22,
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
                 embedding_col: str,
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
            embedding_col: Column name for pre-computed text embeddings
            img_prefix: Prefix for image paths (GCS bucket or local dir)
            transform: Optional transform to apply to images
            classes: Fixed class list (inferred if None)
            use_local: Whether to use local file system or GCS
        """
        # Convert DataFrame to lists for faster, multiprocessing-safe access
        self.image_paths = df[img_col].astype(str).tolist()
        self.labels_raw = df[label_col].astype(str).tolist()
        # Pre-convert embeddings to torch tensors for better performance
        self.text_embeddings = [torch.tensor(embedding_to_array(emb)) for emb in df[embedding_col].tolist()]
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
        """Load and return image, label, and text embedding"""
        attempts = 0
        last_error = None
        current_idx = idx
        path = None

        while attempts < self.max_retries:
            try:
                current_idx = (idx + attempts) % len(self.image_paths)
                filename = self.image_paths[current_idx].lstrip("/")
                path = f"{self.img_prefix}/{filename}" if self.img_prefix else filename
                label = self.labels[current_idx]
                text_embd = self.text_embeddings[current_idx]
                
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
                return img, label, text_embd
                
            except Exception as e:
                last_error = e
                attempts += 1
                # Only print on first attempt to reduce spam (use print, not logger, for multiprocessing)
                if attempts == 1:
                    print(f"Warning: Failed to load image at idx {current_idx} ({path}): {type(e).__name__}")
                # Only raise after all retries exhausted
                if attempts >= self.max_retries:
                    raise RuntimeError(f"Error loading image at index {idx} (path: {path}) after {attempts} attempts: {type(last_error).__name__}: {str(last_error)}") from last_error

def create_dataloaders(
                metadata_df: pd.DataFrame,
                img_prefix: str,
                data_config: dict,
                training_config: dict,
                augmentation_config: dict) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict[str, Any]]:
    """Create train, validation, and optional test DataLoaders with stratification"""
    logger.info("Creating DataLoaders")
    
    batch_size = training_config['batch_size']
    num_workers = training_config['num_workers']
    prefetch_factor = training_config['prefetch_factor']
    test_size = data_config['test_size']
    val_size = data_config['val_size']
    seed = data_config['seed']
    compute_stats = training_config['compute_stats']
    img_size = tuple(data_config['img_size'])
    weighted_sampling = training_config['weighted_sampling']
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
    
    # Build dataloader kwargs once (reused for temp_loader and main dataloaders)
    dataloader_kwargs = {
        'batch_size': batch_size, 
        'num_workers': num_workers, 
        'prefetch_factor': prefetch_factor if num_workers > 0 else None, 
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': use_persistent_workers,
    }
    if not use_local and num_workers > 0:
        dataloader_kwargs['multiprocessing_context'] = multiprocessing.get_context('spawn')
        dataloader_kwargs['worker_init_fn'] = _worker_init_fn
    
    # Compute or validate statistics
    if compute_stats:
        logger.info("Computing dataset statistics from all training data")
        
        temp_dataset = ImageDataset(
                                df=train_df, 
                                img_col=IMG_COL, 
                                label_col=LABEL_COL, 
                                embedding_col=EMBEDDING_COL,
                                img_prefix=img_prefix, 
                                use_local=use_local,
                                transform=get_basic_transform(img_size), 
                                classes=all_classes
                                )
        # Reuse dataloader_kwargs, just add shuffle=False for stats computation
        temp_loader_kwargs = dataloader_kwargs.copy()
        temp_loader_kwargs['shuffle'] = False
        temp_loader = DataLoader(temp_dataset, **temp_loader_kwargs)
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
                        embedding_col=EMBEDDING_COL,
                        img_prefix=img_prefix, 
                        use_local=use_local,
                        transform=get_train_transform(mean, std, img_size, augmentation_config), 
                        classes=all_classes
                        )
    
    test_dataset = ImageDataset(
                        df=test_df, 
                        img_col=IMG_COL, 
                        label_col=LABEL_COL, 
                        embedding_col=EMBEDDING_COL,
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
                        embedding_col=EMBEDDING_COL,
                        img_prefix=img_prefix, 
                        use_local=use_local,
                        transform=get_test_valid_transform(mean, std, img_size), 
                        classes=all_classes
                        )
    else:
        val_dataset = None

    # Compute class frequency weights, also used in focal loss
    class_counts = train_df[LABEL_COL].value_counts()
    
    # For weighted sampling: need per-sample weights
    class_weights_dict = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights_dict[label] for label in train_df[LABEL_COL]]
    
    # For focal loss: need per-class weights in class-index order, classes not in training set get default weight 1.0
    class_weights_list = [class_weights_dict.get(cls, 1.0) for cls in all_classes]
    
    # Log if any classes only appear in val/test (will have untrained output units)
    train_classes = set(train_df[LABEL_COL].unique())
    val_test_only = set(all_classes) - train_classes
    if val_test_only:
        logger.warning(f"Classes only in val/test (untrained output units, weight=1.0): {sorted(val_test_only)}")
    
    if weighted_sampling:
        # Convert sample_weights to tensor (required by WeightedRandomSampler)
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)
        sampler = WeightedRandomSampler(weights=sample_weights_tensor, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, sampler=sampler, drop_last=True, **dataloader_kwargs)
        logger.info("Using weighted sampling for imbalanced data")
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **dataloader_kwargs)    
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs) if val_dataset is not None else None
    info = {
        'num_classes': num_classes, 
        'classes': all_classes,
        'class_to_idx': train_dataset.class_to_idx,
        'train_size': len(train_dataset), 
        'test_size': len(test_dataset),
        'val_size': len(val_dataset) if val_dataset is not None else 0,
        'mean': mean, 
        'std': std,
        'img_size': img_size, 
        'batch_size': batch_size,
        'class_weights': class_weights_list,  # for focal loss
    }
    
    return train_loader, val_loader, test_loader, info