"""Utility functions for dataloader pipeline"""
import logging
import pandas as pd
import torch
import os
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional
try:
    from .constants import TEXT_DESC_COL, ORIG_FILENAME_COL
except ImportError:
    from constants import TEXT_DESC_COL, ORIG_FILENAME_COL

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

def load_metadata(source: str, min_samples: int, datasets: List[str] = None, derm1m_sources: List[str] = None, has_text: bool = None, data_fraction: float = None) -> pd.DataFrame:
    """
    Load metadata from GCS or local file and filter by minimum samples and datasets to be used
    
    Args:
        source: Path to CSV (gs://bucket/path or local/path.csv)
        min_samples: Minimum images per label
        datasets: List of datasets to be used (None = all datasets)
        derm1m_sources: List of Derm1M sources to be used
        has_text: If `True`, include only entries with text data. If `False`, include only entries without text data. If `None`, do not filter on text data.
        data_fraction: Fraction of data to use (0.0-1.0). If None, use all data. Sampling is stratified by label.
    """
    logger.info(f"Loading metadata from: {source}")
    metadata = pd.read_csv(source, low_memory=False)
    logger.info(f"Total images loaded: {len(metadata):,}")
    
    # Filter labels with insufficient samples
    label_counts = metadata["label"].value_counts()
    preserve_labels = label_counts[label_counts >= min_samples].index
    logger.info(f"Labels with >= {min_samples} images: {len(preserve_labels):,}")
    
    metadata = metadata[metadata['label'].isin(preserve_labels)].reset_index(drop=True)

    # Filter for listed datasets only
    if datasets is not None:
        metadata = metadata[metadata['dataset'].isin(datasets)].reset_index(drop=True)

    # Filter to include or not include text data
    if has_text is not None:
        logger.info(f"Filtering to include only images {'with' if has_text else 'without'} text.")
        if has_text:
            keep_rows = metadata[TEXT_DESC_COL].notna() & (metadata[TEXT_DESC_COL] != '')
        else:
            keep_rows = metadata[TEXT_DESC_COL].isna() | (metadata[TEXT_DESC_COL] == '')
        metadata = metadata[keep_rows].reset_index(drop=True)
        logger.info(f"Images {'with' if has_text else 'without'} text descriptions: {metadata.shape[0]}")

    # If using Derm1M, filter for listed sources only
    if 'derm1m' in datasets:
        keep_mask = (
            (metadata['dataset'] != 'derm1m') |  # Keep all non-derm1m
            (metadata[ORIG_FILENAME_COL].str.split('/').str[0].isin(derm1m_sources))  # Keep matching derm1m
        )
        metadata = metadata[keep_mask].reset_index(drop=True)

    # Sample a fraction of data if specified (stratified by label)
    if data_fraction is not None and 0 < data_fraction < 1:
        logger.info(f"Sampling {data_fraction*100:.1f}% of data (stratified by label)...")
        # Filter out classes with < 2 samples (needed for stratified sampling)
        label_counts = metadata['label'].value_counts()
        valid_classes = label_counts[label_counts >= 2].index
        classes_removed = len(label_counts) - len(valid_classes)
        if classes_removed > 0:
            logger.warning(f"Filtering out {classes_removed} classes with < 2 samples (cannot be stratified)")
            metadata = metadata[metadata['label'].isin(valid_classes)].reset_index(drop=True)
        
        metadata, _ = train_test_split(metadata, train_size=data_fraction, stratify=metadata['label'], random_state=42)
        metadata = metadata.reset_index(drop=True)
        logger.info(f"Sampled {len(metadata):,} images ({data_fraction*100:.1f}% of filtered data)")

    logger.info(f"Images after filtering: {len(metadata):,}")
    logger.info(f"Datasets: {datasets if datasets else 'all'}")
    
    return metadata

def stratified_split(df: pd.DataFrame, label_col: str = "label", test_size: float = 0.2, val_size: float = None, seed: int = 42) -> Tuple[pd.DataFrame, ...]:
    """Split data into train/test or train/val/test sets with stratification"""
    # Filter out classes with < 2 samples (needed for stratified splitting)
    label_counts = df[label_col].value_counts()
    valid_classes = label_counts[label_counts >= 2].index
    classes_removed = len(label_counts) - len(valid_classes)
    if classes_removed > 0:
        logger.warning(f"Filtering out {classes_removed} classes with < 2 samples (cannot be stratified)")
        df = df[df[label_col].isin(valid_classes)].reset_index(drop=True)
    
    if val_size is None:
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=seed)
        logger.info(f"Train samples: {len(train_df):,}, Test samples: {len(test_df):,}")
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    else:
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=seed)
        adjusted_val_size = val_size / (1 - test_size)
        # Check again for train/val split (after train/test split, some classes might have < 2 samples)
        train_label_counts = train_df[label_col].value_counts()
        train_valid_classes = train_label_counts[train_label_counts >= 2].index
        train_classes_removed = len(train_label_counts) - len(train_valid_classes)
        if train_classes_removed > 0:
            logger.warning(f"Filtering out {train_classes_removed} classes from train set with < 2 samples (cannot be stratified for val split)")
            train_df = train_df[train_df[label_col].isin(train_valid_classes)].reset_index(drop=True)
        
        train_df, val_df = train_test_split(train_df, test_size=adjusted_val_size, stratify=train_df[label_col], random_state=seed)
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


def save_checkpoint(model: Optional[torch.nn.Module], optimizer: Optional[torch.optim.Optimizer], 
                   epoch: int, loss: float, config: Dict[str, Any], 
                   save_dir: str, experiment_name: str, is_best: bool = False,
                   additional_info: Optional[Dict[str, Any]] = None, 
                   vision_model: Optional[torch.nn.Module] = None,
                   multimodal_classifier: Optional[torch.nn.Module] = None) -> str:
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        config: Configuration dictionary
        save_dir: Directory to save checkpoints
        experiment_name: Name of the experiment
        is_best: Whether this is the best model so far
        additional_info: Additional information to save
        
    Returns:
        Path to saved checkpoint
    """
    # Create save directory if it doesn't exist
    # For Modal volumes, ensure proper permissions
    try:
        os.makedirs(save_dir, mode=0o755, exist_ok=True)
    except OSError as e:
        if e.errno == 30:  # Read-only file system
            logger.error(
                f"Cannot create directory {save_dir}: Read-only file system.\n"
                f"This usually means the Modal volume is not properly mounted.\n"
                f"Verify that the volume 'training-checkpoints' exists and is mounted at /checkpoints."
            )
            raise
        else:
            logger.error(f"Failed to create directory {save_dir}: {e}")
            raise
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'experiment_name': experiment_name
    }
    
    # Handle model state dicts
    if model is not None:
        checkpoint['model_state_dict'] = model.state_dict()
    elif vision_model is not None and multimodal_classifier is not None:
        # New architecture with multimodal classifier
        checkpoint['vision_model_state_dict'] = vision_model.state_dict()
        checkpoint['multimodal_classifier_state_dict'] = multimodal_classifier.state_dict()
    elif vision_model is not None and images_classifier is not None:
        # Legacy architecture with separate classifiers (for backwards compatibility)
        checkpoint['vision_model_state_dict'] = vision_model.state_dict()
        checkpoint['images_classifier_state_dict'] = images_classifier.state_dict()
        if text_classifier is not None:
            checkpoint['text_classifier_state_dict'] = text_classifier.state_dict()
    else:
        raise ValueError("Either model, or vision_model with multimodal_classifier, or vision_model with images_classifier must be provided")
    
    # Handle optimizer state dict
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    # Support saving multiple optimizers if provided in additional_info
    if additional_info and 'optimizers_state_dict' in additional_info:
        checkpoint['optimizers_state_dict'] = additional_info['optimizers_state_dict']
    
    # Add additional info if provided
    if additional_info:
        checkpoint.update(additional_info)
    
    # Determine filename
    if is_best:
        filename = f"{experiment_name}_best.pth"
    else:
        filename = f"{experiment_name}_epoch_{epoch:03d}.pth"
    
    filepath = os.path.join(save_dir, filename)
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    
    if is_best:
        logger.info(f"Best model saved to: {filepath}")
    else:
        logger.info(f"Checkpoint saved to: {filepath}")
    
    return filepath


def load_checkpoint(checkpoint_path: str, model: Optional[torch.nn.Module] = None, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: Optional[torch.device] = None,
                   vision_model: Optional[torch.nn.Module] = None,
                   images_classifier: Optional[torch.nn.Module] = None,
                   text_classifier: Optional[torch.nn.Module] = None,
                   multimodal_classifier: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: PyTorch optimizer to load state into (optional)
        device: Device to load checkpoint on (optional)
        
    Returns:
        Dictionary containing checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    if device is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path)
    
    # Load model state
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif vision_model is not None and multimodal_classifier is not None:
        # New architecture with multimodal classifier
        if 'vision_model_state_dict' in checkpoint:
            vision_model.load_state_dict(checkpoint['vision_model_state_dict'])
        if 'multimodal_classifier_state_dict' in checkpoint:
            multimodal_classifier.load_state_dict(checkpoint['multimodal_classifier_state_dict'])
        else:
            logger.warning("multimodal_classifier_state_dict not found in checkpoint. This may be a legacy checkpoint.")
    elif vision_model is not None and images_classifier is not None:
        # Legacy architecture with separate classifiers (for backwards compatibility)
        if 'vision_model_state_dict' in checkpoint:
            vision_model.load_state_dict(checkpoint['vision_model_state_dict'])
        if 'images_classifier_state_dict' in checkpoint:
            images_classifier.load_state_dict(checkpoint['images_classifier_state_dict'])
        if text_classifier is not None and 'text_classifier_state_dict' in checkpoint:
            text_classifier.load_state_dict(checkpoint['text_classifier_state_dict'])
    else:
        raise ValueError("Either model, or vision_model with multimodal_classifier, or vision_model with images_classifier must be provided")
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Log checkpoint info
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    timestamp = checkpoint.get('timestamp', 'unknown')
    
    logger.info(f"Loaded checkpoint - Epoch: {epoch}, Loss: {loss}, Timestamp: {timestamp}")
    
    return checkpoint


def get_latest_checkpoint(save_dir: str, experiment_name: str) -> Optional[str]:
    """
    Get the path to the latest checkpoint for an experiment
    
    Args:
        save_dir: Directory containing checkpoints
        experiment_name: Name of the experiment
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    if not os.path.exists(save_dir):
        return None
    
    # Look for checkpoint files
    checkpoint_files = []
    for filename in os.listdir(save_dir):
        if filename.startswith(experiment_name) and filename.endswith('.pth'):
            filepath = os.path.join(save_dir, filename)
            # Get modification time
            mtime = os.path.getmtime(filepath)
            checkpoint_files.append((filepath, mtime))
    
    if not checkpoint_files:
        return None
    
    # Return the most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: x[1])
    return latest_checkpoint[0]


def cleanup_old_checkpoints(save_dir: str, experiment_name: str, keep_last: int = 5):
    """
    Clean up old checkpoint files, keeping only the most recent ones
    
    Args:
        save_dir: Directory containing checkpoints
        experiment_name: Name of the experiment
        keep_last: Number of recent checkpoints to keep
    """
    if not os.path.exists(save_dir):
        return
    
    # Get all checkpoint files (excluding best model)
    checkpoint_files = []
    for filename in os.listdir(save_dir):
        if (filename.startswith(experiment_name) and 
            filename.endswith('.pth') and 
            'best' not in filename):
            filepath = os.path.join(save_dir, filename)
            mtime = os.path.getmtime(filepath)
            checkpoint_files.append((filepath, mtime))
    
    if len(checkpoint_files) <= keep_last:
        return
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    
    # Remove old checkpoints
    for filepath, _ in checkpoint_files[keep_last:]:
        try:
            os.remove(filepath)
            logger.info(f"Removed old checkpoint: {filepath}")
        except OSError as e:
            logger.warning(f"Failed to remove checkpoint {filepath}: {e}")
