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

def load_metadata(source: str, min_samples: int, datasets: List[str] = None) -> pd.DataFrame:
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
    metadata = metadata[metadata['dataset'].isin(datasets)]
    logger.info(f"Images after filtering: {len(metadata):,}")
    logger.info(f"Datasets: {datasets}")
    
    return metadata

def stratified_split(df: pd.DataFrame, label_col: str = "label", test_size: float = 0.2, val_size: float = None, seed: int = 42) -> Tuple[pd.DataFrame, ...]:
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


def save_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], 
                   epoch: int, loss: float, config: Dict[str, Any], 
                   save_dir: str, experiment_name: str, is_best: bool = False,
                   additional_info: Optional[Dict[str, Any]] = None) -> str:
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
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': (optimizer.state_dict() if optimizer is not None else None),
        'loss': loss,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'experiment_name': experiment_name
    }
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


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
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
    model.load_state_dict(checkpoint['model_state_dict'])
    
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