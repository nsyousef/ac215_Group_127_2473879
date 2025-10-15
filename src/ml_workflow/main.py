"""Main entry point for training workflow"""

import argparse
import yaml
import torch
from train.train import Trainer
from utils import load_metadata, logger, load_checkpoint
from dataloader.dataloader import create_dataloaders
from model.imagenet import ImageNetModel
from constants import (
    GCS_BUCKET_NAME, 
    GCS_METADATA_PATH, 
    GCS_IMAGE_PREFIX,
)

def initialize_model(config_path):
    """Initialize model and training setup"""
    
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from: {config_path}")
    
    # Extract configuration sections
    data_config = config['data']
    training_config = config['training']
    splits_config = config['splits']
    image_config = config['image']
    data_processing_config = config['data_processing']
    augmentation_config = config['augmentation']
    
    # Determine metadata path
    if data_config.get('use_local'):
        metadata_path = data_config['metadata_path']
        img_prefix = data_config['img_prefix']
    else:
        metadata_path = GCS_METADATA_PATH
        img_prefix = GCS_IMAGE_PREFIX
    
    logger.info(f"Loading metadata from: {metadata_path}")
    metadata = load_metadata(metadata_path, min_samples=data_config['min_samples_per_label'], datasets=data_config['datasets'])
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader, info = create_dataloaders(
        metadata_df=metadata,
        img_prefix=img_prefix,
        data_config=data_config,
        training_config=training_config,
        splits_config=splits_config,
        image_config=image_config,
        data_processing_config=data_processing_config,
        augmentation_config=augmentation_config,
    )
    
    logger.info("DataLoader Info:")
    logger.info(f"  Number of classes: {info['num_classes']}")
    logger.info(f"  Train size: {info['train_size']}")
    logger.info(f"  Val size: {info['val_size']}")
    logger.info(f"  Test size: {info['test_size']}")
    #logger.info(f"  Mean: {info['mean']}")
    #logger.info(f"  Std: {info['std']}")
    logger.info(f"  Image size: {info['img_size']}")
    logger.info(f"  Batch size: {info['batch_size']}")
    logger.info("Dataloaders ready!")
    
    # Update model config with number of classes from data
    model_config = config['model'].copy()
    model_config['num_classes'] = info['num_classes']
    model_config['img_size'] = config['image']['size']
    
    # Initialize model
    logger.info("Initializing model...")
    model = ImageNetModel(model_config)
    model = model.to(device)
    
    # Log model information
    model_info = model.get_model_info()
    logger.info(f"Model: {model_info['model_name']}")
    logger.info(f"Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    logger.info(f"Hidden sizes: {model_info['hidden_sizes']}")
    logger.info(f"Activation: {model_info['activation']}")
    
    # Initialize trainer with dataloaders, model, and device
    trainer = Trainer(config, train_loader, val_loader, test_loader, info, model, device)
    
    # Load checkpoint if specified
    checkpoint_config = config.get('checkpoint', {})
    if checkpoint_config.get('load_from') is not None:
        logger.info("Loading checkpoint")
        try:
            checkpoint = load_checkpoint(
                checkpoint_path=checkpoint_config['load_from'],
                model=model,
                device=device
            )
            logger.info("Checkpoint loaded successfully!")
            
            # Update trainer state for resuming training
            trainer.start_epoch = checkpoint.get('epoch', 0) + 1
            trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            trainer.patience_counter = checkpoint.get('patience_counter', 0)
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Continuing with untrained model...")
    
    logger.info("Model and training setup initialized!")

    return_dict = {
        'trainer': trainer,
        'config': config,
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'info': info
    }
    
    return return_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train skin disease classification model')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to configuration YAML file')
    args = parser.parse_args()

    return_dict = initialize_model(args.config)

    trainer = return_dict['trainer']
    trainer.train()

