"""Main entry point for training workflow"""

import argparse
import yaml
import torch
import warnings

from train.train import Trainer
from utils import load_metadata, logger, load_checkpoint
from dataloader.dataloader import create_dataloaders
from model.vision.vit import VisionTransformer
from model.vision.cnn import CNNModel
from model.classifier.classifier import Classifier
from constants import (
    GCS_BUCKET_NAME, 
    GCS_METADATA_PATH, 
    GCS_IMAGE_PREFIX,
    DEFAULT_IMAGE_MODE,
)

def initialize_model(config_path):
    """This function will initialize everything needed for training based on config being used"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from: {config_path}")
    
    # Extract configuration sections
    data_config = config['data']
    training_config = config['training']
    augmentation_config = config['augmentation']
    
    # Set seed for reproducibility
    seed = data_config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to: {seed}")
    
    # Determine metadata path depending on training using GCP or local data
    if data_config.get('use_local'):
        metadata_path = data_config['metadata_path']
        img_prefix = data_config['img_prefix']
    else:
        metadata_path = GCS_METADATA_PATH
        img_prefix = GCS_IMAGE_PREFIX
    
    # This will load the metadata file and filter it
    # This decides what we train on 
    metadata = load_metadata(metadata_path, min_samples=data_config['min_samples_per_label'], datasets=data_config['datasets'], has_text=data_config['has_text'])
    
    # Create dataloaders with splits
    train_loader, val_loader, test_loader, info = create_dataloaders(
        metadata_df=metadata,
        img_prefix=img_prefix,
        data_config=data_config,
        training_config=training_config,
        augmentation_config=augmentation_config,
    )
    
    logger.info(f"  Image size: {info['img_size']}")
    logger.info(f"  Batch size: {info['batch_size']}")    
    # Update configs with number of classes from data
    vision_config = config['vision_model'].copy()
    vision_config['img_size'] = data_config['img_size']
    
    # Initialize vision model
    logger.info("Initializing vision model...")
    if vision_config['name'] in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']:
        vision_model = VisionTransformer(vision_config)
    else:
        vision_model = CNNModel(vision_config)
    vision_model = vision_model.to(device)
    embedding_dim = vision_model.embedding_dim

    # Update classifier config with embedding size
    classifier_config = config['classifier'].copy()
    classifier_config['num_classes'] = info['num_classes']
    classifier_config['sample_weights'] = info['sample_weights']
    classifier_config['input_size'] = embedding_dim
    
    # Initialize classifier
    classifier = Classifier(classifier_config)
    classifier = classifier.to(device)
    
    # Log model information
    vision_info = vision_model.get_vision_info()
    classifier_info = classifier.get_classifier_info()
    
    logger.info(f"Vision Model: {vision_info['model_name']}")
    logger.info(f"Vision parameters: {vision_info['total_parameters']:,}")
    logger.info(f"Classifier parameters: {classifier_info['total_parameters']:,}")
    logger.info(f"Total parameters: {vision_info['total_parameters'] + classifier_info['total_parameters']:,}")
    logger.info(f"Classifier hidden sizes: {classifier_info['hidden_sizes']}")
    logger.info(f"Classifier activation: {classifier_info['activation']}")
    
    # Initialize trainer with dataloaders, models, and device
    trainer = Trainer(
                config = config, 
                train_loader = train_loader, 
                val_loader = val_loader, 
                test_loader = test_loader, 
                info = info, 
                vision_model = vision_model,
                classifier = classifier,
                device = device
                )
    
    # Load checkpoint if specified
    checkpoint_config = config.get('checkpoint', {})
    if checkpoint_config.get('load_from') is not None:
        logger.info("Loading checkpoint")
        try:
            checkpoint = load_checkpoint(
                checkpoint_path=checkpoint_config['load_from'],
                model=None,
                vision_model=vision_model,
                classifier=classifier,
                device=device
            )
            logger.info("Checkpoint loaded successfully!")
            
            # Update trainer state for resuming training
            trainer.start_epoch = checkpoint.get('epoch', 0) + 1
            trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            trainer.patience_counter = checkpoint.get('patience_counter', 0)
            
            # Load scheduler states if available
            schedulers_state_dict = checkpoint.get('schedulers_state_dict')
            if schedulers_state_dict:
                if trainer.scheduler_vision is not None and 'vision' in schedulers_state_dict:
                    trainer.scheduler_vision.load_state_dict(schedulers_state_dict['vision'])
                if trainer.scheduler_classifier is not None and 'classifier' in schedulers_state_dict:
                    trainer.scheduler_classifier.load_state_dict(schedulers_state_dict['classifier'])
                logger.info("Scheduler states loaded from checkpoint")
            
            # Load vision freeze state
            trainer.vision_frozen = checkpoint.get('vision_frozen', False)
            trainer.n_warmup_epochs = checkpoint.get('n_warmup_epochs', 0)
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Continuing with untrained model...")
    
    logger.info("Model and training setup initialized!")

    return_dict = {
        'trainer': trainer,
        'config': config,
        'vision_model': vision_model,
        'classifier': classifier,
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
