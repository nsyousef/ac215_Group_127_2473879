"""Training class for skin disease classification"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import logger, save_checkpoint, load_checkpoint, cleanup_old_checkpoints


class Trainer:
    """Trainer class for handling model training"""
    
    def __init__(self, config, train_loader, val_loader, test_loader, info, model, device):
        """
        Initialize trainer with configuration, dataloaders, model, and device
        
        Args:
            config: Configuration dictionary from YAML
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (can be None)
            test_loader: Test DataLoader
            info: Dictionary with dataset information
            model: Initialized model
            device: Device to use for training (cuda/cpu)
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.info = info
        self.model = model
        self.device = device
        
        # Extract configuration sections
        self.data_config = config['data']
        self.training_config = config['training']
        self.splits_config = config['splits']
        self.image_config = config['image']
        self.data_processing_config = config['data_processing']
        self.optimizer_config = config['optimizer']
        self.output_config = config['output']
        self.checkpoint_config = config['checkpoint']
        
        # Initialize optimizer
        self.optimizer = self._initialize_optimizer()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.start_epoch = 0
    
    
    def _initialize_optimizer(self):
        """Initialize optimizer based on config"""
        optimizer_name = self.optimizer_config['name'].lower()
        learning_rate = self.optimizer_config['learning_rate']
        weight_decay = self.optimizer_config['weight_decay']
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=self.optimizer_config['betas'],
                eps=self.optimizer_config['eps']
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=self.optimizer_config['betas'],
                eps=self.optimizer_config['eps']
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=self.optimizer_config['momentum']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    
    def train(self):
        """Main training loop"""
        num_epochs = self.training_config['num_epochs']
        patience = self.training_config['patience']
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Starting from epoch: {self.start_epoch}")
        logger.info(f"Optimizer: {self.optimizer_config['name']}")
        logger.info(f"Learning rate: {self.optimizer_config['learning_rate']}")
        
        for epoch in range(self.start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_acc = self._train_epoch()
            
            # Validation phase
            if self.val_loader is not None:
                val_loss, val_acc = self.validate()
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                    # Early stopping
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        # Save best model
                        self._save_checkpoint(epoch, val_loss, is_best=True)
                    else:
                        self.patience_counter += 1
                        
                    if self.patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                else:
                    logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                
                # Save regular checkpoint
                save_frequency = self.checkpoint_config.get('save_frequency', 10)
                if (epoch + 1) % save_frequency == 0:
                    self._save_checkpoint(epoch, train_loss, is_best=False)
            
            logger.info("Training completed!")
    
    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} - Training")
        for batch_idx, (data, target) in enumerate(pbar):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.model.loss(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validation loop"""
        if self.val_loader is None:
            logger.warning("No validation loader available")
            return None, None
            
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for data, target in pbar:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.model.loss(output, target)
                
                # Statistics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def test(self):
        """Test loop"""
        logger.info("Starting test evaluation...")
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            for data, target in pbar:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.model.loss(output, target)
                
                # Statistics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Store predictions and targets for detailed analysis
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        logger.info(f"Test Results:")
        logger.info(f"  Test Loss: {avg_loss:.4f}")
        logger.info(f"  Test Accuracy: {accuracy:.2f}%")
        logger.info(f"  Correct: {correct}/{total}")
        
        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def _save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        save_dir = self.output_config['save_dir']
        experiment_name = self.output_config['experiment_name']
        
        # Prepare additional info
        additional_info = {
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'device': str(self.device)
        }
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=loss,
            config=self.config,
            save_dir=save_dir,
            experiment_name=experiment_name,
            is_best=is_best,
            additional_info=additional_info
        )
        
        # Clean up old checkpoints if not best model
        if not is_best:
            keep_last = self.checkpoint_config.get('keep_last', 5)
            cleanup_old_checkpoints(save_dir, experiment_name, keep_last)
        
        return checkpoint_path
