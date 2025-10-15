"""Training class for skin disease classification"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import logger, save_checkpoint, load_checkpoint, cleanup_old_checkpoints
import time
from collections import defaultdict


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
        self.output_config = config['output']
        self.checkpoint_config = config['checkpoint']
        
        # Initialize optimizers (requires separate backbone and head)
        self.optimizer_backbone, self.optimizer_head = self._initialize_optimizers()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.start_epoch = 0
    
    
    def _make_optimizer(self, name_cfg, lr, wd, betas, eps, momentum, params):
        if name_cfg == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
        if name_cfg == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
        if name_cfg == 'sgd':
            return optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
        raise ValueError(f"Unsupported optimizer: {name_cfg}")

    def _initialize_optimizers(self):
        """Initialize backbone/head optimizers from optimizer_backbone/head configs (required)."""
        opt_backbone_cfg = self.config.get('optimizer_backbone')
        opt_head_cfg = self.config.get('optimizer_head')
        if not opt_backbone_cfg or not opt_head_cfg:
            raise ValueError("optimizer_backbone and optimizer_head configs are required; single-optimizer mode is disabled.")

        backbone_params = list(self.model.backbone.parameters())
        head_params = list(self.model.classifier.parameters())

        ob_name = opt_backbone_cfg['name'].lower()
        ob_lr = opt_backbone_cfg['learning_rate']
        ob_wd = opt_backbone_cfg['weight_decay']
        ob_betas = opt_backbone_cfg.get('betas', (0.9, 0.999))
        ob_eps = opt_backbone_cfg.get('eps', 1e-8)
        ob_momentum = opt_backbone_cfg.get('momentum', 0.9)

        oh_name = opt_head_cfg['name'].lower()
        oh_lr = opt_head_cfg['learning_rate']
        oh_wd = opt_head_cfg['weight_decay']
        oh_betas = opt_head_cfg.get('betas', (0.9, 0.999))
        oh_eps = opt_head_cfg.get('eps', 1e-8)
        oh_momentum = opt_head_cfg.get('momentum', 0.9)

        optimizer_backbone = self._make_optimizer(ob_name, ob_lr, ob_wd, ob_betas, ob_eps, ob_momentum, backbone_params)
        optimizer_head = self._make_optimizer(oh_name, oh_lr, oh_wd, oh_betas, oh_eps, oh_momentum, head_params)
        return optimizer_backbone, optimizer_head
    
    
    def train(self):
        """Main training loop"""
        num_epochs = self.training_config['num_epochs']
        patience = self.training_config['patience']
        validation_interval = max(1, int(self.training_config.get('validation_interval', 1)))
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Starting from epoch: {self.start_epoch}")
        bb_cfg = self.config['optimizer_backbone']
        hd_cfg = self.config['optimizer_head']
        logger.info(f"Optimizers -> backbone: {bb_cfg['name']} (lr={bb_cfg['learning_rate']}), head: {hd_cfg['name']} (lr={hd_cfg['learning_rate']})")
        
        for epoch in range(self.start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_acc, train_f1 = self._train_epoch()
            
            ran_validation = False
            val_loss = None
            val_acc = None
            val_f1 = None

            # Validation phase only on interval
            if self.val_loader is not None and ((epoch + 1) % validation_interval == 0):
                ran_validation = True
                val_loss, val_acc, val_f1 = self.validate()
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

                # Early stopping only when validation runs
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_loss, is_best=True)
                else:
                    self.patience_counter += 1

                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # If validation did not run, still log training metrics
            if not ran_validation:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
                
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
        accuracies = []
        # Running confusion matrix for macro-F1
        num_classes = self.info['num_classes']
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} - Training")
        stime = time.time()
        for batch_idx, (data, target) in enumerate(pbar):
            #print(f"Time taken: {time.time() - stime}")
            # Move data to device
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer_backbone.zero_grad()
            self.optimizer_head.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.model.loss(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer_backbone.step()
            self.optimizer_head.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            total = target.size(0)
            # Update confusion matrix
            pred_flat = pred.view(-1).detach().cpu().numpy()
            target_flat = target.view(-1).detach().cpu().numpy()
            for t, p in zip(target_flat, pred_flat):
                confusion[t, p] += 1

            avg_running_loss = total_loss / (batch_idx + 1)
            accuracies.append(100. * correct / total)
            avg_running_acc = np.mean(accuracies)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{avg_running_loss:.4f}',
                'Acc': f'{avg_running_acc:.2f}%'
            })
            stime = time.time()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = np.mean(accuracies)
        # Compute macro-F1 from confusion
        with np.errstate(divide='ignore', invalid='ignore'):
            tp = np.diag(confusion).astype(np.float64)
            fp = confusion.sum(axis=0) - tp
            fn = confusion.sum(axis=1) - tp
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f1_per_class = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
            macro_f1 = float(np.mean(f1_per_class))

        return avg_loss, accuracy, macro_f1
    
    def validate(self):
        """Validation loop"""
        if self.val_loader is None:
            logger.warning("No validation loader available")
            return None, None, None
            
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_classes = self.info['num_classes']
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for data, target in pbar:
                # Move data to device
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # Forward pass
                output = self.model(data)
                loss = self.model.loss(output, target)
                
                # Statistics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                pred_flat = pred.view(-1).detach().cpu().numpy()
                target_flat = target.view(-1).detach().cpu().numpy()
                for t, p in zip(target_flat, pred_flat):
                    confusion[t, p] += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        with np.errstate(divide='ignore', invalid='ignore'):
            tp = np.diag(confusion).astype(np.float64)
            fp = confusion.sum(axis=0) - tp
            fn = confusion.sum(axis=1) - tp
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f1_per_class = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
            macro_f1 = float(np.mean(f1_per_class))
        
        return avg_loss, accuracy, macro_f1
    
    def test(self):
        """Test loop"""
        logger.info("Starting test evaluation...")
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        num_classes = self.info['num_classes']
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            for data, target in pbar:
                # Move data to device
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # Forward pass
                output = self.model(data)
                loss = self.model.loss(output, target)
                
                # Statistics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                pred_flat = pred.view(-1).detach().cpu().numpy()
                target_flat = target.view(-1).detach().cpu().numpy()
                for t, p in zip(target_flat, pred_flat):
                    confusion[t, p] += 1
                
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
        with np.errstate(divide='ignore', invalid='ignore'):
            tp = np.diag(confusion).astype(np.float64)
            fp = confusion.sum(axis=0) - tp
            fn = confusion.sum(axis=1) - tp
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f1_per_class = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
            macro_f1 = float(np.mean(f1_per_class))
        
        logger.info(f"Test Results:")
        logger.info(f"  Test Loss: {avg_loss:.4f}")
        logger.info(f"  Test Accuracy: {accuracy:.2f}%")
        logger.info(f"  Test Macro-F1: {macro_f1:.4f}")
        logger.info(f"  Correct: {correct}/{total}")
        
        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': all_predictions,
            'targets': all_targets,
            'macro_f1': macro_f1
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

        # Include both optimizer states if using split optimizers
        if self.optimizer_backbone is not None and self.optimizer_head is not None:
            additional_info['optimizers_state_dict'] = {
                'backbone': self.optimizer_backbone.state_dict(),
                'head': self.optimizer_head.state_dict(),
            }
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model=self.model,
            optimizer=None,
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
