import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import logger, save_checkpoint, load_checkpoint, cleanup_old_checkpoints
import time
import wandb
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from train.train_utils import make_optimizer

class Trainer:
    
    def __init__(self, config, train_loader, val_loader, test_loader, info, vision_model, classifier, device):
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
        self.vision_model = vision_model
        self.classifier = classifier
        self.device = device
        
        # Extract configuration sections
        self.data_config = config['data']
        self.training_config = config['training']
        self.output_config = config['output']
        self.checkpoint_config = config['checkpoint']
        
        self.optimizer_vision, self.optimizer_classifier = self._initialize_optimizers()
        self.scheduler_vision, self.scheduler_classifier = self._initialize_schedulers()
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.start_epoch = 0
        
        self.n_warmup_epochs = self.training_config['n_warmup_epochs']
        self.vision_frozen = False
        self.accuracy_history = {}
    
    def _initialize_optimizers(self):
        """Initialize vision/classifier optimizers from optimizer config (required)."""
        optimizer_config = self.config.get('optimizer')

        opt_vision_cfg = optimizer_config.get('vision_model')
        opt_classifier_cfg = optimizer_config.get('classifier')

        vision_params = list(self.vision_model.parameters())
        classifier_params = list(self.classifier.parameters())

        ob_name = opt_vision_cfg['name'].lower()
        ob_lr = opt_vision_cfg['learning_rate']
        ob_wd = opt_vision_cfg['weight_decay']
        ob_betas = opt_vision_cfg.get('betas', (0.9, 0.999))
        ob_eps = opt_vision_cfg.get('eps', 1e-8)
        ob_momentum = opt_vision_cfg.get('momentum', 0.9)

        oh_name = opt_classifier_cfg['name'].lower()
        oh_lr = opt_classifier_cfg['learning_rate']
        oh_wd = opt_classifier_cfg['weight_decay']
        oh_betas = opt_classifier_cfg.get('betas', (0.9, 0.999))
        oh_eps = opt_classifier_cfg.get('eps', 1e-8)
        oh_momentum = opt_classifier_cfg.get('momentum', 0.9)

        optimizer_vision = make_optimizer(ob_name, ob_lr, ob_wd, ob_betas, ob_eps, ob_momentum, vision_params)
        optimizer_classifier = make_optimizer(oh_name, oh_lr, oh_wd, oh_betas, oh_eps, oh_momentum, classifier_params)
        return optimizer_vision, optimizer_classifier
    
    def _initialize_schedulers(self):
        """Initialize cosine annealing schedulers for both optimizers."""
        num_epochs = self.training_config['num_epochs']
        
        # Get scheduler configuration
        scheduler_config = self.training_config.get('scheduler', {})
        use_cosine_annealing = scheduler_config.get('use_cosine_annealing', True)
        
        if use_cosine_annealing:
            # Cosine annealing schedulers
            scheduler_vision= CosineAnnealingLR(
                self.optimizer_vision, 
                T_max=num_epochs,
                eta_min=scheduler_config.get('vision_eta_min', 0.0)
            )
            scheduler_classifier = CosineAnnealingLR(
                self.optimizer_classifier, 
                T_max=num_epochs,
                eta_min=scheduler_config.get('classifier_eta_min', 0.0)
            )
        else:
            # No schedulers (step function that does nothing)
            scheduler_vision = None
            scheduler_classifier = None
            
        return scheduler_vision, scheduler_classifier
    
    def train(self):
        """Main training loop"""
        num_epochs = self.training_config['num_epochs']
        patience = self.training_config['patience']
        validation_interval = max(1, int(self.training_config.get('validation_interval', 1)))
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Starting from epoch: {self.start_epoch}")
        vision_optimizer_cfg = self.config['optimizer']['vision_model']
        classifier_optimizer_cfg = self.config['optimizer']['classifier']
        logger.info(f"Optimizers -> vision_model: {vision_optimizer_cfg['name']} (lr={vision_optimizer_cfg['learning_rate']}), classifier: {classifier_optimizer_cfg['name']} (lr={classifier_optimizer_cfg['learning_rate']})")

        with wandb.init(project=self.output_config['wandb_project'], name=self.output_config['experiment_name'], config=self.config) as run:

            config = run.config
        
            for epoch in range(self.start_epoch, num_epochs):
                self.current_epoch = epoch
                
                # Update vision freeze state
                if epoch < self.n_warmup_epochs:
                    self.vision_model._freeze_layers(num_layers = -1) # will freeze all layers
                else:
                    self.vision_model._freeze_layers() # Will unfreeze last n layers
                # Training phase
                train_loss, train_acc, train_f1 = self._train_epoch()

                run.log({
                    'train/loss': train_loss,
                    'train/acc': train_acc,
                    'train/f1': train_f1,
                    'epoch': epoch,
                    'params/learning_rate_vision': vision_optimizer_cfg['learning_rate'],
                    'params/learning_rate_classifier': classifier_optimizer_cfg['learning_rate'],
                })
                
                ran_validation = False
                val_loss = None
                val_acc = None
                val_f1 = None

                # Validation phase only on interval
                if self.val_loader is not None and ((epoch + 1) % validation_interval == 0):
                    ran_validation = True
                    val_loss, val_acc, val_f1, confusion_matrix = self.validate()
                    logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

                    # Create and log confusion analysis table
                    if confusion_matrix is not None:
                        class_names = self.info['classes']
                        confusion_table, accuracy_history_table = self._make_wandb_tables(confusion_matrix, class_names)
                        run.log({"val/confusion_analysis": confusion_table})
                        run.log({"val/accuracy_history": accuracy_history_table})

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
                
                run.log({
                    'val/loss': val_loss,
                    'val/acc': val_acc,
                    'val/f1': val_f1,
                    'epoch': epoch,
                })
                
                # Save regular checkpoint
                save_frequency = self.checkpoint_config.get('save_frequency', 10)
                if (epoch + 1) % save_frequency == 0:
                    self._save_checkpoint(epoch, train_loss, is_best=False)
                
                # Step schedulers
                if epoch > self.n_warmup_epochs:
                    if self.scheduler_vision is not None:
                        self.scheduler_vision.step()
                if self.scheduler_classifier is not None:
                    self.scheduler_classifier.step()
                
            logger.info("Training completed!")
    
    def _train_epoch(self):
        """Train for one epoch"""
        self.vision_model.train()
        self.classifier.train()
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

            # Move data to device
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            # Zero gradients
            if not self.vision_frozen:
                self.optimizer_vision.zero_grad()
            self.optimizer_classifier.zero_grad()
            
            # Forward pass
            vision_embeddings = self.vision_model(data)
            output = self.classifier(vision_embeddings)
            loss = self.classifier.loss(output, target)
            
            # Backward pass
            loss.backward()
            if not self.vision_frozen:
                self.optimizer_vision.step()
            self.optimizer_classifier.step()
            
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
            return None, None, None, None
            
        self.vision_model.eval()
        self.classifier.eval()
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
                vision_embeddings = self.vision_model(data)
                output = self.classifier(vision_embeddings)
                loss = self.classifier.loss(output, target)
                
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
        
        return avg_loss, accuracy, macro_f1, confusion
    
    def _make_wandb_tables(self, confusion_matrix, class_names):
        """Create wandb table with confusion matrix analysis for each class"""
        num_classes = len(class_names)
        
        # Create table data
        table_data = []
        accuracy_history_data = []
        
        for predicted_class in range(num_classes):
            # Get all incorrect predictions for this class (excluding correct predictions)
            incorrect_predictions = []
            for true_class in range(num_classes):
                if true_class != predicted_class and confusion_matrix[true_class, predicted_class] > 0:
                    incorrect_predictions.append({
                        'true_class': class_names[true_class],
                        'count': int(confusion_matrix[true_class, predicted_class])
                    })
        
            if predicted_class not in self.accuracy_history:
                self.accuracy_history[predicted_class] = []
            accuracy = confusion_matrix[predicted_class, predicted_class] / confusion_matrix[predicted_class].sum()
            self.accuracy_history[predicted_class].append(accuracy)
            
            # Sort by count (most common mistakes first)
            incorrect_predictions.sort(key=lambda x: x['count'], reverse=True)
            
            # Create distribution plot
            if incorrect_predictions:
                fig, ax = plt.subplots(figsize=(8, 4))
                true_classes = [item['true_class'] for item in incorrect_predictions]
                counts = [item['count'] for item in incorrect_predictions]
                
                # Create bar plot
                bars = ax.bar(range(len(true_classes)), counts)
                ax.set_xlabel('True Class')
                ax.set_ylabel('Count')
                ax.set_title(f'Incorrectly Classified as {class_names[predicted_class]}')
                ax.set_xticks(range(len(true_classes)))
                ax.set_xticklabels(true_classes, rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Convert plot to wandb image
                plot_image = wandb.Image(fig)
                plt.close(fig)
            else:
                # No incorrect predictions for this class
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.text(0.5, 0.5, 'No incorrect predictions', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Incorrectly Classified as {class_names[predicted_class]}')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                plot_image = wandb.Image(fig)
                plt.close(fig)
            
            # Add row to table
            table_data.append([
                class_names[predicted_class],
                plot_image
            ])

            fig, ax = plt.subplots(figsize=(8, 4))
            line = ax.plot(range(len(self.accuracy_history[predicted_class])), self.accuracy_history[predicted_class])
            accuracy_history_image = wandb.Image(fig)
            plt.close(fig)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Accuracy History')
            ax.set_xticks(range(len(self.accuracy_history[predicted_class])))
            ax.set_xticklabels(range(len(self.accuracy_history[predicted_class])), ha='right')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            accuracy_history_image = wandb.Image(fig)
            plt.close(fig)

            accuracy_history_data.append([class_names[predicted_class], accuracy_history_image])

        
        # Create wandb table
        table = wandb.Table(
            columns=["Predicted Class", "Distribution of Incorrect Classifications"],
            data=table_data
        )
        accuracy_history_table = wandb.Table(
            columns=["Class", "Accuracy History"],
            data=accuracy_history_data
        )
        
        return table, accuracy_history_table
    
    def test(self):
        """Test loop"""
        logger.info("Starting test evaluation...")
        
        self.vision_model.eval()
        self.classifier.eval()
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
                vision_embeddings = self.vision_model(data)
                output = self.classifier(vision_embeddings)
                loss = self.classifier.loss(output, target)
                
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
        if self.optimizer_vision is not None and self.optimizer_classifier is not None:
            additional_info['optimizers_state_dict'] = {
                'vision': self.optimizer_vision.state_dict(),
                'classifier': self.optimizer_classifier.state_dict(),
            }
        
        # Include scheduler states
        if self.scheduler_vision is not None or self.scheduler_classifier is not None:
            additional_info['schedulers_state_dict'] = {}
            if self.scheduler_vision is not None:
                additional_info['schedulers_state_dict']['vision'] = self.scheduler_vision.state_dict()
            if self.scheduler_classifier is not None:
                additional_info['schedulers_state_dict']['classifier'] = self.scheduler_classifier.state_dict()
        
        # Include vision freeze state
        additional_info['vision_frozen'] = self.vision_frozen
        additional_info['n_warmup_epochs'] = self.n_warmup_epochs
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model=None,
            optimizer=None,
            epoch=epoch,
            loss=loss,
            config=self.config,
            save_dir=save_dir,
            experiment_name=experiment_name,
            is_best=is_best,
            additional_info=additional_info,
            vision_model=self.vision_model,
            classifier=self.classifier
        )
        
        # Save checkpoint to wandb
        try:
            wandb.save(checkpoint_path, base_path=save_dir)
            if is_best:
                logger.info(f"Best checkpoint saved to wandb: {checkpoint_path}")
            else:
                logger.info(f"Checkpoint saved to wandb: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint to wandb: {e}")
        
        # Clean up old checkpoints if not best model
        if not is_best:
            keep_last = self.checkpoint_config.get('keep_last', 5)
            cleanup_old_checkpoints(save_dir, experiment_name, keep_last)
        
        return checkpoint_path
