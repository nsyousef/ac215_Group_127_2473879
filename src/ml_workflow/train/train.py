import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
try:
    from ..utils import logger, save_checkpoint, load_checkpoint, cleanup_old_checkpoints
except ImportError:
    # Fallback for when not imported as package (shouldn't normally happen)
    import sys
    from pathlib import Path
    # Add ml_workflow parent to path if needed
    ml_workflow_path = Path(__file__).parent.parent.parent
    if str(ml_workflow_path) not in sys.path:
        sys.path.insert(0, str(ml_workflow_path))
    from ml_workflow.utils import logger, save_checkpoint, load_checkpoint, cleanup_old_checkpoints
import time
import wandb
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from train.train_utils import make_optimizer

class Trainer:
    
    def __init__(self, config, train_loader, val_loader, test_loader, info, vision_model, images_classifier, text_classifier, fusion_config, device):
        """
        Initialize trainer with configuration, dataloaders, model, and device
        
        Args:
            config: Configuration dictionary from YAML
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (can be None)
            test_loader: Test DataLoader
            info: Dictionary with dataset information
            vision_model: Initialized vision model
            images_classifier: Initialized images classifier
            text_classifier: Initialized text classifier
            fusion_config: Fusion configuration dictionary
            device: Device to use for training (cuda/cpu)
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.info = info
        self.vision_model = vision_model
        self.images_classifier = images_classifier
        self.text_classifier = text_classifier
        self.device = device
        
        # Extract configuration sections
        self.data_config = config['data']
        self.training_config = config['training']
        self.output_config = config['output']
        self.checkpoint_config = config['checkpoint']
        
        # Extract fusion weights
        self.alpha_image = fusion_config.get('alpha_image', 0.5)
        self.alpha_text = 1.0 - self.alpha_image
        
        self.optimizer_vision, self.optimizer_images, self.optimizer_text = self._initialize_optimizers()
        self.scheduler_vision, self.scheduler_images, self.scheduler_text = self._initialize_schedulers()
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.start_epoch = 0
        
        self.n_warmup_epochs = self.training_config['n_warmup_epochs']
        self.vision_frozen = False
        self.accuracy_history = {}
    
    def _initialize_optimizers(self):
        """Initialize vision/images classifier/text classifier optimizers from optimizer config (required)."""
        optimizer_config = self.config.get('optimizer')

        opt_vision_cfg = optimizer_config.get('vision_model')
        opt_images_cfg = optimizer_config.get('images_classifier')
        opt_text_cfg = optimizer_config.get('text_classifier')

        vision_params = list(self.vision_model.parameters())
        images_classifier_params = list(self.images_classifier.parameters())
        text_classifier_params = list(self.text_classifier.parameters())

        ob_name = opt_vision_cfg['name'].lower()
        ob_lr = opt_vision_cfg['learning_rate']
        ob_wd = opt_vision_cfg['weight_decay']
        ob_betas = opt_vision_cfg.get('betas', (0.9, 0.999))
        ob_eps = opt_vision_cfg.get('eps', 1e-8)
        ob_momentum = opt_vision_cfg.get('momentum', 0.9)

        oi_name = opt_images_cfg['name'].lower()
        oi_lr = opt_images_cfg['learning_rate']
        oi_wd = opt_images_cfg['weight_decay']
        oi_betas = opt_images_cfg.get('betas', (0.9, 0.999))
        oi_eps = opt_images_cfg.get('eps', 1e-8)
        oi_momentum = opt_images_cfg.get('momentum', 0.9)

        ot_name = opt_text_cfg['name'].lower()
        ot_lr = opt_text_cfg['learning_rate']
        ot_wd = opt_text_cfg['weight_decay']
        ot_betas = opt_text_cfg.get('betas', (0.9, 0.999))
        ot_eps = opt_text_cfg.get('eps', 1e-8)
        ot_momentum = opt_text_cfg.get('momentum', 0.9)

        optimizer_vision = make_optimizer(ob_name, ob_lr, ob_wd, ob_betas, ob_eps, ob_momentum, vision_params)
        optimizer_images = make_optimizer(oi_name, oi_lr, oi_wd, oi_betas, oi_eps, oi_momentum, images_classifier_params)
        optimizer_text = make_optimizer(ot_name, ot_lr, ot_wd, ot_betas, ot_eps, ot_momentum, text_classifier_params)
        return optimizer_vision, optimizer_images, optimizer_text
    
    def _initialize_schedulers(self):
        """Initialize cosine annealing schedulers for all optimizers."""
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
            scheduler_images = CosineAnnealingLR(
                self.optimizer_images, 
                T_max=num_epochs,
                eta_min=scheduler_config.get('images_classifier_eta_min', 0.0)
            )
            scheduler_text = CosineAnnealingLR(
                self.optimizer_text, 
                T_max=num_epochs,
                eta_min=scheduler_config.get('images_classifier_eta_min', 0.0)  # Use same as images classifier
            )
        else:
            # No schedulers (step function that does nothing)
            scheduler_vision = None
            scheduler_images = None
            scheduler_text = None
            
        return scheduler_vision, scheduler_images, scheduler_text
    
    def train(self):
        """Main training loop"""
        num_epochs = self.training_config['num_epochs']
        patience = self.training_config['patience']
        validation_interval = max(1, int(self.training_config.get('validation_interval', 1)))
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Starting from epoch: {self.start_epoch}")
        vision_optimizer_cfg = self.config['optimizer']['vision_model']
        images_optimizer_cfg = self.config['optimizer']['images_classifier']
        text_optimizer_cfg = self.config['optimizer']['text_classifier']
        logger.info(f"Optimizers -> vision_model: {vision_optimizer_cfg['name']} (lr={vision_optimizer_cfg['learning_rate']}), images_classifier: {images_optimizer_cfg['name']} (lr={images_optimizer_cfg['learning_rate']}), text_classifier: {text_optimizer_cfg['name']} (lr={text_optimizer_cfg['learning_rate']})")
        logger.info(f"Fusion weights -> alpha_image: {self.alpha_image:.3f}, alpha_text: {self.alpha_text:.3f}")

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
                    'params/learning_rate_images_classifier': images_optimizer_cfg['learning_rate'],
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
                if self.scheduler_images is not None:
                    self.scheduler_images.step()
                if self.scheduler_text is not None:
                    self.scheduler_text.step()
                
            logger.info("Training completed!")
    
    def _train_epoch(self):
        """Train for one epoch"""
        self.vision_model.train()
        self.images_classifier.train()
        self.text_classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        accuracies = []
        # Running confusion matrix for macro-F1
        num_classes = self.info['num_classes']
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} - Training")
        stime = time.time()

        for batch_idx, (images, targets, text_embeddings) in enumerate(pbar):

            # Move data to device
            images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            # Convert text_embeddings to tensor if needed (DataLoader may return numpy arrays)
            if not isinstance(text_embeddings, torch.Tensor):
                text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
            text_embeddings = text_embeddings.to(self.device, non_blocking=True)
            
            # Zero gradients
            if not self.vision_frozen:
                self.optimizer_vision.zero_grad()
            self.optimizer_images.zero_grad()
            self.optimizer_text.zero_grad()
            
            # Forward pass - image branch
            vision_embeddings = self.vision_model(images)
            image_logits = self.images_classifier(vision_embeddings)
            
            # Forward pass - text branch
            text_logits = self.text_classifier(text_embeddings)
            
            # Fuse logits
            fused_logits = self.alpha_image * image_logits + self.alpha_text * text_logits
            
            # Compute loss on fused logits
            loss = self.images_classifier.loss(fused_logits, targets)
            
            # Backward pass
            loss.backward()
            if not self.vision_frozen:
                self.optimizer_vision.step()
            self.optimizer_images.step()
            self.optimizer_text.step()
            
            # Statistics
            total_loss += loss.item()
            pred = fused_logits.argmax(dim=1, keepdim=True)
            correct = pred.eq(targets.view_as(pred)).sum().item()
            total = targets.size(0)
            # Update confusion matrix
            pred_flat = pred.view(-1).detach().cpu().numpy()
            target_flat = targets.view(-1).detach().cpu().numpy()
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
        self.images_classifier.eval()
        self.text_classifier.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_classes = self.info['num_classes']
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for images, targets, text_embeddings in pbar:
                # Move data to device
                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32).to(self.device, non_blocking=True)
                
                # Forward pass - image branch
                vision_embeddings = self.vision_model(images)
                image_logits = self.images_classifier(vision_embeddings)
                
                # Forward pass - text branch
                text_logits = self.text_classifier(text_embeddings)
                
                # Fuse logits
                fused_logits = self.alpha_image * image_logits + self.alpha_text * text_logits
                
                # Compute loss on fused logits
                loss = self.images_classifier.loss(fused_logits, targets)
                
                # Statistics
                total_loss += loss.item()
                pred = fused_logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()
                total += targets.size(0)
                pred_flat = pred.view(-1).detach().cpu().numpy()
                target_flat = targets.view(-1).detach().cpu().numpy()
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
        self.images_classifier.eval()
        self.text_classifier.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        num_classes = self.info['num_classes']
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            for images, targets, text_embeddings in pbar:
                # Move data to device
                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32).to(self.device, non_blocking=True)
                
                # Forward pass - image branch
                vision_embeddings = self.vision_model(images)
                image_logits = self.images_classifier(vision_embeddings)
                
                # Forward pass - text branch
                text_logits = self.text_classifier(text_embeddings)
                
                # Fuse logits
                fused_logits = self.alpha_image * image_logits + self.alpha_text * text_logits
                
                # Compute loss on fused logits
                loss = self.images_classifier.loss(fused_logits, targets)
                
                # Statistics
                total_loss += loss.item()
                pred = fused_logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()
                total += targets.size(0)
                pred_flat = pred.view(-1).detach().cpu().numpy()
                target_flat = targets.view(-1).detach().cpu().numpy()
                for t, p in zip(target_flat, pred_flat):
                    confusion[t, p] += 1
                
                # Store predictions and targets for detailed analysis
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                
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

        # Include all optimizer states if using split optimizers
        if self.optimizer_vision is not None and self.optimizer_images is not None and self.optimizer_text is not None:
            additional_info['optimizers_state_dict'] = {
                'vision': self.optimizer_vision.state_dict(),
                'images_classifier': self.optimizer_images.state_dict(),
                'text_classifier': self.optimizer_text.state_dict(),
            }
        
        # Include scheduler states
        if self.scheduler_vision is not None or self.scheduler_images is not None or self.scheduler_text is not None:
            additional_info['schedulers_state_dict'] = {}
            if self.scheduler_vision is not None:
                additional_info['schedulers_state_dict']['vision'] = self.scheduler_vision.state_dict()
            if self.scheduler_images is not None:
                additional_info['schedulers_state_dict']['images_classifier'] = self.scheduler_images.state_dict()
            if self.scheduler_text is not None:
                additional_info['schedulers_state_dict']['text_classifier'] = self.scheduler_text.state_dict()
        
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
            images_classifier=self.images_classifier,
            text_classifier=self.text_classifier
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
