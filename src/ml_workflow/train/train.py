import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

try:
    from ..utils import logger, save_checkpoint, cleanup_old_checkpoints
except ImportError:
    from ml_workflow.utils import logger, save_checkpoint, cleanup_old_checkpoints
import wandb
import matplotlib.pyplot as plt


class Trainer:

    def __init__(
        self,
        config,
        train_loader,
        val_loader,
        test_loader,
        info,
        vision_model,
        multimodal_classifier,
        device,
        config_path=None,
        config_filename=None,
    ):
        """
        Initialize trainer with configuration, dataloaders, model, and device

        Args:
            config: Configuration dictionary from YAML
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (can be None)
            test_loader: Test DataLoader
            info: Dictionary with dataset information
            vision_model: Initialized vision model
            multimodal_classifier: Initialized multimodal classifier
            device: Device to use for training (cuda/cpu)
            config_path: Path to the config file (optional)
            config_filename: Filename of the config (without path/extension) for wandb naming (optional)
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.info = info
        self.vision_model = vision_model
        self.multimodal_classifier = multimodal_classifier
        self.device = device
        self.config_path = config_path
        self.config_filename = config_filename

        # Extract configuration sections
        self.data_config = config["data"]
        self.training_config = config["training"]
        self.output_config = config["output"]
        self.checkpoint_config = config["checkpoint"]
        self.masking_config = config.get("masking", {})

        # Initialize masking settings
        self.mask_complete = self.masking_config.get("mask_complete", None)
        self.random_mask_config = self.masking_config.get("random_mask", {})
        self.random_mask_enabled = self.random_mask_config.get("enabled", False) and self.mask_complete is None
        self.image_mask_prob = self.random_mask_config.get("image_prob", 0.0) if self.random_mask_enabled else 0.0
        self.text_mask_prob = self.random_mask_config.get("text_prob", 0.0) if self.random_mask_enabled else 0.0

        # Initialize epoch-based masking schedule
        self.masking_schedule_config = self.masking_config.get("epoch_schedule", {})
        self.masking_schedule_enabled = self.masking_schedule_config.get("enabled", False)
        self.vision_mask_schedule = self.masking_schedule_config.get("vision", {"mask_epochs": []})
        self.text_mask_schedule = self.masking_schedule_config.get("text", {"mask_epochs": []})

        # Initialize vision-only pre-training configuration
        self.vision_only_config = self.training_config.get("vision_only_pretraining", {})
        self.vision_only_enabled = self.vision_only_config.get("enabled", False)
        self.vision_only_epochs = self.vision_only_config.get("epochs", 0)

        # Vision-only classifier and optimizer (initialized later)
        self.vision_only_classifier = None
        self.optimizer_vision_only = None
        self.vision_only_criterion = None

        # Initialize main optimizers and schedulers
        self.optimizer_vision, self.optimizer_multimodal = self._initialize_optimizers()
        self.scheduler_vision, self.scheduler_multimodal = self._initialize_schedulers()

        self.current_epoch = 0
        self.start_epoch = 0

        # Separate early stopping tracking for vision-only and multimodal phases
        self.best_val_loss_vision_only = float("inf")
        self.best_val_loss_multimodal = float("inf")
        self.patience_counter_vision_only = 0
        self.patience_counter_multimodal = 0

        # For backward compatibility (used in checkpointing)
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        self.n_warmup_epochs = self.training_config["n_warmup_epochs"]
        self.vision_frozen = False
        self.accuracy_history = {}

        # Initialize auxiliary loss scheduling
        self.aux_loss_schedule_config = self.training_config.get("auxiliary_loss_schedule", {})

        # Vision auxiliary loss schedule
        self.vision_aux_schedule = self.aux_loss_schedule_config.get(
            "vision",
            {
                "schedule_type": "constant",
                "start_epoch": 0,
                "end_epoch": 100,
                "start_weight": self.config["multimodal_classifier"].get("auxiliary_vision_loss_weight", 0.3),
                "end_weight": self.config["multimodal_classifier"].get("auxiliary_vision_loss_weight", 0.3),
            },
        )

        # Text auxiliary loss schedule
        self.text_aux_schedule = self.aux_loss_schedule_config.get(
            "text",
            {
                "schedule_type": "constant",
                "start_epoch": 0,
                "end_epoch": 100,
                "start_weight": self.config["multimodal_classifier"].get("auxiliary_text_loss_weight", 0.3),
                "end_weight": self.config["multimodal_classifier"].get("auxiliary_text_loss_weight", 0.3),
            },
        )

    @staticmethod
    def _make_optimizer(name_cfg, lr, wd, betas, eps, momentum, params):
        """Create optimizer based on configuration"""
        if name_cfg == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
        if name_cfg == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
        if name_cfg == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
        raise ValueError(f"Unsupported optimizer: {name_cfg}")

    def _initialize_optimizers(self):
        """Initialize vision and multimodal classifier optimizers from optimizer config (required)."""
        optimizer_config = self.config.get("optimizer")

        opt_vision_cfg = optimizer_config.get("vision_model")
        opt_multimodal_cfg = optimizer_config.get("multimodal_classifier")

        vision_params = list(self.vision_model.parameters())
        multimodal_classifier_params = list(self.multimodal_classifier.parameters())

        optimizer_vision_name = opt_vision_cfg["name"].lower()
        optimizer_vision_lr = opt_vision_cfg["learning_rate"]
        optimizer_vision_wd = opt_vision_cfg["weight_decay"]
        optimizer_vision_betas = opt_vision_cfg.get("betas", (0.9, 0.999))
        optimizer_vision_eps = opt_vision_cfg.get("eps", 1e-8)
        optimizer_vision_momentum = opt_vision_cfg.get("momentum", 0.9)

        optimizer_multimodal_name = opt_multimodal_cfg["name"].lower()
        optimizer_multimodal_lr = opt_multimodal_cfg["learning_rate"]
        optimizer_multimodal_wd = opt_multimodal_cfg["weight_decay"]
        optimizer_multimodal_betas = opt_multimodal_cfg.get("betas", (0.9, 0.999))
        optimizer_multimodal_eps = opt_multimodal_cfg.get("eps", 1e-8)
        optimizer_multimodal_momentum = opt_multimodal_cfg.get("momentum", 0.9)

        optimizer_vision = self._make_optimizer(
            optimizer_vision_name,
            optimizer_vision_lr,
            optimizer_vision_wd,
            optimizer_vision_betas,
            optimizer_vision_eps,
            optimizer_vision_momentum,
            vision_params,
        )
        optimizer_multimodal = self._make_optimizer(
            optimizer_multimodal_name,
            optimizer_multimodal_lr,
            optimizer_multimodal_wd,
            optimizer_multimodal_betas,
            optimizer_multimodal_eps,
            optimizer_multimodal_momentum,
            multimodal_classifier_params,
        )
        return optimizer_vision, optimizer_multimodal

    def _initialize_schedulers(self):
        """Initialize cosine annealing schedulers for all optimizers."""
        num_epochs = self.training_config["num_epochs"]

        # Get scheduler configuration
        scheduler_config = self.training_config.get("scheduler", {})
        use_cosine_annealing = scheduler_config.get("use_cosine_annealing", True)

        if use_cosine_annealing:
            # Cosine annealing schedulers
            scheduler_vision = CosineAnnealingLR(
                self.optimizer_vision, T_max=num_epochs, eta_min=scheduler_config.get("vision_eta_min", 0.0)
            )
            scheduler_multimodal = CosineAnnealingLR(
                self.optimizer_multimodal,
                T_max=num_epochs,
                eta_min=scheduler_config.get("multimodal_classifier_eta_min", 0.0),
            )
        else:
            # No schedulers
            scheduler_vision = None
            scheduler_multimodal = None

        return scheduler_vision, scheduler_multimodal

    def _initialize_vision_only_classifier(self):
        """
        Initialize vision pathway for pre-training by reusing multimodal components
        This is more efficient than creating a separate classifier
        """
        if not self.vision_only_enabled or self.vision_only_epochs == 0:
            logger.info("Vision-only pre-training disabled")
            return

        # SAFETY CHECK: Vision-only pre-training requires auxiliary loss to be enabled
        if not self.multimodal_classifier.use_auxiliary_loss:
            raise ValueError(
                "Vision-only pre-training requires use_auxiliary_loss=True in multimodal_classifier config. "
                "The vision auxiliary classifier is needed for vision-only classification."
            )

        # We'll reuse the vision projection + aux classifier from multimodal_classifier
        # No need to create separate components!
        self.vision_only_classifier = None  # Set to None to signal we're reusing
        self.optimizer_vision_only = None  # Will use optimizer_multimodal for projection
        self.vision_only_criterion = None  # Will use same loss as multimodal

        logger.info("Vision-only pre-training will use Vision Projection + Aux Classifier")
        logger.info("  This allows the vision pathway to be pre-trained before multimodal fusion")
        logger.info("  Vision Projection parameters: ~329k")
        logger.info("  Vision Aux Classifier parameters: ~15k")

    def _compute_auxiliary_weight(self, schedule_config: dict, current_epoch: int) -> float:
        """
        Compute auxiliary loss weight based on schedule configuration

        Args:
            schedule_config: Dict with schedule parameters
            current_epoch: Current training epoch

        Returns:
            Computed weight for current epoch
        """
        start_weight = schedule_config.get("start_weight", 0.3)
        end_weight = schedule_config.get("end_weight", 0.3)
        start_epoch = schedule_config.get("start_epoch", 0)
        end_epoch = schedule_config.get("end_epoch", 100)
        schedule_type = schedule_config.get("schedule_type", "constant")

        # Before start_epoch, return 0.0 (disabled)
        if current_epoch < start_epoch:
            return 0.0

        # After end_epoch, use end_weight
        if current_epoch >= end_epoch:
            return end_weight

        # If constant schedule, return start_weight
        if schedule_type == "constant":
            return start_weight

        # Compute progress between start and end epochs
        progress = (current_epoch - start_epoch) / (end_epoch - start_epoch)

        # Apply schedule
        if schedule_type == "linear":
            weight = start_weight + (end_weight - start_weight) * progress

        elif schedule_type == "exponential":
            import math

            if start_weight > 0:
                weight = start_weight * ((end_weight / start_weight) ** progress)
            else:
                weight = end_weight * progress

        elif schedule_type == "cosine":
            import math

            weight = start_weight + (end_weight - start_weight) * (1 - math.cos(progress * math.pi)) / 2

        elif schedule_type == "step":
            weight = start_weight if progress < 0.5 else end_weight

        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        return weight

    def _get_auxiliary_weights(self, epoch: int) -> tuple:
        """Get the current auxiliary loss weights for both modalities"""
        vision_weight = self._compute_auxiliary_weight(self.vision_aux_schedule, epoch)
        text_weight = self._compute_auxiliary_weight(self.text_aux_schedule, epoch)
        return vision_weight, text_weight

    def _should_mask_modality(self, modality: str, epoch: int) -> bool:
        """Check if a modality should be completely masked in the current epoch"""
        if not self.masking_schedule_enabled:
            return False

        if modality == "vision":
            mask_ranges = self.vision_mask_schedule.get("mask_epochs", [])
        elif modality == "text":
            mask_ranges = self.text_mask_schedule.get("mask_epochs", [])
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Check if current epoch falls within any mask range
        for start_epoch, end_epoch in mask_ranges:
            if start_epoch <= epoch < end_epoch:
                return True

        return False

    def _is_vision_only_epoch(self, epoch: int) -> bool:
        """Check if current epoch is in vision-only pre-training phase"""
        if not self.vision_only_enabled:
            return False
        return epoch < self.vision_only_epochs

    def train(self):
        """Main training loop"""
        num_epochs = self.training_config["num_epochs"]
        patience = self.training_config["patience"]
        validation_interval = max(1, int(self.training_config.get("validation_interval", 1)))

        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Starting from epoch: {self.start_epoch}")

        # Log optimizer configuration
        vision_optimizer_cfg = self.config["optimizer"]["vision_model"]
        multimodal_optimizer_cfg = self.config["optimizer"]["multimodal_classifier"]
        logger.info(
            f"Optimizers -> vision_model: {vision_optimizer_cfg['name']} "
            f"(lr={vision_optimizer_cfg['learning_rate']}), "
            f"multimodal_classifier: {multimodal_optimizer_cfg['name']} "
            f"(lr={multimodal_optimizer_cfg['learning_rate']})"
        )

        # Log fusion strategy and parameters
        multimodal_info = self.multimodal_classifier.get_model_info()
        logger.info(f"Fusion strategy: {multimodal_info['fusion_strategy']}")
        if "fusion_weights" in multimodal_info:
            logger.info(
                f"Fusion weights -> alpha_vision: "
                f"{multimodal_info['fusion_weights']['alpha_vision']:.3f}, "
                f"alpha_text: {multimodal_info['fusion_weights']['alpha_text']:.3f}"
            )

        # Log vision-only pre-training
        if self.vision_only_enabled:
            logger.info("Vision-Only Pre-training:")
            logger.info(f"  Epochs: 0-{self.vision_only_epochs-1} (vision-only)")
            logger.info(f"  Epochs: {self.vision_only_epochs}+ (multimodal)")

        # Log auxiliary loss schedules
        logger.info("Auxiliary Loss Schedules:")
        logger.info(f"  Vision: {self.vision_aux_schedule['schedule_type']}")
        logger.info(
            f"    Weight: {self.vision_aux_schedule['start_weight']:.3f} → {self.vision_aux_schedule['end_weight']:.3f}"
        )
        logger.info(f"    Epochs: {self.vision_aux_schedule['start_epoch']} → {self.vision_aux_schedule['end_epoch']}")
        logger.info(f"  Text: {self.text_aux_schedule['schedule_type']}")
        logger.info(
            f"    Weight: {self.text_aux_schedule['start_weight']:.3f} → {self.text_aux_schedule['end_weight']:.3f}"
        )
        logger.info(f"    Epochs: {self.text_aux_schedule['start_epoch']} → {self.text_aux_schedule['end_epoch']}")

        # Log masking schedule (if not using vision-only pretraining)
        if not self.vision_only_enabled:
            if self.masking_schedule_enabled:
                logger.info("Epoch-Based Masking Schedule:")
                vision_mask_epochs = self.vision_mask_schedule.get("mask_epochs", [])
                text_mask_epochs = self.text_mask_schedule.get("mask_epochs", [])
                if vision_mask_epochs:
                    logger.info(f"  Vision masked during epochs: {vision_mask_epochs}")
                if text_mask_epochs:
                    logger.info(f"  Text masked during epochs: {text_mask_epochs}")
            elif self.mask_complete:
                logger.info(f"Complete masking enabled: {self.mask_complete}")
            elif self.random_mask_enabled:
                logger.info(
                    f"Random masking enabled - image_prob: {self.image_mask_prob}, text_prob: {self.text_mask_prob}"
                )

        # Use config filename for wandb run name if available
        run_name = self.config_filename if self.config_filename else self.output_config["experiment_name"]
        with wandb.init(project=self.output_config["wandb_project"], name=run_name, config=self.config) as run:

            for epoch in range(self.start_epoch, num_epochs):
                self.current_epoch = epoch

                # Check if in vision-only pre-training phase
                is_vision_only = self._is_vision_only_epoch(epoch)

                if is_vision_only:
                    # Vision-only pre-training mode
                    training_stage = "vision_only_pretraining"
                    logger.info(f"\nEpoch {epoch+1}/{num_epochs} - Stage: {training_stage}")
                    logger.info("  Mode: Training ONLY vision encoder (no multimodal classifier)")

                    vision_aux_weight = 0.0
                    text_aux_weight = 0.0
                    vision_masked = False
                    text_masked = False
                else:
                    # Multimodal training mode
                    # Update auxiliary loss weights for current epoch
                    vision_aux_weight, text_aux_weight = self._get_auxiliary_weights(epoch)
                    self.multimodal_classifier.auxiliary_vision_loss_weight = vision_aux_weight
                    self.multimodal_classifier.auxiliary_text_loss_weight = text_aux_weight

                    # Check masking state for current epoch
                    vision_masked = (
                        self._should_mask_modality("vision", epoch) if self.masking_schedule_enabled else False
                    )
                    text_masked = self._should_mask_modality("text", epoch) if self.masking_schedule_enabled else False

                    # Determine training stage
                    if text_masked or (text_aux_weight == 0.0 and vision_aux_weight > 0.0):
                        training_stage = "vision_only"
                    elif vision_masked or (vision_aux_weight == 0.0 and text_aux_weight > 0.0):
                        training_stage = "text_only"
                    elif vision_aux_weight > 0.0 and text_aux_weight > 0.0 and not vision_masked and not text_masked:
                        training_stage = "multimodal"
                    else:
                        training_stage = "fusion_only"

                    logger.info(f"\nEpoch {epoch+1}/{num_epochs} - Stage: {training_stage}")
                    logger.info(f"  Vision aux weight: {vision_aux_weight:.4f} | Masked: {vision_masked}")
                    logger.info(f"  Text aux weight: {text_aux_weight:.4f} | Masked: {text_masked}")

                # Update vision freeze state
                if epoch < self.n_warmup_epochs:
                    self.vision_model._freeze_layers(warmup_mode=True)
                    self.vision_frozen = True
                elif epoch == self.n_warmup_epochs:
                    self.vision_model._freeze_layers(warmup_mode=False)
                    self.vision_frozen = False

                # Training phase
                if is_vision_only:
                    train_loss, train_acc, train_f1 = self._train_epoch_vision_only()
                else:
                    train_loss, train_acc, train_f1 = self._train_epoch()

                # Get current learning rates
                current_lr_vision = self.optimizer_vision.param_groups[0]["lr"]
                current_lr_multimodal = self.optimizer_multimodal.param_groups[0]["lr"]

                # Prepare logging dict
                log_dict = {
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "train/f1": train_f1,
                    "epoch": epoch,
                    "params/learning_rate_vision": current_lr_vision,
                    "params/training_stage": training_stage,
                    "params/is_vision_only_pretraining": is_vision_only,
                }

                # Add multimodal-specific metrics if not in vision-only mode
                if not is_vision_only:
                    log_dict.update(
                        {
                            "params/learning_rate_multimodal_classifier": current_lr_multimodal,
                            "params/aux_vision_weight": vision_aux_weight,
                            "params/aux_text_weight": text_aux_weight,
                            "params/vision_masked": vision_masked,
                            "params/text_masked": text_masked,
                        }
                    )

                run.log(log_dict)

                ran_validation = False
                val_loss = None
                val_acc = None
                val_f1 = None

                # Validation phase
                if self.val_loader is not None and ((epoch + 1) % validation_interval == 0):
                    ran_validation = True
                    if is_vision_only:
                        val_loss, val_acc, val_f1, confusion_matrix = self.validate_vision_only()
                    else:
                        val_loss, val_acc, val_f1, confusion_matrix = self.validate()

                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
                        f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
                    )

                    # Create and log confusion analysis table (only in multimodal mode)
                    if confusion_matrix is not None and not is_vision_only:
                        class_names = self.info["classes"]
                        confusion_table, accuracy_history_table = self._make_wandb_tables(confusion_matrix, class_names)
                        run.log({"val/confusion_analysis": confusion_table})
                        run.log({"val/accuracy_history": accuracy_history_table})

                    # Early stopping (separate tracking for vision-only vs multimodal)
                    if is_vision_only:
                        # Vision-only phase early stopping
                        if val_loss < self.best_val_loss_vision_only:
                            self.best_val_loss_vision_only = val_loss
                            self.patience_counter_vision_only = 0
                            self.best_val_loss = val_loss  # For checkpointing
                            self._save_checkpoint(epoch, val_loss, is_best=True)
                            logger.info(f"  New best vision-only val loss: {val_loss:.4f}")
                        else:
                            self.patience_counter_vision_only += 1
                            logger.info(f"  Vision-only patience: {self.patience_counter_vision_only}/{patience}")

                        if self.patience_counter_vision_only >= patience:
                            logger.info(f"Early stopping in vision-only phase at epoch {epoch+1}")
                            break
                    else:
                        # Multimodal phase early stopping
                        if val_loss < self.best_val_loss_multimodal:
                            self.best_val_loss_multimodal = val_loss
                            self.patience_counter_multimodal = 0
                            self.best_val_loss = val_loss  # For checkpointing
                            self._save_checkpoint(epoch, val_loss, is_best=True)
                            logger.info(f"  New best multimodal val loss: {val_loss:.4f}")
                        else:
                            self.patience_counter_multimodal += 1
                            logger.info(f"  Multimodal patience: {self.patience_counter_multimodal}/{patience}")

                        if self.patience_counter_multimodal >= patience:
                            logger.info(f"Early stopping in multimodal phase at epoch {epoch+1}")
                            break

                # If validation did not run, still log training metrics
                if not ran_validation:
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
                        f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}"
                    )

                run.log(
                    {
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "val/f1": val_f1,
                        "epoch": epoch,
                    }
                )

                # Save regular checkpoint
                save_frequency = self.checkpoint_config.get("save_frequency", 10)
                if (epoch + 1) % save_frequency == 0:
                    self._save_checkpoint(epoch, train_loss, is_best=False)

                # Step schedulers (after warmup period)
                if epoch >= self.n_warmup_epochs:
                    if self.scheduler_vision is not None:
                        self.scheduler_vision.step()
                    if self.scheduler_multimodal is not None:
                        self.scheduler_multimodal.step()

            logger.info("Training completed!")

    def _apply_masking(self, images: torch.Tensor, text_embeddings: torch.Tensor, epoch: int = None):
        """Apply modality masking to images and/or text embeddings (zero masking)"""
        batch_size = images.size(0)
        vision_valid = torch.ones(batch_size, dtype=torch.bool, device=images.device)
        text_valid = torch.ones(batch_size, dtype=torch.bool, device=text_embeddings.device)
        mask_applied = False

        # PRIORITY 1: Epoch-based masking schedule
        if epoch is not None and self.masking_schedule_enabled:
            should_mask_vision = self._should_mask_modality("vision", epoch)
            should_mask_text = self._should_mask_modality("text", epoch)

            if should_mask_vision:
                images = torch.zeros_like(images)
                vision_valid.zero_()
                mask_applied = True

            if should_mask_text:
                text_embeddings = torch.zeros_like(text_embeddings)
                text_valid.zero_()
                mask_applied = True

            if should_mask_vision or should_mask_text:
                mask_info = {"vision_valid_mask": vision_valid, "text_valid_mask": text_valid} if mask_applied else None
                return images, text_embeddings, mask_info

        # PRIORITY 2: Complete masking
        if self.mask_complete == "image":
            images = torch.zeros_like(images)
            vision_valid.zero_()
            mask_applied = True
        elif self.mask_complete == "text":
            text_embeddings = torch.zeros_like(text_embeddings)
            text_valid.zero_()
            mask_applied = True

        # PRIORITY 3: Random masking
        elif self.random_mask_enabled:
            images = images.clone()
            text_embeddings = text_embeddings.clone()

            image_mask = (
                torch.rand(batch_size, device=images.device) < self.image_mask_prob
                if self.image_mask_prob > 0.0
                else torch.zeros(batch_size, dtype=torch.bool, device=images.device)
            )

            text_mask = (
                torch.rand(batch_size, device=text_embeddings.device) < self.text_mask_prob
                if self.text_mask_prob > 0.0
                else torch.zeros(batch_size, dtype=torch.bool, device=text_embeddings.device)
            )

            both_masked = image_mask & text_mask

            if both_masked.any():
                p_keep_image = 1 - self.image_mask_prob
                p_keep_text = 1 - self.text_mask_prob
                p_keep_image = p_keep_image / (p_keep_image + p_keep_text)
                keep_image = torch.rand(both_masked.sum().item(), device=images.device) < p_keep_image

                image_mask[both_masked] = ~keep_image
                text_mask[both_masked] = keep_image

            if image_mask.any():
                images[image_mask] = 0.0
            vision_valid = ~image_mask

            if text_mask.any():
                text_embeddings[text_mask] = 0.0
            text_valid = ~text_mask

            mask_applied = True

        mask_info = None
        if mask_applied:
            mask_info = {"vision_valid_mask": vision_valid, "text_valid_mask": text_valid}

        return images, text_embeddings, mask_info

    def _train_epoch_vision_only(self):
        """
        Train for one epoch using vision pathway of multimodal classifier
        This reuses Vision Projection + Vision Aux Classifier
        """
        self.vision_model.train()
        self.multimodal_classifier.train()  # Only vision pathway will be used

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} - Vision-Only Pre-training")

        for batch_idx, (images, targets, text_embeddings, text_content_flags) in enumerate(pbar):
            # Move data to device (ignore text_embeddings)
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Zero gradients for vision pathway only
            if not self.vision_frozen:
                self.optimizer_vision.zero_grad(set_to_none=True)
            # Use multimodal optimizer for vision projection + aux classifier
            self.optimizer_multimodal.zero_grad(set_to_none=True)

            # Forward pass: vision model → projection → aux classifier
            vision_embeddings = self.vision_model(images)
            projected_vision = self.multimodal_classifier.vision_projection(vision_embeddings)
            logits = self.multimodal_classifier.aux_vision_classifier(projected_vision)

            # Compute loss
            loss_fn = self.multimodal_classifier.criterion
            loss = loss_fn(logits, targets)

            # Backward pass
            loss.backward()
            if not self.vision_frozen:
                self.optimizer_vision.step()
            self.optimizer_multimodal.step()

            # Statistics
            total_loss += loss.detach().item()
            pred = logits.argmax(dim=1, keepdim=True)
            batch_correct = pred.eq(targets.view_as(pred)).sum().item()
            batch_total = targets.size(0)
            correct += batch_correct
            total += batch_total

            # Accumulate predictions
            all_preds.append(pred.view(-1).detach())
            all_targets.append(targets.view(-1).detach())

            avg_running_loss = total_loss / (batch_idx + 1)
            running_accuracy = 100.0 * correct / total

            # Update progress bar
            pbar.set_postfix({"Loss": f"{avg_running_loss:.4f}", "Acc": f"{running_accuracy:.2f}%"})

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        # Compute macro-F1 (same as before)
        all_preds_tensor = torch.cat(all_preds)
        all_targets_tensor = torch.cat(all_targets)
        pred_np = all_preds_tensor.cpu().numpy()
        target_np = all_targets_tensor.cpu().numpy()

        num_classes = self.info["num_classes"]
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(target_np, pred_np):
            confusion[t, p] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            tp = np.diag(confusion).astype(np.float64)
            fp = confusion.sum(axis=0) - tp
            fn = confusion.sum(axis=1) - tp
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f1_per_class = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
            macro_f1 = float(np.mean(f1_per_class))

        return avg_loss, accuracy, macro_f1

    def _train_epoch(self):
        """Train for one epoch"""
        self.vision_model.train()
        self.multimodal_classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        # Check if we need zero detection (optimization: skip if text aux is disabled)
        text_aux_enabled = self.multimodal_classifier.auxiliary_text_loss_weight > 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} - Training")

        for batch_idx, (images, targets, text_embeddings, text_content_flags) in enumerate(pbar):  # Unpack 4 items

            # Move data to device
            images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            if not isinstance(text_embeddings, torch.Tensor):
                text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
            text_embeddings = text_embeddings.to(self.device, non_blocking=True)
            text_content_flags = text_content_flags.to(self.device, non_blocking=True)  # Boolean tensor

            # Apply masking (if enabled)
            images, text_embeddings, modality_mask_info = self._apply_masking(
                images, text_embeddings, self.current_epoch
            )

            # Use pre-computed flag (only if text aux is enabled)
            if text_aux_enabled:
                if modality_mask_info is None:
                    modality_mask_info = {"text_valid_mask": text_content_flags}
                elif "text_valid_mask" in modality_mask_info:
                    # Combine with masking schedule
                    modality_mask_info["text_valid_mask"] &= text_content_flags
                else:
                    modality_mask_info["text_valid_mask"] = text_content_flags

            # Zero gradients
            if not self.vision_frozen:
                self.optimizer_vision.zero_grad(set_to_none=True)
            self.optimizer_multimodal.zero_grad(set_to_none=True)

            # Forward pass through vision model
            vision_embeddings = self.vision_model(images)

            # Forward pass through multimodal classifier
            outputs = self.multimodal_classifier(vision_embeddings, text_embeddings, modality_mask_info)
            logits = outputs["logits"]

            # Compute loss (includes auxiliary losses if enabled)
            loss = self.multimodal_classifier.compute_loss(outputs, targets, modality_mask_info)

            # Clear references to intermediate outputs to free memory
            del vision_embeddings

            # Backward pass
            loss.backward()
            if not self.vision_frozen:
                self.optimizer_vision.step()
            self.optimizer_multimodal.step()

            # Statistics (detach to avoid keeping computation graph)
            total_loss += loss.detach().item()
            pred = logits.argmax(dim=1, keepdim=True)
            batch_correct = pred.eq(targets.view_as(pred)).sum().item()
            batch_total = targets.size(0)
            correct += batch_correct
            total += batch_total

            # Accumulate predictions on GPU (non-blocking, no CPU transfer)
            all_preds.append(pred.view(-1).detach())
            all_targets.append(targets.view(-1).detach())

            avg_running_loss = total_loss / (batch_idx + 1)
            running_accuracy = 100.0 * correct / total

            # Update progress bar
            pbar.set_postfix({"Loss": f"{avg_running_loss:.4f}", "Acc": f"{running_accuracy:.2f}%"})

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        # Compute macro-F1 from accumulated predictions (single CPU transfer at end)
        all_preds_tensor = torch.cat(all_preds)
        all_targets_tensor = torch.cat(all_targets)

        # Transfer to CPU only once at end of epoch
        pred_np = all_preds_tensor.cpu().numpy()
        target_np = all_targets_tensor.cpu().numpy()

        # Build confusion matrix
        num_classes = self.info["num_classes"]
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(target_np, pred_np):
            confusion[t, p] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
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
        self.multimodal_classifier.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_classes = self.info["num_classes"]
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

        # Check if we need zero detection
        text_aux_enabled = self.multimodal_classifier.auxiliary_text_loss_weight > 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for images, targets, text_embeddings, text_content_flags in pbar:
                # Move data to device
                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                if not isinstance(text_embeddings, torch.Tensor):
                    text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
                text_embeddings = text_embeddings.to(self.device, non_blocking=True)
                text_content_flags = text_content_flags.to(self.device, non_blocking=True)

                # Apply masking (epoch-based or complete) - no random masking in validation
                images, text_embeddings, modality_mask_info = self._apply_masking(
                    images, text_embeddings, self.current_epoch
                )

                # Use pre-computed flag (only if needed)
                if text_aux_enabled:
                    if modality_mask_info is None:
                        modality_mask_info = {"text_valid_mask": text_content_flags}
                    elif "text_valid_mask" in modality_mask_info:
                        modality_mask_info["text_valid_mask"] &= text_content_flags
                    else:
                        modality_mask_info["text_valid_mask"] = text_content_flags

                # Forward pass through vision model
                vision_embeddings = self.vision_model(images)

                # Forward pass through multimodal classifier
                outputs = self.multimodal_classifier(vision_embeddings, text_embeddings, modality_mask_info)
                logits = outputs["logits"]

                # Compute loss
                loss = self.multimodal_classifier.compute_loss(outputs, targets, modality_mask_info)

                # Statistics
                total_loss += loss.item()
                pred = logits.argmax(dim=1, keepdim=True)
                batch_correct = pred.eq(targets.view_as(pred)).sum().item()
                batch_total = targets.size(0)
                correct += batch_correct
                total += batch_total
                pred_flat = pred.view(-1).detach().cpu().numpy()
                target_flat = targets.view(-1).detach().cpu().numpy()
                for t, p in zip(target_flat, pred_flat):
                    confusion[t, p] += 1

                # Update progress bar
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100. * correct / total:.2f}%"})

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        with np.errstate(divide="ignore", invalid="ignore"):
            tp = np.diag(confusion).astype(np.float64)
            fp = confusion.sum(axis=0) - tp
            fn = confusion.sum(axis=1) - tp
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f1_per_class = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
            macro_f1 = float(np.mean(f1_per_class))

        return avg_loss, accuracy, macro_f1, confusion

    def validate_vision_only(self):
        """Validation loop for vision-only pre-training using reused components"""
        if self.val_loader is None:
            logger.warning("No validation loader available")
            return None, None, None, None

        self.vision_model.eval()
        self.multimodal_classifier.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        num_classes = self.info["num_classes"]
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation (Vision-Only)", leave=False)
            for images, targets, text_embeddings, text_content_flags in pbar:
                # Move data to device (ignore text_embeddings)
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Forward pass: vision → projection → aux classifier
                vision_embeddings = self.vision_model(images)
                projected_vision = self.multimodal_classifier.vision_projection(vision_embeddings)
                logits = self.multimodal_classifier.aux_vision_classifier(projected_vision)

                # Compute loss
                loss = self.multimodal_classifier.criterion(logits, targets)

                # Statistics
                total_loss += loss.item()
                pred = logits.argmax(dim=1, keepdim=True)
                batch_correct = pred.eq(targets.view_as(pred)).sum().item()
                batch_total = targets.size(0)
                correct += batch_correct
                total += batch_total

                pred_flat = pred.view(-1).detach().cpu().numpy()
                target_flat = targets.view(-1).detach().cpu().numpy()
                for t, p in zip(target_flat, pred_flat):
                    confusion[t, p] += 1

                # Update progress bar
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100. * correct / total:.2f}%"})

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        with np.errstate(divide="ignore", invalid="ignore"):
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
                    incorrect_predictions.append(
                        {
                            "true_class": class_names[true_class],
                            "count": int(confusion_matrix[true_class, predicted_class]),
                        }
                    )

            if predicted_class not in self.accuracy_history:
                self.accuracy_history[predicted_class] = []
            # Calculate accuracy, handling division by zero (when class has no predictions)
            class_sum = confusion_matrix[predicted_class].sum()
            if class_sum > 0:
                accuracy = confusion_matrix[predicted_class, predicted_class] / class_sum
            else:
                accuracy = 0.0
            self.accuracy_history[predicted_class].append(accuracy)

            # Sort by count (most common mistakes first)
            incorrect_predictions.sort(key=lambda x: x["count"], reverse=True)

            # Create distribution plot
            if incorrect_predictions:
                fig, ax = plt.subplots(figsize=(8, 4))
                true_classes = [item["true_class"] for item in incorrect_predictions]
                counts = [item["count"] for item in incorrect_predictions]

                # Create bar plot
                bars = ax.bar(range(len(true_classes)), counts)
                ax.set_xlabel("True Class")
                ax.set_ylabel("Count")
                ax.set_title(f"Incorrectly Classified as {class_names[predicted_class]}")
                ax.set_xticks(range(len(true_classes)))
                ax.set_xticklabels(true_classes, rotation=45, ha="right")

                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(count), ha="center", va="bottom"
                    )

                plt.tight_layout()

                # Convert plot to wandb image
                plot_image = wandb.Image(fig)
                plt.close(fig)
            else:
                # No incorrect predictions for this class
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.text(
                    0.5, 0.5, "No incorrect predictions", ha="center", va="center", transform=ax.transAxes, fontsize=12
                )
                ax.set_title(f"Incorrectly Classified as {class_names[predicted_class]}")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")

                plot_image = wandb.Image(fig)
                plt.close(fig)

            # Add row to table
            table_data.append([class_names[predicted_class], plot_image])

            # Create accuracy history plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(len(self.accuracy_history[predicted_class])), self.accuracy_history[predicted_class])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Accuracy History for {class_names[predicted_class]}")
            ax.set_xticks(range(len(self.accuracy_history[predicted_class])))
            ax.set_xticklabels(range(len(self.accuracy_history[predicted_class])), ha="right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            accuracy_history_image = wandb.Image(fig)
            plt.close(fig)

            accuracy_history_data.append([class_names[predicted_class], accuracy_history_image])

        # Create wandb table
        table = wandb.Table(columns=["Predicted Class", "Distribution of Incorrect Classifications"], data=table_data)
        accuracy_history_table = wandb.Table(columns=["Class", "Accuracy History"], data=accuracy_history_data)

        return table, accuracy_history_table

    def test(self):
        """Test loop"""
        # Handle case where test_loader is None
        if self.test_loader is None:
            logger.warning("No test loader available (test_size=0.0). Skipping test evaluation.")
            return {
                "test_loss": None,
                "test_accuracy": None,
                "correct": 0,
                "total": 0,
                "predictions": [],
                "targets": [],
                "macro_f1": None,
            }

        logger.info("Starting test evaluation...")

        self.vision_model.eval()
        self.multimodal_classifier.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        num_classes = self.info["num_classes"]
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

        # Check if we need zero detection
        text_aux_enabled = self.multimodal_classifier.auxiliary_text_loss_weight > 0

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            # Unpack 4 items now
            for images, targets, text_embeddings, text_content_flags in pbar:
                # Move data to device
                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                if not isinstance(text_embeddings, torch.Tensor):
                    text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
                text_embeddings = text_embeddings.to(self.device, non_blocking=True)
                text_content_flags = text_content_flags.to(self.device, non_blocking=True)  # Move to device

                # Apply masking
                images, text_embeddings, modality_mask_info = self._apply_masking(
                    images, text_embeddings, self.current_epoch
                )

                # Use pre-computed flag (only if needed)
                if text_aux_enabled:
                    if modality_mask_info is None:
                        modality_mask_info = {"text_valid_mask": text_content_flags}
                    elif "text_valid_mask" in modality_mask_info:
                        modality_mask_info["text_valid_mask"] &= text_content_flags
                    else:
                        modality_mask_info["text_valid_mask"] = text_content_flags

                # Forward pass through vision model
                vision_embeddings = self.vision_model(images)

                # Forward pass through multimodal classifier
                outputs = self.multimodal_classifier(vision_embeddings, text_embeddings, modality_mask_info)
                logits = outputs["logits"]

                # Compute loss
                loss = self.multimodal_classifier.compute_loss(outputs, targets, modality_mask_info)

                # Statistics
                total_loss += loss.item()
                pred = logits.argmax(dim=1, keepdim=True)
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
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100. * correct / total:.2f}%"})

        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        with np.errstate(divide="ignore", invalid="ignore"):
            tp = np.diag(confusion).astype(np.float64)
            fp = confusion.sum(axis=0) - tp
            fn = confusion.sum(axis=1) - tp
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f1_per_class = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
            macro_f1 = float(np.mean(f1_per_class))

        logger.info("Test Results:")
        logger.info(f"  Test Loss: {avg_loss:.4f}")
        logger.info(f"  Test Accuracy: {accuracy:.2f}%")
        logger.info(f"  Test Macro-F1: {macro_f1:.4f}")
        logger.info(f"  Correct: {correct}/{total}")

        return {
            "test_loss": avg_loss,
            "test_accuracy": accuracy,
            "correct": correct,
            "total": total,
            "predictions": all_predictions,
            "targets": all_targets,
            "macro_f1": macro_f1,
        }

    def _save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint with all information needed for inference"""
        save_dir = self.output_config["save_dir"]
        experiment_name = self.output_config["experiment_name"]

        # ========================================================================
        # Get actual model configurations (with all runtime values filled in)
        # ========================================================================
        vision_model_info = self.vision_model.get_vision_info()
        multimodal_info = self.multimodal_classifier.get_model_info()

        # Build complete vision model config for inference
        vision_config = {
            "name": vision_model_info["model_name"],
            "pretrained": vision_model_info["pretrained"],
            "pooling_type": vision_model_info["pooling_type"],
            "img_size": self.info["img_size"],
            "unfreeze_layers": self.config["vision_model"].get("unfreeze_layers", 0),
        }

        # Build complete multimodal classifier config for inference
        multimodal_config = {
            # Runtime values (critical for inference)
            "num_classes": multimodal_info["num_classes"],
            "vision_embedding_dim": multimodal_info["vision_embedding_dim"],
            "text_embedding_dim": multimodal_info["text_embedding_dim"],
            # Architecture configuration
            "projection_dim": multimodal_info["projection_dim"],
            "fusion_strategy": multimodal_info["fusion_strategy"],
            "image_projection_hidden": multimodal_info["image_projection_hidden"],
            "text_projection_hidden": multimodal_info["text_projection_hidden"],
            "projection_activation": self.config["multimodal_classifier"].get("projection_activation", "relu"),
            "projection_dropout": self.config["multimodal_classifier"].get("projection_dropout", 0.2),
            "final_hidden_sizes": multimodal_info["final_hidden_sizes"],
            "final_activation": self.config["multimodal_classifier"].get("final_activation", "relu"),
            "final_dropout": self.config["multimodal_classifier"].get("final_dropout", 0.3),
            # Auxiliary loss configuration
            "use_auxiliary_loss": multimodal_info["use_auxiliary_loss"],
            "auxiliary_loss_weight": self.config["multimodal_classifier"].get("auxiliary_loss_weight", 0.3),
            # Loss function configuration
            "loss_fn": self.config["multimodal_classifier"]["loss_fn"],
            "label_smoothing": self.config["multimodal_classifier"].get("label_smoothing", 0.0),
        }

        # Add focal loss parameters if using focal loss
        if multimodal_config["loss_fn"] == "focal":
            multimodal_config["gamma"] = self.config["multimodal_classifier"].get("gamma", 2.0)
            multimodal_config["reduction"] = self.config["multimodal_classifier"].get("reduction", "mean")

        # Build complete encoder config for inference (text embedding)
        encoder_config = {
            "model_name": self.config["encoder"]["model_name"],
            "max_length": self.config["encoder"]["max_length"],
            "pooling_type": self.config["encoder"]["pooling_type"],
            "qwen_instr": self.config["encoder"].get("qwen_instr", ""),
        }

        # ========================================================================
        # Create modified config with actual runtime values
        # ========================================================================
        config_to_save = {
            # Inference-critical configs (with runtime values)
            "vision_model": vision_config,
            "multimodal_classifier": multimodal_config,
            "encoder": encoder_config,
            # Keep original configs for reference/resuming training
            "data": self.config["data"],
            "training": self.config["training"],
            "augmentation": self.config["augmentation"],
            "optimizer": self.config["optimizer"],
            "output": self.config["output"],
            "checkpoint": self.config["checkpoint"],
        }

        # ========================================================================
        # Prepare additional info (training state + inference metadata)
        # ========================================================================
        additional_info = {
            # Training state (for resuming training)
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "vision_frozen": self.vision_frozen,
            "n_warmup_epochs": self.n_warmup_epochs,
            "device": str(self.device),
            # Inference metadata (critical for proper inference)
            "normalization_mean": self.info["mean"],
            "normalization_std": self.info["std"],
            "classes": self.info["classes"],
            "class_to_idx": self.info["class_to_idx"],
            "num_classes": self.info["num_classes"],
            # Add current auxiliary loss weights for proper resuming
            "current_aux_vision_weight": self.multimodal_classifier.auxiliary_vision_loss_weight,
            "current_aux_text_weight": self.multimodal_classifier.auxiliary_text_loss_weight,
        }

        # Include optimizer states (for resuming training)
        if self.optimizer_vision is not None and self.optimizer_multimodal is not None:
            additional_info["optimizers_state_dict"] = {
                "vision": self.optimizer_vision.state_dict(),
                "multimodal_classifier": self.optimizer_multimodal.state_dict(),
            }

        # Include scheduler states (for resuming training)
        if self.scheduler_vision is not None or self.scheduler_multimodal is not None:
            additional_info["schedulers_state_dict"] = {}
            if self.scheduler_vision is not None:
                additional_info["schedulers_state_dict"]["vision"] = self.scheduler_vision.state_dict()
            if self.scheduler_multimodal is not None:
                additional_info["schedulers_state_dict"][
                    "multimodal_classifier"
                ] = self.scheduler_multimodal.state_dict()

        # ========================================================================
        # Save checkpoint with complete config
        # ========================================================================
        checkpoint_path = save_checkpoint(
            model=None,
            optimizer=None,
            epoch=epoch,
            loss=loss,
            config=config_to_save,  # Complete config with all runtime values
            save_dir=save_dir,
            experiment_name=experiment_name,
            is_best=is_best,
            additional_info=additional_info,
            vision_model=self.vision_model,
            multimodal_classifier=self.multimodal_classifier,
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
            keep_last = self.checkpoint_config.get("keep_last", 5)
            cleanup_old_checkpoints(save_dir, experiment_name, keep_last)

        return checkpoint_path
