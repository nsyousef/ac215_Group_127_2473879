import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

try:
    from ..utils import MLP, FocalLoss
except ImportError:
    from ml_workflow.model.utils import MLP, FocalLoss


class MultimodalClassifier(nn.Module):
    """
    Multimodal classifier that fuses vision and text embeddings through projection layers.

    Supports two fusion strategies:
    1. weighted_sum: Learnable weighted combination of projected embeddings
    2. concat_mlp: Concatenation of projected embeddings followed by MLP

    Optional auxiliary losses on individual modality predictions before fusion.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MultimodalClassifier

        Args:
            config:
                Required keys:
                    - vision_embedding_dim: Size of vision embeddings from vision model
                    - text_embedding_dim: Size of text embeddings from dataloader
                    - num_classes: Number of output classes
                    - projection_dim: Output dimension for both projections
                    - fusion_strategy: 'weighted_sum' or 'concat_mlp'
                    - loss_fn: Loss function type ('cross_entropy' or 'focal')
                Optional keys:
                    - image_projection_hidden: List of hidden layer sizes for image projection (default: [])
                    - text_projection_hidden: List of hidden layer sizes for text projection (default: [])
                    - projection_activation: Activation function for projections (default: 'relu')
                    - projection_dropout: Dropout rate for projections (default: 0.2)
                    - final_hidden_sizes: List of hidden layer sizes for final classifier (default: [])
                    - final_activation: Activation function for final classifier (default: 'relu')
                    - final_dropout: Dropout rate for final classifier (default: 0.3)
                    - use_auxiliary_loss: Enable auxiliary losses on individual modalities (default: False)
                    - auxiliary_loss_weight: Weight for auxiliary losses (default: 0.3)
                    - label_smoothing: Label smoothing for cross entropy (default: 0.0)
                    - alpha, gamma, reduction: For focal loss
                    - sample_weights: Sample weights for focal loss
        """
        super().__init__()

        # Extract configuration
        self.vision_embedding_dim = config["vision_embedding_dim"]
        self.text_embedding_dim = config["text_embedding_dim"]
        self.num_classes = config["num_classes"]
        self.projection_dim = config["projection_dim"]
        self.fusion_strategy = config["fusion_strategy"]
        self.use_l2_normalization = config.get("use_l2_normalization", False)

        # Projection layer configuration
        self.image_projection_hidden = config.get("image_projection_hidden", [])
        self.text_projection_hidden = config.get("text_projection_hidden", [])
        self.projection_activation = config.get("projection_activation", "relu")
        self.projection_dropout = config.get("projection_dropout", 0.2)
        # Allow separate dropout for text projection (falls back to projection_dropout if not specified)
        self.text_projection_dropout = config.get("text_projection_dropout", self.projection_dropout)

        # Final classifier configuration
        self.final_hidden_sizes = config.get("final_hidden_sizes", [])
        self.final_activation = config.get("final_activation", "relu")
        self.final_dropout = config.get("final_dropout", 0.3)

        # Auxiliary loss configuration
        self.use_auxiliary_loss = config.get("use_auxiliary_loss", False)
        self.auxiliary_loss_weight = config.get("auxiliary_loss_weight", 0.3)

        # Validate fusion strategy
        if self.fusion_strategy not in ["weighted_sum", "concat_mlp"]:
            raise ValueError(
                f"Unsupported fusion strategy: {self.fusion_strategy}. Must be 'weighted_sum' or 'concat_mlp'"
            )

        # Build projection layers
        self.vision_projection = self._build_projection(
            input_dim=self.vision_embedding_dim,
            hidden_sizes=self.image_projection_hidden,
            output_dim=self.projection_dim,
            activation=self.projection_activation,
            dropout_rate=self.projection_dropout,
        )

        self.text_projection = self._build_projection(
            input_dim=self.text_embedding_dim,
            hidden_sizes=self.text_projection_hidden,
            output_dim=self.projection_dim,
            activation=self.projection_activation,
            dropout_rate=self.text_projection_dropout,
        )

        # Build fusion mechanism
        if self.fusion_strategy == "weighted_sum":
            # Learnable scalar weights (gates) for weighted fusion
            self.alpha_vision = nn.Parameter(torch.tensor(0.5))
            self.alpha_text = nn.Parameter(torch.tensor(0.5))

            # Final classifier takes projection_dim as input
            final_input_dim = self.projection_dim
        else:  # concat_mlp
            # Final classifier takes concatenated embeddings
            final_input_dim = self.projection_dim * 2

        # Build final classifier
        if len(self.final_hidden_sizes) > 0:
            self.final_classifier = MLP(
                input_size=final_input_dim,
                hidden_sizes=self.final_hidden_sizes,
                output_size=self.num_classes,
                activation=self.final_activation,
                dropout_rate=self.final_dropout,
            )
        else:
            self.final_classifier = nn.Linear(final_input_dim, self.num_classes)

        # Auxiliary classifiers (optional)
        if self.use_auxiliary_loss:
            self.aux_vision_classifier = nn.Linear(self.projection_dim, self.num_classes)
            self.aux_text_classifier = nn.Linear(self.projection_dim, self.num_classes)

        # Loss function
        self.loss_fn_name = config["loss_fn"]
        if self.loss_fn_name == "cross_entropy":
            self.label_smoothing = config.get("label_smoothing", 0.0)
            class_weights = config.get("class_weights")
            if class_weights is not None:
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
                self.register_buffer("class_weights_tensor", weight_tensor)
            else:
                self.class_weights_tensor = None
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor, label_smoothing=self.label_smoothing)
        elif self.loss_fn_name == "focal":
            if "alpha" in config:
                self.alpha = config["alpha"]
            elif "class_weights" in config and config["class_weights"] is not None:
                self.alpha = config["class_weights"]
            else:
                self.alpha = 1.0
            self.gamma = config.get("gamma", 2.0)
            self.reduction = config.get("reduction", "mean")
            self.criterion = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn_name}")

    def _build_projection(
        self, input_dim: int, hidden_sizes: list, output_dim: int, activation: str, dropout_rate: float
    ) -> nn.Module:
        """Build a projection network (MLP or linear layer)"""
        if len(hidden_sizes) > 0:
            return MLP(
                input_size=input_dim,
                hidden_sizes=hidden_sizes,
                output_size=output_dim,
                activation=activation,
                dropout_rate=dropout_rate,
            )
        else:
            return nn.Linear(input_dim, output_dim)

    def forward(self, vision_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multimodal classifier

        Args:
            vision_embeddings: Vision embeddings from vision model (batch_size, vision_embedding_dim)
            text_embeddings: Pre-computed text embeddings from dataloader (batch_size, text_embedding_dim)

        Returns:
            Dictionary with keys:
                - 'logits': Main prediction logits (batch_size, num_classes)
                - 'aux_vision_logits': Auxiliary vision logits (if use_auxiliary_loss=True)
                - 'aux_text_logits': Auxiliary text logits (if use_auxiliary_loss=True)
        """

        # Project embeddings to common dimension
        projected_vision = self.vision_projection(vision_embeddings)
        projected_text = self.text_projection(text_embeddings)

        if self.use_l2_normalization:
            projected_vision = F.normalize(projected_vision, p=2, dim=1)
            projected_text = F.normalize(projected_text, p=2, dim=1)

        # Fuse embeddings based on strategy
        if self.fusion_strategy == "weighted_sum":
            # Normalize weights using softmax to ensure they sum to 1
            weights = torch.softmax(torch.stack([self.alpha_vision, self.alpha_text]), dim=0)
            fused = weights[0] * projected_vision + weights[1] * projected_text
        else:  # concat_mlp
            fused = torch.cat([projected_vision, projected_text], dim=1)

        # Final classification
        logits = self.final_classifier(fused)

        # Prepare output dictionary
        outputs = {"logits": logits}

        # Auxiliary predictions (optional)
        if self.use_auxiliary_loss:
            outputs["aux_vision_logits"] = self.aux_vision_classifier(projected_vision)
            outputs["aux_text_logits"] = self.aux_text_classifier(projected_text)

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        modality_mask_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute loss with optional auxiliary losses

        Args:
            outputs: Dictionary from forward() containing logits
            targets: Ground truth labels (batch_size,)

        Returns:
            Total loss (scalar tensor)
        """
        # Main loss
        main_loss = self.criterion(outputs["logits"], targets)

        # Auxiliary losses (optional)
        if self.use_auxiliary_loss and "aux_vision_logits" in outputs:
            # Build per-modality masks (1 = valid sample for that modality)
            vision_mask = None
            text_mask = None

            if modality_mask_info is not None:
                vision_mask = modality_mask_info.get("vision_valid_mask")
                text_mask = modality_mask_info.get("text_valid_mask")

            aux_losses = []
            aux_total = None

            # Vision auxiliary loss only if modality is valid
            if vision_mask is None:
                loss_term = self.criterion(outputs["aux_vision_logits"], targets)
                aux_total = loss_term if aux_total is None else aux_total + loss_term
            elif vision_mask.any():
                loss_term = self._compute_masked_loss(outputs["aux_vision_logits"], targets, vision_mask)
                aux_total = loss_term if aux_total is None else aux_total + loss_term

            # Text auxiliary loss only if modality is valid
            if text_mask is None:
                loss_term = self.criterion(outputs["aux_text_logits"], targets)
                aux_total = loss_term if aux_total is None else aux_total + loss_term
            elif text_mask.any():
                loss_term = self._compute_masked_loss(outputs["aux_text_logits"], targets, text_mask)
                aux_total = loss_term if aux_total is None else aux_total + loss_term

            if aux_total is not None:
                total_loss = main_loss + self.auxiliary_loss_weight * aux_total
            else:
                # Both modalities fully masked, use only main loss
                total_loss = main_loss
        else:
            total_loss = main_loss

        return total_loss

    def _compute_masked_loss(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute loss only on samples where mask==1"""
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool)
        if mask.any():
            masked_logits = logits[mask]
            masked_targets = targets[mask]
            return self.criterion(masked_logits, masked_targets)
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including parameter counts"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count parameters by component
        vision_proj_params = sum(p.numel() for p in self.vision_projection.parameters())
        text_proj_params = sum(p.numel() for p in self.text_projection.parameters())
        final_classifier_params = sum(p.numel() for p in self.final_classifier.parameters())

        info = {
            "vision_embedding_dim": self.vision_embedding_dim,
            "text_embedding_dim": self.text_embedding_dim,
            "num_classes": self.num_classes,
            "projection_dim": self.projection_dim,
            "fusion_strategy": self.fusion_strategy,
            "image_projection_hidden": self.image_projection_hidden,
            "text_projection_hidden": self.text_projection_hidden,
            "final_hidden_sizes": self.final_hidden_sizes,
            "use_auxiliary_loss": self.use_auxiliary_loss,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": f"{(trainable_params/total_params)*100:.2f}%",
            "vision_projection_parameters": vision_proj_params,
            "text_projection_parameters": text_proj_params,
            "final_classifier_parameters": final_classifier_params,
        }

        if self.fusion_strategy == "weighted_sum":
            info["fusion_weights"] = {"alpha_vision": self.alpha_vision.item(), "alpha_text": self.alpha_text.item()}

        return info
