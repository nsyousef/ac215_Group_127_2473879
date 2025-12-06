"""
Vision + text embedding classifier for inference.
Loads checkpoint, computes text embeddings, and takes vision/text embeddings to return skin condition predictions.
"""

import torch
from typing import Union, List, Dict
import numpy as np
from transformers import AutoTokenizer, AutoModel

from ml_workflow.model.classifier.multimodal_classifier import MultimodalClassifier
from ml_workflow.dataloader.embedding_utils import mean_pool, cls_pool, last_token_pool, get_detailed_instruct
from ml_workflow.constants import MODELS
from ml_workflow.utils import logger


class InferenceClassifier:
    """Multimodal classifier for inference with vision and text embeddings."""

    def __init__(self, checkpoint_path: str = None, device: str = "cuda"):
        """
        Initialize classifier from checkpoint or with random weights.

        Args:
            checkpoint_path: Path to trained model checkpoint (None for baseline random initialization)
            device: Device to use (cuda/cpu)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # If no checkpoint provided, initialize with default baseline configuration
        if checkpoint_path is None:
            logger.info("No checkpoint provided, initializing baseline classifier with random weights")
            self._init_baseline()
            return

        # Load from checkpoint
        logger.info(f"Loading multimodal classifier from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract config
        config = checkpoint.get("config", {})
        if not config:
            raise ValueError("No config found in checkpoint")

        multimodal_config = config.get("multimodal_classifier")
        if multimodal_config is None:
            raise ValueError(
                f"No 'multimodal_classifier' key found in checkpoint config. " f"Available keys: {list(config.keys())}"
            )

        # Get encoder config for text embeddings
        encoder_config = config.get("encoder")
        if encoder_config is None:
            raise ValueError(f"No 'encoder' key found in checkpoint config. " f"Available keys: {list(config.keys())}")

        # Get class information from checkpoint
        self.classes = checkpoint.get("classes")
        self.class_to_idx = checkpoint.get("class_to_idx")

        if self.classes is None:
            logger.warning("No class information found in checkpoint")
            # Will be indices only
            self.classes = [str(i) for i in range(multimodal_config["num_classes"])]
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Create multimodal classifier
        logger.info(f"Creating multimodal classifier with config: {multimodal_config}")
        self.model = MultimodalClassifier(multimodal_config)

        # Load weights
        if "multimodal_classifier_state_dict" not in checkpoint:
            raise ValueError(
                f"No 'multimodal_classifier_state_dict' found in checkpoint. "
                f"Available keys: {list(checkpoint.keys())}"
            )

        state_dict = checkpoint["multimodal_classifier_state_dict"]
        self.model.load_state_dict(state_dict, strict=True)

        self.model.to(self.device)
        self.model.eval()

        # Initialize text encoder
        self._init_text_encoder(encoder_config)

        logger.info("Multimodal classifier loaded successfully")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Number of classes: {len(self.classes)}")
        logger.info(f"  Fusion strategy: {multimodal_config['fusion_strategy']}")
        logger.info(f"  Text encoder: {self.text_model_name}")
        logger.info(f"  Text pooling: {self.pooling_strategy}")

    def _init_baseline(self):
        """Initialize baseline classifier with random weights and default configuration."""
        # Default class names (57 skin conditions as indices)
        num_classes = 57
        self.classes = [f"condition_{i}" for i in range(num_classes)]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Create multimodal classifier with random weights
        logger.info("Creating baseline multimodal classifier with random weights")
        self.model = MultimodalClassifier.create_random(
            vision_embedding_dim=2048,
            text_embedding_dim=768,
            num_classes=num_classes,
            projection_dim=256,
            fusion_strategy="concat_mlp",
        )

        self.model.to(self.device)
        self.model.eval()

        # Default encoder config (use PubMedBERT)
        default_encoder_config = {
            "model_name": "pubmedbert",
            "max_length": 512,
            "pooling_type": "cls",
            "qwen_instr": "",
        }

        # Initialize text encoder
        self._init_text_encoder(default_encoder_config)

        logger.info("Baseline classifier initialized successfully")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Number of classes: {len(self.classes)}")
        logger.info("  Fusion strategy: concat_mlp")
        logger.info(f"  Text encoder: {self.text_model_name}")
        logger.info(f"  Text pooling: {self.pooling_strategy}")

    def _init_text_encoder(self, encoder_config: dict):
        """Initialize text encoder from config."""
        model_name = encoder_config["model_name"]
        self.max_length = encoder_config.get("max_length", 512)
        self.pooling_strategy = encoder_config.get("pooling_type", "cls")
        self.qwen_instr = encoder_config.get("qwen_instr", "")

        # Convert short name to full model name if needed
        if model_name in MODELS:
            model_name = MODELS[model_name]

        self.text_model_name = model_name

        # Validate pooling strategy
        if "qwen" in model_name.lower():
            if self.pooling_strategy != "last_token":
                raise ValueError("For QWEN models, only 'last_token' pooling is supported.")
        else:
            if self.pooling_strategy == "last_token":
                raise ValueError("'last_token' pooling is only intended for QWEN models.")

        logger.info(f"Loading text encoder: {model_name}")

        # Set padding side based on pooling strategy
        padding_side = "left" if self.pooling_strategy == "last_token" else "right"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.padding_side = padding_side

        self.text_model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        self.text_model.to(self.device)
        self.text_model.eval()

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed text description using the same encoder as training.

        Args:
            text: Text description (e.g., patient description, symptoms)

        Returns:
            Text embedding as numpy array (embedding_dim,)
        """
        # Handle missing or empty text
        if not text or text.strip() == "":
            logger.warning("Empty text provided, using [MISSING] token")
            text = "[MISSING]"

        # Format with instruction for QWEN models
        if self.pooling_strategy == "last_token":
            text = get_detailed_instruct(self.qwen_instr, text)

        # Tokenize
        encoded = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        # Move to device
        for k in encoded:
            encoded[k] = encoded[k].to(self.device)

        # Get embeddings
        outputs = self.text_model(**encoded)
        last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        attention_mask = encoded["attention_mask"]

        # Apply pooling
        if self.pooling_strategy == "last_token":
            pooled = last_token_pool(last_hidden, attention_mask)
        elif self.pooling_strategy == "cls":
            pooled = cls_pool(last_hidden, attention_mask)
        elif self.pooling_strategy == "mean":
            pooled = mean_pool(last_hidden, attention_mask)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Convert to numpy and remove batch dimension
        embedding = pooled.cpu().numpy().squeeze()

        return embedding

    @torch.no_grad()
    def predict(
        self,
        vision_embedding: np.ndarray,
        text_embedding: np.ndarray,
        return_probs: bool = True,
        top_k: int = 5,
        confidence_threshold: float = 0.5,
        min_gap_threshold: float = 0.1,
        max_entropy_threshold: float = 0.8,
    ) -> Dict[str, Union[str, float, List[tuple], bool]]:
        """
        Predict skin condition from embeddings.

        Args:
            vision_embedding: Vision embedding vector (embedding_dim,)
            text_embedding: Text embedding vector (embedding_dim,)
            return_probs: Whether to return probabilities
            top_k: Number of top predictions to return
            confidence_threshold: Minimum confidence to consider prediction reliable (0-1)
            min_gap_threshold: Minimum probability gap between top 2 predictions (0-1)
            max_entropy_threshold: Maximum normalized entropy allowed (0-1). Higher = more uniform allowed

        Returns:
            Dictionary with:
                - 'predicted_class': Most likely class name (or "UNCERTAIN" if not confident)
                - 'predicted_idx': Class index (or -1 if uncertain)
                - 'confidence': Confidence score (0-1)
                - 'is_confident': Boolean indicating if prediction meets confidence criteria
                - 'uncertainty_reason': Reason for uncertainty (if applicable)
                - 'top_k': List of (class_name, probability) tuples (if return_probs=True)
        """
        # Convert to tensors and add batch dimension
        vision_tensor = torch.from_numpy(vision_embedding).float().unsqueeze(0).to(self.device)
        text_tensor = torch.from_numpy(text_embedding).float().unsqueeze(0).to(self.device)

        # Forward pass
        outputs = self.model(vision_tensor, text_tensor)
        logits = outputs["logits"]

        # Get probabilities
        probs = torch.softmax(logits, dim=1)

        # Get top prediction
        top_prob, top_idx = probs.max(dim=1)
        predicted_idx = top_idx.item()
        predicted_class = self.idx_to_class[predicted_idx]
        confidence = top_prob.item()

        # Check confidence criteria
        is_confident = True
        uncertainty_reasons = []

        # Check 1: Maximum probability threshold
        if confidence < confidence_threshold:
            is_confident = False
            uncertainty_reasons.append(f"max_probability_too_low ({confidence:.3f} < {confidence_threshold})")

        # Check 2: Gap between top 2 predictions (ambiguity check)
        if len(self.classes) > 1:
            top2_probs, top2_indices = probs.topk(2, dim=1)
            prob_gap = (top2_probs[0, 0] - top2_probs[0, 1]).item()

            if prob_gap < min_gap_threshold:
                is_confident = False
                uncertainty_reasons.append(f"predictions_too_close (gap={prob_gap:.3f} < {min_gap_threshold})")

        # Check 3: High entropy (uniform distribution indicates uncertainty)
        entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=1).item()  # Fixed epsilon
        max_entropy = np.log(len(self.classes))  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Check if entropy exceeds threshold (configurable now!)
        if normalized_entropy > max_entropy_threshold:
            is_confident = False
            uncertainty_reasons.append(f"high_entropy (normalized={normalized_entropy:.3f} > {max_entropy_threshold})")

        result = {
            "predicted_class": predicted_class if is_confident else "UNCERTAIN",
            "predicted_idx": predicted_idx if is_confident else -1,
            "confidence": confidence,
            "is_confident": is_confident,
            "uncertainty_reason": "; ".join(uncertainty_reasons) if uncertainty_reasons else None,
            "top_prediction": predicted_class,  # Always include actual top prediction
            "entropy": normalized_entropy,
        }

        if return_probs:
            # Get top-k predictions
            top_k = min(top_k, len(self.classes))
            topk_probs, topk_indices = probs.topk(top_k, dim=1)

            top_predictions = [
                (self.idx_to_class[idx.item()], prob.item()) for idx, prob in zip(topk_indices[0], topk_probs[0])
            ]
            result["top_k"] = top_predictions

        return result

    def predict_as_dict(
        self, vision_embedding: np.ndarray, text_embedding: np.ndarray, top_k: int = 5
    ) -> Dict[str, float]:
        """
        Predict and return results in simple dict format (class_name -> probability).
        Compatible with _run_cloud_ml_model() format.

        Args:
            vision_embedding: Vision embedding vector
            text_embedding: Text embedding vector
            top_k: Number of top predictions to return

        Returns:
            Dictionary mapping disease names to probabilities
        """
        result = self.predict(
            vision_embedding=vision_embedding, text_embedding=text_embedding, return_probs=True, top_k=top_k
        )

        # Convert to simple dict format
        return {class_name: prob for class_name, prob in result["top_k"]}

    def get_class_names(self) -> List[str]:
        """Get list of all class names."""
        return self.classes

    def get_model_info(self) -> dict:
        """Get information about the loaded classifier."""
        model_info = self.model.get_model_info()
        model_info["device"] = str(self.device)
        model_info["classes"] = self.classes
        model_info["num_classes"] = len(self.classes)
        model_info["text_encoder"] = self.text_model_name
        model_info["text_pooling"] = self.pooling_strategy
        model_info["max_length"] = self.max_length
        return model_info
