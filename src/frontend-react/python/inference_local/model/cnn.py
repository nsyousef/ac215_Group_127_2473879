import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet101_Weights, DenseNet121_Weights, EfficientNet_B0_Weights, EfficientNet_B4_Weights, VGG16_Weights
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ..logger import logger

class CNNModel(nn.Module):
    """CNN-based transfer learning model that returns embeddings only"""
    
    def __init__(self, vision_config: Dict[str, Any]):
        """
        Initialize CNN-based transfer learning model
        
        Args:
            vision_config: Dictionary containing vision model configuration parameters
                Required keys:
                    - name: Model variant ('resnet50', 'resnet101', 'densenet121', 'efficientnet_b0', 'efficientnet_b4', 'vgg16')
                    - pretrained: Whether to use pretrained weights
                Optional keys:
                    - img_size: Input image size as tuple (default: (224, 224))
                    - pooling_type: Pooling type ('avg', 'max', 'concat') (default: 'avg')
                    - unfreeze_layers: Number of layers to unfreeze from end (default: 0)
        """
        super().__init__()
        
        self.model_name = vision_config['name']
        self.pretrained = vision_config['pretrained']
        self.img_size = tuple(vision_config.get('img_size', (224, 224)))
        self.pooling_type = vision_config.get('pooling_type', 'avg')
        self.unfreeze_layers = vision_config.get('unfreeze_layers', 0)
        
        self.model = self._get_model(self.model_name, self.pretrained) # pre trained part
        
        # Apply layer freezing/unfreezing
        self._freeze_layers()
        
        # Add pooling layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.embedding_dim = self._get_embedding_dim()
        
    def _get_model(self, model_name: str, pretrained: bool) -> nn.Module:
        """Get the pretrained model"""
        if model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            # Remove the original classifier and pooling
            model = nn.Sequential(*list(model.children())[:-2])
        elif model_name == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            model = models.resnet101(weights=weights)
            model = nn.Sequential(*list(model.children())[:-2])
        elif model_name == "densenet121":
            weights = DenseNet121_Weights.DEFAULT if pretrained else None
            model = models.densenet121(weights=weights)
            # Remove the original classifier, keep only features
            model = model.features
        elif model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            # Remove classifier and pooling
            model = model.features
        elif model_name == "efficientnet_b4":
            weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b4(weights=weights)
            # Remove classifier and pooling
            model = model.features
        elif model_name == "vgg16":
            weights = VGG16_Weights.DEFAULT if pretrained else None
            model = models.vgg16(weights=weights)
            model = model.features
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        return model
    
    def _freeze_layers(self, warmup_mode: bool = False):
        """
        Freeze layers in the model based on unfreeze_layers parameter
        
        Args:
            warmup_mode: If True, freeze ALL layers (for warmup phase)
                         If False, use self.unfreeze_layers setting
        
        unfreeze_layers: 
            - 0: All layers frozen
            - -1: All layers unfrozen
            - positive number: Unfreeze that many layers from the end
        """
        # Warmup phase: freeze everything
        if warmup_mode:
            for param in self.model.parameters():
                param.requires_grad = False
            return

        # Normal training: use unfreeze_layers setting
        if self.unfreeze_layers == -1:
            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True
            return
        
        # Partial unfreezing
        # First freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        if self.unfreeze_layers == 0:
            # Keep all layers frozen
            return
        
        # Get direct children (layers)
        layers = list(self.model.children())
        total_layers = len(layers)
        
        # Determine which layers to unfreeze
        if self.unfreeze_layers > total_layers:
            logger.warning(f"unfreeze_layers ({self.unfreeze_layers}) > total layers ({total_layers}). Unfreezing all layers.")
            layers_to_unfreeze = layers
        else:
            layers_to_unfreeze = layers[-self.unfreeze_layers:]
        
        # Unfreeze the specified layers
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Log freeze status
        trainable_layers = sum(1 for layer in layers if any(p.requires_grad for p in layer.parameters()))
        logger.info(f"Vision model: {total_layers} total layers, {trainable_layers} trainable layers")
    
    def _get_embedding_dim(self) -> int:
        """Get the number of output features from the model"""
        # Save original training state
        was_training = self.model.training
        self.model.eval()  # Set to eval mode to use running stats for BatchNorm
        
        # Get device from model parameters
        device = next(self.model.parameters()).device
        
        # Create dummy input on same device as model
        dummy_input = torch.randn(1, 3, self.img_size[0], self.img_size[1], device=device)
        with torch.no_grad():
            features = self.model(dummy_input)
            # Apply pooling based on pooling_type
            if self.pooling_type == 'avg':
                pooled = self.avg_pool(features)
            elif self.pooling_type == 'max':
                pooled = self.max_pool(features)
            elif self.pooling_type == 'concat':
                avg_pooled = self.avg_pool(features)
                max_pooled = self.max_pool(features)
                pooled = torch.cat([avg_pooled, max_pooled], dim=1)
            else:
                raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
            embedding_dim = pooled.view(pooled.size(0), -1).size(1)
        
        # Restore original training state
        if was_training:
            self.model.train()
            
        return embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output embeddings of shape (batch_size, embedding_dim)
        """
        # Extract features using model
        features = self.model(x)
        
        # Apply pooling based on pooling_type
        if self.pooling_type == 'avg':
            features = self.avg_pool(features)
        elif self.pooling_type == 'max':
            features = self.max_pool(features)
        elif self.pooling_type == 'concat':
            avg_pooled = self.avg_pool(features)
            max_pooled = self.max_pool(features)
            features = torch.cat([avg_pooled, max_pooled], dim=1)
        else:
            features = features
        
        # Flatten features to get embeddings
        embeddings = features.view(features.size(0), -1)
        
        return embeddings
    
    def get_vision_info(self) -> Dict[str, Any]:
        """
        Get vision model information
        
        Returns:
            Dictionary containing vision model configuration and statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_params = sum(p.numel() for p in self.model.parameters())
        model_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'pretrained': self.pretrained,
            'pooling_type': self.pooling_type,
            'unfreeze_layers': self.unfreeze_layers,
            'embedding_dim': self.embedding_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_parameters': model_params,
            'model_trainable_parameters': model_trainable,
            'trainable_percentage': f"{(trainable_params/total_params)*100:.2f}%"
        }
    