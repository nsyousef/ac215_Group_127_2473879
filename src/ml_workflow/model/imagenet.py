"""ImageNet-based transfer learning model for skin disease classification"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    ResNet101_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B4_Weights,
    VGG16_Weights,
)
from typing import Dict, Any, Optional, List
from .utils import MLP, FocalLoss

class ImageNetModel(nn.Module):
    """Transfer learning model based on ImageNet pretrained architectures"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize ImageNet-based transfer learning model
        
        Args:
            model_config: Dictionary containing model configuration parameters
        """
        super().__init__()
        
        # Extract configuration parameters
        self.model_name = model_config['name']
        self.num_classes = model_config['num_classes']
        self.pretrained = model_config['pretrained']
        self.hidden_sizes = model_config['hidden_sizes']
        self.activation = model_config['activation']
        self.dropout_rate = model_config['dropout_rate']
        self.img_size = model_config['img_size']
        self.pooling_type = model_config.get('pooling_type')  # 'avg', 'max', 'concat'
        self.unfreeze_layers = model_config.get('unfreeze_layers', 0)  # Number of layers to unfreeze from the end
        
        # Initialize loss function with label smoothing
        self.loss_fn = model_config['loss_fn']
        if self.loss_fn == 'cross_entropy':
            self.label_smoothing = model_config.get('label_smoothing', 0.0)
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        elif self.loss_fn == 'focal':
            if 'alpha' in model_config:
                self.alpha = model_config['alpha']
            else:
                self.alpha = model_config['sample_weights']
            self.gamma = model_config['gamma']
            self.reduction = model_config['reduction']
            self.loss_fn = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")    
               
        self.backbone = self._get_backbone(self.model_name, self.pretrained)
        
        # Apply layer freezing/unfreezing
        self._freeze_layers()
        
        # Add pooling layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.num_features = self._get_num_features()
        
        self.classifier = MLP(
            input_size=self.num_features,
            hidden_sizes=self.hidden_sizes,
            output_size=self.num_classes,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        )
        
    def _get_backbone(self, model_name: str, pretrained: bool) -> nn.Module:
        """Get the pretrained backbone model"""
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
            # Remove the original classifier
            model.features = model.features
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
    
    def _freeze_layers(self):
        """
        Freeze layers in the backbone based on unfreeze_layers parameter
        unfreeze_layers: 0 means all layers frozen, -1 means all unfrozen,
                        positive number means unfreeze that many individual layers from the end
        """
        if self.unfreeze_layers == -1:
            # Unfreeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = True
            return
        
        # First freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        if self.unfreeze_layers == 0:
            # Keep all layers frozen
            return
        
        # Get all modules/layers in the backbone (excluding the Sequential wrapper itself)
        all_modules = list(self.backbone.modules())[1:]  # Skip the Sequential wrapper
        
        # Filter to only modules that have parameters (learnable layers)
        # This excludes things like ReLU, pooling layers without params, etc.
        learnable_modules = [m for m in all_modules if len(list(m.parameters())) > 0]
        
        # Calculate number of layers to unfreeze
        num_learnable_layers = len(learnable_modules)
        if self.unfreeze_layers > num_learnable_layers:
            print(f"Warning: unfreeze_layers ({self.unfreeze_layers}) is greater than total learnable layers ({num_learnable_layers}). Unfreezing all layers.")
            modules_to_unfreeze = learnable_modules
        else:
            modules_to_unfreeze = learnable_modules[-self.unfreeze_layers:]
        
        # Unfreeze the specified layers
        for module in modules_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True
        
        # Print freeze status
        trainable_count = sum(1 for p in self.backbone.parameters() if p.requires_grad)
        total_count = sum(1 for p in self.backbone.parameters())
        print(f"Backbone: {num_learnable_layers} learnable layers, {len(modules_to_unfreeze)} unfrozen layers ({trainable_count}/{total_count} parameters trainable)")
    
    def _get_num_features(self) -> int:
        """Get the number of output features from the backbone"""
        # Create a dummy input to get the feature size
        dummy_input = torch.randn(1, 3, self.img_size[0], self.img_size[1])
        with torch.no_grad():
            features = self.backbone(dummy_input)
            
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
            
            return pooled.view(pooled.size(0), -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        """
        # Extract features using backbone
        features = self.backbone(x)
        
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
        
        # Flatten features
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        
        return output
    
    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between predictions and targets
        """
        return self.loss_fn(outputs, targets)
    
    def unfreeze_more_layers(self, num_additional_layers: int):
        """
        Unfreeze additional layers during training (for progressive unfreezing)
        
        Args:
            num_additional_layers: Number of additional layers to unfreeze from the end
        """
        self.unfreeze_layers += num_additional_layers
        self._freeze_layers()
        print(f"Unfroze {num_additional_layers} additional layers. Total unfrozen: {self.unfreeze_layers}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'pooling_type': self.pooling_type,
            'unfreeze_layers': self.unfreeze_layers,
            'num_features': self.num_features,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': backbone_params,
            'backbone_trainable_parameters': backbone_trainable,
            'trainable_percentage': f"{(trainable_params/total_params)*100:.2f}%"
        }