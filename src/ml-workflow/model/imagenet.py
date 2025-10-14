"""ImageNet-based transfer learning model for skin disease classification"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Any, Optional, List
from .utils import MLP


class ImageNetModel(nn.Module):
    """Transfer learning model based on ImageNet pretrained architectures"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize ImageNet-based transfer learning model
        
        Args:
            model_config: Dictionary containing model configuration parameters
        """
        super(ImageNetModel, self).__init__()
        
        # Extract configuration parameters
        self.model_name = model_config['name']
        self.num_classes = model_config['num_classes']
        self.pretrained = model_config['pretrained']
        self.hidden_sizes = model_config['hidden_sizes']
        self.activation = model_config['activation']
        self.dropout_rate = model_config['dropout_rate']
        
        # Initialize loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Load pretrained backbone
        self.backbone = self._get_backbone(self.model_name, self.pretrained)
        
        # Get the number of features from the backbone
        self.num_features = self._get_num_features()
        
        # Replace classifier with custom MLP
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
            model = models.resnet50(pretrained=pretrained)
            # Remove the original classifier
            model = nn.Sequential(*list(model.children())[:-1])
        elif model_name == "resnet101":
            model = models.resnet101(pretrained=pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
        elif model_name == "densenet121":
            model = models.densenet121(pretrained=pretrained)
            # Remove the original classifier
            model = nn.Sequential(*list(model.children())[:-1])
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
        elif model_name == "vgg16":
            model = models.vgg16(pretrained=pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        return model
    
    def _get_num_features(self) -> int:
        """Get the number of output features from the backbone"""
        # Create a dummy input to get the feature size
        dummy_input = torch.randn(1, 3, self.img_size[0], self.img_size[1])
        with torch.no_grad():
            features = self.backbone(dummy_input)
            return features.view(features.size(0), -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        # Pass through classifier
        output = self.classifier(features)
        
        return output
    
    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between predictions and targets
        
        Args:
            outputs: Model predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Loss tensor
        """
        return self.loss_fn(outputs, targets)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'num_features': self.num_features,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
